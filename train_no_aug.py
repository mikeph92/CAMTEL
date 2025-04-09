import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader
import pandas as pd
import os
import wandb
from datasets import MultiTaskDataset
from models import UNIMultitask
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser(description='MultiTask Learning for Pathology Image Classification')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to tumor_dataset.csv')
    parser.add_argument('--cluster_path', type=str, required=True, help='Path to clustering result directory')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--classification_task', type=str, default='tumor', choices=['tumor', 'TIL'], help='Classification task')
    parser.add_argument('--testset', type=str, required=True, help='Test dataset name')
    parser.add_argument('--crop_size', type=int, default=96, help='Crop size for images')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train split ratio')
    parser.add_argument('--multitask', action='store_true', help='Enable multitask learning')
    parser.add_argument('--lambda_ortho', type=float, default=0.01, help='Weight for orthogonality loss')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--num_chunks', type=int, default=4, help='Number of chunks to split each batch into for more frequent updates')
    return parser.parse_args()

def setup_device(num_gpus):
    if torch.cuda.is_available() and num_gpus > 0:
        device = torch.device("cuda")
        available_gpus = min(torch.cuda.device_count(), num_gpus)
        print(f"Using {available_gpus} GPUs")
        if available_gpus < num_gpus:
            print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} are available")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def get_max_batch_size(model, device, input_shape, num_gpus):
    batch_size = 32 * max(1, num_gpus)  # Scale initial batch size with number of GPUs
    while True:
        try:
            x = torch.randn(batch_size, *input_shape).to(device)
            model(x)
            return batch_size
        except RuntimeError:
            batch_size //= 2
            if batch_size < 1:
                raise ValueError("Cannot find a suitable batch size")

def train_epoch(model, optimizer, pos_weights, dataloader, num_tasks, device, lambda_ortho, num_chunks=4):
    model.train()
    epoch_loss = 0.0
    total_updates = 0
    
    for images, labels, clusters in dataloader:
        images, labels, clusters = images.to(device), labels.to(device), clusters.to(device)
        batch_size = len(images)
        chunk_size = batch_size // num_chunks
        batch_loss = 0.0
        
        # Process the batch in smaller chunks with parameter updates after each chunk
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_chunks - 1 else batch_size
            
            # Extract the current chunk
            chunk_images = images[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]
            chunk_clusters = clusters[start_idx:end_idx]
            
            # Skip processing if chunk is empty (could happen in the last chunk)
            if len(chunk_images) == 0:
                continue
                
            # Zero gradients before processing this chunk
            optimizer.zero_grad()
            
            # Forward pass
            outputs_list, _ = model(chunk_images)
            if isinstance(model, nn.DataParallel):
                outputs = outputs_list
            else:
                outputs = outputs_list
            
            # Calculate task loss for this chunk
            task_loss = 0.0
            unique_clusters = torch.unique(chunk_clusters)
            if len(unique_clusters) > 0 and max(unique_clusters) >= num_tasks:
                raise ValueError(f"Cluster index {max(unique_clusters)} exceeds num_tasks {num_tasks}")
            
            for cluster in unique_clusters:
                mask = chunk_clusters == cluster
                if not torch.any(mask):
                    continue  # Skip if no samples in this cluster
                
                cluster_outputs = outputs[cluster][mask]
                cluster_labels = chunk_labels[mask]
                pos_weight = pos_weights[cluster]
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                task_loss += criterion(cluster_outputs.view(-1), cluster_labels.float())
            
            # Orthogonality loss calculation
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            ortho_loss = 0.0
            for j in range(num_tasks):
                for k in range(j + 1, num_tasks):
                    w_j = actual_model.heads[j][2].weight
                    w_k = actual_model.heads[k][2].weight
                    cos_sim = nn.functional.cosine_similarity(w_j.flatten(), w_k.flatten(), dim=0)
                    ortho_loss += torch.abs(cos_sim)
            
            # Calculate total loss for this chunk
            chunk_loss = task_loss + lambda_ortho * ortho_loss
            
            # Backward pass and optimization for this chunk
            chunk_loss.backward()
            optimizer.step()
            
            # Accumulate loss for logging
            batch_loss += chunk_loss.item()
            total_updates += 1
        
        # Add average batch loss to epoch loss
        epoch_loss += batch_loss / max(1, min(num_chunks, batch_size))
    
    # Return average loss across all batches
    return epoch_loss / len(dataloader), total_updates

def val_epoch(model, dataloader, device):
    model.eval()
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, clusters in dataloader:
            images, labels, clusters = images.to(device), labels.to(device), clusters.to(device)
            
            # Handle output from DataParallel model
            outputs_list, _ = model(images)
            if isinstance(model, nn.DataParallel):
                outputs = outputs_list
            else:
                outputs = outputs_list
                
            selected_outputs = torch.stack([outputs[cluster][i] for i, cluster in enumerate(clusters)])
            loss = nn.BCEWithLogitsLoss()(selected_outputs.squeeze(), labels.float())
            val_loss += loss.item()
            preds = (torch.sigmoid(selected_outputs) > 0.5).int().squeeze()
            acc_metric(preds, labels.int())
    val_acc = acc_metric.compute().item()
    val_loss = val_loss / len(dataloader)
    return val_loss, val_acc

def main():
    args = parse_arguments()
    device = setup_device(args.num_gpus)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Experiment name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"UNI_{args.classification_task}_{args.testset}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Wandb with all configuration
    wandb.init(project="pathology_multitask", name=exp_name, config=vars(args))

    # Log GPU information
    if torch.cuda.is_available():
        gpu_count = min(torch.cuda.device_count(), args.num_gpus)
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        wandb.log({"gpu_count": gpu_count, "gpu_names": gpu_names})

    # Load and merge datasets
    df_dataset = pd.read_csv(args.dataset_path)
    cluster_file = os.path.join(args.cluster_path, f"clustering_result_{args.classification_task}_{args.testset}.csv")
    df_cluster = pd.read_csv(cluster_file)
    
    # Merge on 'dataset' and 'img_path'
    df = df_dataset.merge(df_cluster[['dataset', 'img_path', 'labelCluster']], 
                          on=['dataset', 'img_path'], 
                          how='inner')
    
    # Exclude testset from training data
    train_val_df = df[df.dataset != args.testset].reset_index(drop=True)
    
    # Determine num_tasks based on unique clusters in training/validation data
    unique_clusters = sorted(train_val_df['labelCluster'].unique())
    num_tasks = len(unique_clusters) if args.multitask else 1
    if args.multitask:
        print(f"Multi-task mode enabled. Number of tasks: {num_tasks}, Unique clusters: {unique_clusters}")
    else:
        print("Single-task mode enabled. Number of tasks: 1")

    # Compute pos_weights per cluster
    pos_weights = []
    for cluster in range(num_tasks):
        cluster_df = train_val_df[train_val_df['labelCluster'] == cluster]
        pos_weight = (cluster_df['label'] == 0).sum() / (cluster_df['label'] == 1).sum() if (cluster_df['label'] == 1).sum() > 0 else float('inf')
        pos_weights.append(pos_weight)
    pos_weights = torch.tensor(pos_weights, device=device)

    # Split based on unique img_path
    unique_img_paths = train_val_df['img_path'].unique()
    train_img_paths, val_img_paths = train_test_split(
        unique_img_paths, 
        train_size=args.train_split, 
        random_state=args.seed
    )
    
    # Create train and validation DataFrames based on img_path
    train_df = train_val_df[train_val_df['img_path'].isin(train_img_paths)].reset_index(drop=True)
    val_df = train_val_df[train_val_df['img_path'].isin(val_img_paths)].reset_index(drop=True)
    
    # Verify split
    print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
    assert len(set(train_img_paths) & set(val_img_paths)) == 0, "Overlap detected between train and validation img_paths!"

    # Create datasets
    train_dataset = MultiTaskDataset(train_df, args.classification_task, crop_size=args.crop_size)
    val_dataset = MultiTaskDataset(val_df, args.classification_task, crop_size=args.crop_size)

    # Initialize model
    model = UNIMultitask(num_tasks=num_tasks)
    
    # Wrap model with DataParallel if using multiple GPUs
    if torch.cuda.is_available() and args.num_gpus > 1:
        if torch.cuda.device_count() > 1:
            print(f"Using {min(torch.cuda.device_count(), args.num_gpus)} GPUs with DataParallel")
            model = nn.DataParallel(model, device_ids=list(range(min(torch.cuda.device_count(), args.num_gpus))))
    
    model.to(device)
    
    # Determine batch size
    batch_size = get_max_batch_size(model, device, (3, args.crop_size, args.crop_size), args.num_gpus)
    print(f"Using batch size: {batch_size}")

    # DataLoaders with dynamic num_workers
    num_workers = min(8, os.cpu_count())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    total_train_updates = 0
    
    print(f"Starting training with {args.num_chunks} updates per batch")
    
    for epoch in range(args.epochs):
        train_loss, epoch_updates = train_epoch(
            model, 
            optimizer, 
            pos_weights, 
            train_dataloader, 
            num_tasks, 
            device, 
            args.lambda_ortho,
            num_chunks=args.num_chunks
        )
        
        total_train_updates += epoch_updates
        val_loss, val_acc = val_epoch(model, val_dataloader, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Updates: {epoch_updates} (Total: {total_train_updates}), "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch+1, 
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "val_acc": val_acc,
            "updates_per_epoch": epoch_updates,
            "total_updates": total_train_updates
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_filename = f"UNI_{args.classification_task}_{args.testset}_valacc{best_val_acc:.4f}_{timestamp}.pth"
            # Save the model state dict (access .module if it's a DataParallel model to save the actual model)
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
                       os.path.join(args.output_dir, model_filename))
            print(f"Saved best model: {model_filename}")

    wandb.finish()
    print(f"Training completed with {total_train_updates} total parameter updates.")

if __name__ == "__main__":
    main()
    