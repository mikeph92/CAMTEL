import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader
import pandas as pd
import os
from datetime import datetime
import wandb
from datasets import MultiTaskDataset
from models import UNIMultitask
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description='MultiTask Learning for Pathology Image Classification with DataParallel')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to tumor_dataset.csv')
    parser.add_argument('--cluster_path', type=str, required=True, help='Path to clustering result directory')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--classification_task', type=str, default='tumor', choices=['tumor', 'TIL'], help='Classification task')
    parser.add_argument('--testset', type=str, required=True, help='Test dataset name')
    parser.add_argument('--crop_size', type=int, default=96, help='Crop size for images')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--multitask', action='store_true', help='Enable multitask learning')
    parser.add_argument('--lambda_ortho', type=float, default=0.01, help='Weight for orthogonality loss')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_batch_size', type=int, default=128, help='Maximum batch size to try')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--max_checkpoints', type=int, default=3, help='Maximum number of checkpoints to keep')
    parser.add_argument('--t_max', type=int, default=10, help='T_max parameter for CosineAnnealingLR')
    return parser.parse_args()

def find_optimal_batch_size(model, device, input_shape, min_batch=32, max_batch=512):
    batch_size = max_batch
    while batch_size >= min_batch:
        try:
            torch.cuda.empty_cache()
            x = torch.randn(batch_size, *input_shape).to(device)
            model(x)
            del x
            torch.cuda.empty_cache()
            return batch_size
        except RuntimeError:
            batch_size //= 2
    return min_batch  # Return min_batch instead of hardcoded value

# Custom Binary Cross Entropy with Label Smoothing
class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.1):
        super(BCEWithLogitsLossWithSmoothing, self).__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing
        
    def forward(self, logits, target):
        # Apply label smoothing: move labels away from 0 and 1
        smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        # Use standard BCE with the smoothed targets
        if self.pos_weight is not None:
            return nn.functional.binary_cross_entropy_with_logits(
                logits, smooth_target, pos_weight=self.pos_weight
            )
        else:
            return nn.functional.binary_cross_entropy_with_logits(
                logits, smooth_target
            )

def train_epoch(model, optimizer, pos_weights, dataloader, num_tasks, device, lambda_ortho, 
                label_smoothing=0.1, scaler=None, use_amp=False, accumulation_steps=1):
    model.train()
    epoch_loss = 0.0
    total_updates = 0
    
    for i, (images, labels, clusters) in enumerate(dataloader):
        images, labels, clusters = images.to(device), labels.to(device), clusters.to(device)
        
        if i % accumulation_steps == 0:
            optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs_list, _ = model(images)
                task_loss = 0.0
                unique_clusters = torch.unique(clusters)
                for cluster in unique_clusters:
                    mask = clusters == cluster
                    if not torch.any(mask):
                        continue
                    cluster_outputs = outputs_list[cluster][mask]
                    cluster_labels = labels[mask]
                    pos_weight = pos_weights[cluster]
                    criterion = BCEWithLogitsLossWithSmoothing(pos_weight=pos_weight, smoothing=label_smoothing)
                    task_loss += criterion(cluster_outputs.view(-1), cluster_labels.float())
                
                actual_model = model.module if isinstance(model, nn.DataParallel) else model
                ortho_loss = 0.0
                for j in range(num_tasks):
                    for k in range(j + 1, num_tasks):
                        w_j = actual_model.heads[j][2].weight
                        w_k = actual_model.heads[k][2].weight
                        cos_sim = nn.functional.cosine_similarity(w_j.flatten(), w_k.flatten(), dim=0)
                        ortho_loss += torch.abs(cos_sim)
                
                loss = task_loss + lambda_ortho * ortho_loss
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                total_updates += 1
        else:
            outputs_list, _ = model(images)
            task_loss = 0.0
            unique_clusters = torch.unique(clusters)
            for cluster in unique_clusters:
                mask = clusters == cluster
                if not torch.any(mask):
                    continue
                cluster_outputs = outputs_list[cluster][mask]
                cluster_labels = labels[mask]
                pos_weight = pos_weights[cluster]
                criterion = BCEWithLogitsLossWithSmoothing(pos_weight=pos_weight, smoothing=label_smoothing)
                task_loss += criterion(cluster_outputs.view(-1), cluster_labels.float())
            
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            ortho_loss = 0.0
            for j in range(num_tasks):
                for k in range(j + 1, num_tasks):
                    w_j = actual_model.heads[j][2].weight
                    w_k = actual_model.heads[k][2].weight
                    cos_sim = nn.functional.cosine_similarity(w_j.flatten(), w_k.flatten(), dim=0)
                    ortho_loss += torch.abs(cos_sim)
            
            loss = task_loss + lambda_ortho * ortho_loss
            loss = loss / accumulation_steps
            
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                total_updates += 1
        
        epoch_loss += loss.item() * accumulation_steps
    
    return epoch_loss / len(dataloader), total_updates

def val_epoch(model, dataloader, device, label_smoothing=0.1, use_amp=False):
    model.eval()
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, clusters in dataloader:
            images, labels, clusters = images.to(device), labels.to(device), clusters.to(device)
            if use_amp:
                with autocast():
                    outputs_list, _ = model(images)
                    selected_outputs = torch.stack([outputs_list[cluster][i] for i, cluster in enumerate(clusters)])
                    # Note: For evaluation, we use standard BCE without smoothing for accurate loss measurement
                    loss = nn.BCEWithLogitsLoss()(selected_outputs.squeeze(), labels.float())
            else:
                outputs_list, _ = model(images)
                selected_outputs = torch.stack([outputs_list[cluster][i] for i, cluster in enumerate(clusters)])
                loss = nn.BCEWithLogitsLoss()(selected_outputs.squeeze(), labels.float())
            
            val_loss += loss.item()
            preds = (torch.sigmoid(selected_outputs) > 0.5).int().squeeze()
            acc_metric(preds, labels.int())
    
    val_acc = acc_metric.compute().item()
    val_loss = val_loss / len(dataloader)
    return val_loss, val_acc

def cleanup_checkpoints(output_dir, model_prefix, max_to_keep=3):
    """Delete older checkpoints, keeping only the most recent ones."""
    pattern = os.path.join(output_dir, f"{model_prefix}_*_valacc*.pth")
    checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    # Keep only the most recent checkpoints
    for checkpoint in checkpoints[max_to_keep:]:
        try:
            os.remove(checkpoint)
            print(f"Removed old checkpoint: {os.path.basename(checkpoint)}")
        except Exception as e:
            print(f"Error removing checkpoint {checkpoint}: {e}")

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_prefix = f"UNI_{args.classification_task}_{args.testset}"
    exp_name = f"{model_prefix}_{timestamp}"
    wandb.init(project="pathology_multitask", name=exp_name, config=vars(args))
    
    df_dataset = pd.read_csv(args.dataset_path)
    cluster_file = os.path.join(args.cluster_path, f"clustering_result_{args.classification_task}_{args.testset}.csv")
    df_cluster = pd.read_csv(cluster_file)
    df = df_dataset.merge(df_cluster[['dataset', 'img_path', 'labelCluster']], on=['dataset', 'img_path'], how='inner')
    train_val_df = df[df.dataset != args.testset].reset_index(drop=True)
    
    unique_clusters = sorted(train_val_df['labelCluster'].unique())
    num_tasks = len(unique_clusters) if args.multitask else 1
    print(f"{'Multi-task' if args.multitask else 'Single-task'} mode. Tasks: {num_tasks}, Clusters: {unique_clusters}")
    
    pos_weights = [torch.tensor((train_val_df[train_val_df['labelCluster'] == cluster]['label'] == 0).sum() / 
                               max(1, (train_val_df[train_val_df['labelCluster'] == cluster]['label'] == 1).sum()), 
                               device=device) for cluster in range(num_tasks)]
    
    unique_img_paths = train_val_df['img_path'].unique()
    train_paths, val_paths = train_test_split(unique_img_paths, train_size=args.train_split, random_state=args.seed)
    train_df = train_val_df[train_val_df['img_path'].isin(train_paths)].reset_index(drop=True)
    val_df = train_val_df[train_val_df['img_path'].isin(val_paths)].reset_index(drop=True)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    train_dataset = MultiTaskDataset(train_df, args.classification_task, crop_size=args.crop_size)
    val_dataset = MultiTaskDataset(val_df, args.classification_task, crop_size=args.crop_size)
    
    model = UNIMultitask(num_tasks=num_tasks).to(device)
    if torch.cuda.is_available() and args.num_gpus > 1:
        gpu_ids = list(range(min(torch.cuda.device_count(), args.num_gpus)))
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    optimal_batch_size = find_optimal_batch_size(model, device, (3, args.crop_size, args.crop_size), 
                                                min_batch=args.batch_size, max_batch=args.max_batch_size)
    print(f"Optimal batch size: {optimal_batch_size}")
    
    # Set number of workers properly based on available CPU cores
    num_workers = min(args.num_workers, os.cpu_count() or 4)
    print(f"Using {num_workers} workers for data loading")
    
    train_dataloader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=optimal_batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Replace ReduceLROnPlateau with CosineAnnealingLR
    t_max = args.t_max if args.t_max > 0 else args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.lr * 0.01)
    
    scaler = GradScaler() if args.use_amp else None
    
    best_val_acc = 0.0
    total_updates = 0
    
    print(f"Training started. AMP: {'Enabled' if args.use_amp else 'Disabled'}, "
          f"Accumulation steps: {args.gradient_accumulation_steps}, "
          f"Label smoothing: {args.label_smoothing}")
    
    for epoch in range(args.epochs):
        train_loss, epoch_updates = train_epoch(
            model, optimizer, pos_weights, train_dataloader, num_tasks, device, 
            args.lambda_ortho, args.label_smoothing, scaler, args.use_amp, args.gradient_accumulation_steps
        )
        total_updates += epoch_updates
        
        val_loss, val_acc = val_epoch(model, val_dataloader, device, 
                                     label_smoothing=args.label_smoothing, use_amp=args.use_amp)
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs}, Updates: {epoch_updates} (Total: {total_updates}), "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"LR: {current_lr:.6f}")
        
        wandb.log({
            "epoch": epoch+1, 
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "val_acc": val_acc,
            "updates_per_epoch": epoch_updates, 
            "total_updates": total_updates,
            "learning_rate": current_lr
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{model_prefix}_valacc{best_val_acc:.4f}_{timestamp}.pth"
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
                      os.path.join(args.output_dir, model_filename))
            print(f"Saved best model: {model_filename}")
            
            # Clean up old checkpoints
            cleanup_checkpoints(args.output_dir, model_prefix, args.max_checkpoints)
    
    # Save final model regardless of performance
    final_model_filename = f"{model_prefix}_final_epoch{args.epochs}_{timestamp}.pth"
    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
              os.path.join(args.output_dir, final_model_filename))
    print(f"Saved final model: {final_model_filename}")
    
    wandb.finish()
    print(f"Training completed with {total_updates} updates.")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()