import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader
import pandas as pd
import os
import shutil
from datetime import datetime
import wandb
from datasets import MultiTaskDataset
from models import UNIMultitask
from sklearn.model_selection import train_test_split
# Updated import path - now from torch.amp instead of torch.cuda.amp
from torch.amp import autocast, GradScaler

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
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--t_max', type=int, default=10, help='T_max for CosineAnnealingLR')
    parser.add_argument('--keep_n_checkpoints', type=int, default=3, help='Number of best checkpoints to keep')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
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
    return min_batch

def train_epoch(model, optimizer, pos_weights, dataloader, num_tasks, device, lambda_ortho, 
                label_smoothing=0.1, scaler=None, use_amp=False, accumulation_steps=1):
    model.train()
    epoch_loss = 0.0
    total_updates = 0
    
    # Determine device type for autocast
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i, (images, labels, clusters) in enumerate(dataloader):
        images, labels, clusters = images.to(device), labels.to(device), clusters.to(device)
        
        if i % accumulation_steps == 0:
            optimizer.zero_grad()
        
        if use_amp:
            # Updated autocast with device type parameter
            with autocast(device_type=device_type):
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
                    
                    # Apply label smoothing
                    smooth_targets = cluster_labels.float() * (1 - label_smoothing) + 0.5 * label_smoothing
                    
                    # BCEWithLogitsLoss with pos_weight and manually applied label smoothing
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                    cluster_loss = criterion(cluster_outputs.view(-1), smooth_targets)
                    task_loss += cluster_loss.mean()
                
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
                
                # Apply label smoothing
                smooth_targets = cluster_labels.float() * (1 - label_smoothing) + 0.5 * label_smoothing
                
                # BCEWithLogitsLoss with pos_weight and manually applied label smoothing
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                cluster_loss = criterion(cluster_outputs.view(-1), smooth_targets)
                task_loss += cluster_loss.mean()
            
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

def val_epoch(model, dataloader, device, num_tasks, use_amp=False):
    model.eval()
    task_aucs = torch.zeros(num_tasks, device=device)
    task_confidence = torch.zeros(num_tasks, device=device)
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_clusters = []
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        for images, labels, clusters in dataloader:
            images, labels, clusters = images.to(device), labels.to(device), clusters.to(device)
            
            if use_amp:
                with autocast(device_type=device_type):
                    outputs_list, cluster_probs = model(images)
                    selected_outputs_list = []
                    for i, cluster in enumerate(clusters):
                        if cluster < len(outputs_list) and outputs_list[cluster].numel() > 0:
                            selected_outputs_list.append(outputs_list[cluster][i])
                    if not selected_outputs_list:
                        print("Warning: No valid outputs for batch, skipping")
                        continue
                    selected_outputs = torch.stack(selected_outputs_list)
                    loss = nn.BCEWithLogitsLoss()(selected_outputs.squeeze(), labels.float())
            else:
                outputs_list, cluster_probs = model(images)
                selected_outputs_list = []
                for i, cluster in enumerate(clusters):
                    if cluster < len(outputs_list) and outputs_list[cluster].numel() > 0:
                        selected_outputs_list.append(outputs_list[cluster][i])
                if not selected_outputs_list:
                    print("Warning: No valid outputs for batch, skipping")
                    continue
                selected_outputs = torch.stack(selected_outputs_list)
                loss = nn.BCEWithLogitsLoss()(selected_outputs.squeeze(), labels.float())
            
            val_loss += loss.item()
            
            preds = torch.sigmoid(selected_outputs.squeeze())
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_clusters.append(clusters.cpu())
    
    if not all_preds:
        raise RuntimeError("No valid predictions collected during validation")
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_clusters = torch.cat(all_clusters)
    
    # Calculate overall validation accuracy
    acc_metric = torchmetrics.classification.BinaryAccuracy()
    val_acc = acc_metric(all_preds > 0.5, all_labels.int()).item()
    
    # Calculate per-task AUC for task head relevance weighting
    for task_id in range(num_tasks):
        task_mask = all_clusters == task_id
        if torch.sum(task_mask) > 0:
            task_labels = all_labels[task_mask]
            task_pred_list = []
            for images, labels, clusters in dataloader:
                images = images.to(device)
                task_mask = clusters == task_id
                if torch.sum(task_mask) > 0:
                    with torch.no_grad():
                        if use_amp:
                            with autocast(device_type=device_type):
                                outputs_list, _ = model(images[task_mask])
                                preds = torch.sigmoid(outputs_list[task_id])
                        else:
                            outputs_list, _ = model(images[task_mask])
                            preds = torch.sigmoid(outputs_list[task_id])
                        
                        if preds.numel() > 0:
                            task_pred_list.append(preds.cpu().squeeze())
            
            if task_pred_list and all(p.numel() > 0 for p in task_pred_list):
                task_preds = torch.cat(task_pred_list)
                if len(torch.unique(task_labels)) > 1:
                    auroc_metric = torchmetrics.classification.BinaryAUROC()
                    task_auc = auroc_metric(task_preds, task_labels.int()).item()
                    task_aucs[task_id] = task_auc
                    
                    confidence = (torch.abs(task_preds - 0.5) * 2).mean().item()
                    task_confidence[task_id] = confidence
    
    temperature = 1.0 / (task_confidence + 1e-6)
    temperature = temperature / temperature.mean()
    
    val_loss = val_loss / len(dataloader)
    return val_loss, val_acc, task_aucs, temperature

def clean_old_checkpoints(output_dir, model_prefix, keep_n):
    """Remove old checkpoints keeping only the best keep_n models."""
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith(model_prefix)]
    if len(checkpoints) <= keep_n:
        return
    
    # Sort by validation accuracy (descending)
    pattern = re.compile(r'.*valacc(\d+\.\d+)_.*\.pth')
    checkpoint_scores = []
    for ckpt in checkpoints:
        match = pattern.match(ckpt)
        if match:
            score = float(match.group(1))
            checkpoint_scores.append((ckpt, score))
    
    # Sort by score (descending)
    checkpoint_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Remove oldest checkpoints
    for ckpt, _ in checkpoint_scores[keep_n:]:
        os.remove(os.path.join(output_dir, ckpt))
        print(f"Removed old checkpoint: {ckpt}")

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"UNI_{args.classification_task}_{args.testset}_{timestamp}"
    model_prefix = f"UNI_{args.classification_task}_{args.testset}"
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
    
    # Initialize model
    model = UNIMultitask(num_tasks=num_tasks).to(device)
    
    # Setup DataParallel if multiple GPUs are available
    if torch.cuda.is_available() and args.num_gpus > 1:
        gpu_ids = list(range(min(torch.cuda.device_count(), args.num_gpus)))
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    # Find optimal batch size
    optimal_batch_size = find_optimal_batch_size(model, device, (3, args.crop_size, args.crop_size), 
                                               min_batch=args.batch_size, max_batch=args.max_batch_size)
    print(f"Optimal batch size: {optimal_batch_size}")
    
    # Create dataloaders with appropriate batch size and num_workers
    train_dataloader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True, 
                                 num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=optimal_batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
    
    # Updated for modern PyTorch (2.0+): specify device type for GradScaler
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = GradScaler() if args.use_amp else None
    
    best_val_acc = 0.0
    total_updates = 0
    best_task_aucs = torch.zeros(num_tasks)
    best_temperature = torch.ones(num_tasks)
    checkpoints_info = []
    
    print(f"Training started. AMP: {'Enabled' if args.use_amp else 'Disabled'}, "
          f"Accumulation steps: {args.gradient_accumulation_steps}, "
          f"Label smoothing: {args.label_smoothing}")
    
    for epoch in range(args.epochs):
        train_loss, epoch_updates = train_epoch(model, optimizer, pos_weights, train_dataloader, num_tasks, device, 
                                               args.lambda_ortho, args.label_smoothing, scaler, args.use_amp, 
                                               args.gradient_accumulation_steps)
        total_updates += epoch_updates
        val_loss, val_acc, task_aucs, temperature = val_epoch(model, val_dataloader, device, num_tasks, 
                                                            use_amp=args.use_amp)
        
        # Update learning rate using cosine annealing
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Updates: {epoch_updates} (Total: {total_updates}), "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Task AUCs: {task_aucs.cpu().numpy()}")
        print(f"Task Temperatures: {temperature.cpu().numpy()}")
        
        wandb.log({
            "epoch": epoch+1, 
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "val_acc": val_acc,
            "updates_per_epoch": epoch_updates, 
            "total_updates": total_updates,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Log per-task metrics
        for task_id in range(num_tasks):
            wandb.log({
                f"task_{task_id}_auc": task_aucs[task_id].item(),
                f"task_{task_id}_temperature": temperature[task_id].item(),
            })
        
        # Save model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_task_aucs = task_aucs.clone()
            best_temperature = temperature.clone()
            
            # Save checkpoint
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{model_prefix}_valacc{best_val_acc:.4f}_{timestamp}.pth"
            model_path = os.path.join(args.output_dir, model_filename)
            
            # Save model state dict
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            # Create metadata for validation-based inference parameters
            metadata = {
                'val_acc': best_val_acc,
                'task_aucs': best_task_aucs.cpu().tolist(),
                'temperature': best_temperature.cpu().tolist(),
                'num_tasks': num_tasks,
                'val_paths': val_paths.tolist(),  # Save validation image paths for potential thresholding
                'testset': args.testset,
                'classification_task': args.classification_task,
            }
            
            # Save model with metadata
            torch.save({
                'model_state_dict': model_state,
                'metadata': metadata
            }, model_path)
            
            print(f"Saved best model: {model_filename}")
            checkpoints_info.append((model_path, best_val_acc))
            
            # Clean up old checkpoints
            clean_old_checkpoints(args.output_dir, model_prefix, args.keep_n_checkpoints)
    
    # Save optimal threshold values for each task based on validation data
    if len(checkpoints_info) > 0:
        best_checkpoint_path, _ = max(checkpoints_info, key=lambda x: x[1])
        checkpoint = torch.load(best_checkpoint_path)
        
        # Compute optimal thresholds for each task using validation data
        optimal_thresholds = []
        for task_id in range(num_tasks):
            # Get validation samples for this task
            task_val_df = val_df[val_df['labelCluster'] == task_id]
            if len(task_val_df) > 0:
                task_val_dataset = MultiTaskDataset(task_val_df, args.classification_task, crop_size=args.crop_size)
                task_val_loader = DataLoader(task_val_dataset, batch_size=optimal_batch_size, shuffle=False, 
                                            num_workers=args.num_workers, pin_memory=True)
                
                # Collect predictions and labels
                task_preds = []
                task_labels = []
                model.eval()
                with torch.no_grad():
                    for images, labels, _ in task_val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs_list, _ = model(images)
                        preds = torch.sigmoid(outputs_list[task_id])
                        # Add checks to ensure preds is not empty
                        if preds.numel() > 0:
                            task_preds.append(preds.cpu())
                            task_labels.append(labels.cpu())
                
                # Check if we have predictions before concatenating
                if task_preds and all(p.numel() > 0 for p in task_preds):
                    task_preds = torch.cat(task_preds)
                    task_labels = torch.cat(task_labels)
                    
                    # Find optimal threshold using F1 score
                    best_f1 = 0
                    best_threshold = 0.5  # Default
                    for threshold in torch.linspace(0.3, 0.7, 40):  # Search in reasonable range
                        f1 = torchmetrics.functional.f1_score(task_preds > threshold, task_labels.int(), task='binary')
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold.item()
                    
                    optimal_thresholds.append(best_threshold)
                else:
                    optimal_thresholds.append(0.5)  # Default threshold if no predictions
            else:
                optimal_thresholds.append(0.5)  # Default threshold if no validation data
        
        # Update checkpoint with optimal thresholds
        checkpoint['metadata']['optimal_thresholds'] = optimal_thresholds
        torch.save(checkpoint, best_checkpoint_path)
        print(f"Updated best model with optimal thresholds: {optimal_thresholds}")
    
    wandb.finish()
    print(f"Training completed with {total_updates} updates.")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import re  # Added for regex pattern matching in clean_old_checkpoints
    main()
    