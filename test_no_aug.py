import argparse
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
import pandas as pd
import os
import re
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import wandb
from datasets import MultiTaskDataset
from models import UNIMultitask
from torch.cuda.amp import autocast
import warnings
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='MultiTask Learning Model Testing with DataParallel')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--inference_path', type=str, required=True, help='Path to inference results')
    parser.add_argument('--models_dir', type=str, default='./models', help='Directory with trained models')
    parser.add_argument('--plots_dir', type=str, default='./plots', help='Directory to save plots')
    parser.add_argument('--classification_task', type=str, default='tumor', choices=['tumor', 'TIL'], help='Classification task')
    parser.add_argument('--testset', type=str, required=True, help='Test dataset name')
    parser.add_argument('--crop_size', type=int, default=96, help='Crop size for images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--multitask', action='store_true', help='Enable multitask learning')
    parser.add_argument('--inference_methods', type=str, nargs='+', default=['cluster'], 
                        choices=['cluster', 'weighted_voting', 'weighted_sum'],
                        help='List of inference methods: cluster, weighted_voting, weighted_sum')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--max_batch_size', type=int, default=128, help='Maximum batch size to try')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--model_path', type=str, help='Path to a specific model to evaluate (overrides auto-selection)')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to CSV file')
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

def find_best_model(models_dir, pattern):
    """Find the best model based on validation accuracy from filenames."""
    # Create a Path object for safer file operations
    models_path = Path(models_dir)
    
    # Ensure directory exists
    if not models_path.exists() or not models_path.is_dir():
        raise ValueError(f"Models directory {models_dir} does not exist or is not a directory")
    
    pattern_regex = re.compile(pattern + r"_valacc(\d+\.\d+)_\d{8}_\d{6}\.pth")
    best_val_acc = 0.0
    best_model_path = None
    
    # List all files in the directory
    for file_path in models_path.glob("*.pth"):
        filename = file_path.name
        match = pattern_regex.match(filename)
        if match:
            try:
                val_acc = float(match.group(1))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = file_path
            except (ValueError, IndexError) as e:
                warnings.warn(f"Error parsing accuracy from filename {filename}: {e}")
    
    if best_model_path is None:
        raise ValueError(f"No model found matching pattern {pattern_regex.pattern} in {models_dir}")
    
    print(f"Selected best model: {best_model_path} with validation accuracy {best_val_acc}")
    return str(best_model_path)

def test_model(model, dataloader, device, method, num_tasks, use_amp=False):
    """Test model using specified inference method."""
    model.eval()
    all_preds = []
    all_labels = []
    all_img_paths = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, clusters = batch
                img_paths = None
            else:
                images, labels, clusters, img_paths = batch
            
            images, labels, clusters = images.to(device), labels.to(device), clusters.to(device)
            
            # Implementation with or without AMP
            def process_batch():
                outputs_list, cluster_probs = model(images)
                
                if method == 'cluster':
                    unique_clusters = torch.unique(clusters)
                    if max(unique_clusters) >= num_tasks:
                        raise ValueError(f"Cluster index {max(unique_clusters)} exceeds num_tasks {num_tasks}")
                    selected_outputs = torch.stack([outputs_list[cluster][i] for i, cluster in enumerate(clusters)])
                    preds = torch.sigmoid(selected_outputs.view(-1))
                elif method == 'weighted_voting':
                    outputs_tensor = torch.stack(outputs_list, dim=1).squeeze(-1)
                    probs = torch.sigmoid(outputs_tensor)
                    binary_preds = (probs > 0.5).float()
                    # Avoid division by zero
                    sum_probs = probs.sum(dim=1)
                    preds = torch.zeros_like(sum_probs)
                    mask = sum_probs > 0
                    preds[mask] = (binary_preds * probs).sum(dim=1)[mask] / sum_probs[mask]
                elif method == 'weighted_sum':
                    outputs_tensor = torch.stack(outputs_list, dim=1).squeeze(-1)
                    preds = torch.sigmoid((outputs_tensor * cluster_probs).sum(dim=1))
                else:
                    raise ValueError(f"Unsupported inference method: {method}")
                return preds
            
            # Run with or without AMP
            if use_amp:
                with autocast():
                    preds = process_batch()
            else:
                preds = process_batch()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            if img_paths is not None:
                all_img_paths.extend(img_paths)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    return all_preds, all_labels, all_img_paths if img_paths is not None else None

def calculate_metrics(predictions, labels):
    """Calculate and return various metrics."""
    binary_preds = predictions > 0.5
    
    metrics = {
        'accuracy': torchmetrics.functional.accuracy(binary_preds, labels.int(), task='binary'),
        'f1': torchmetrics.functional.f1_score(binary_preds, labels.int(), task='binary'),
        'auc': torchmetrics.functional.auroc(predictions, labels.int(), task='binary'),
        'precision': torchmetrics.functional.precision(binary_preds, labels.int(), task='binary'),
        'recall': torchmetrics.functional.recall(binary_preds, labels.int(), task='binary'),
        'specificity': torchmetrics.functional.specificity(binary_preds, labels.int(), task='binary'),
        'ap': average_precision_score(labels.numpy(), predictions.numpy())
    }
    
    return metrics

def plot_curves(predictions, labels, method, exp_name, plots_dir):
    """Create and save ROC and PR curves."""
    # ROC curve
    fpr, tpr, _ = roc_curve(labels.numpy(), predictions.numpy())
    auc = torchmetrics.functional.auroc(predictions, labels.int(), task='binary').item()
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {method}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(plots_dir, f"{exp_name}_{method}_roc.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels.numpy(), predictions.numpy())
    ap = average_precision_score(labels.numpy(), predictions.numpy())
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {method}')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    pr_path = os.path.join(plots_dir, f"{exp_name}_{method}_pr.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_path, pr_path

def save_predictions_to_csv(predictions, labels, img_paths, method, exp_name, plots_dir):
    """Save predictions to a CSV file for further analysis."""
    if img_paths is None or len(img_paths) == 0:
        warnings.warn("No image paths available to save predictions")
        return None
    
    results_df = pd.DataFrame({
        'img_path': img_paths,
        'true_label': labels.numpy(),
        'prediction': predictions.numpy(),
        'predicted_label': (predictions > 0.5).int().numpy()
    })
    
    # Create a unique filename
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    predictions_path = os.path.join(plots_dir, f"{exp_name}_{method}_predictions_{timestamp}.csv")
    results_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")
    
    return predictions_path

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directories
    os.makedirs(args.plots_dir, exist_ok=True)
    exp_name = f"UNI_{args.classification_task}_{args.testset}"
    
    # Initialize wandb for tracking experiments
    wandb.init(project="pathology_multitask_test", name=f"{exp_name}_test", config=vars(args))
    
    # Load test data
    df = pd.read_csv(args.dataset_path)
    df = df[df.dataset == args.testset].reset_index(drop=True)
    
    # Load inference results
    inference_file = os.path.join(args.inference_path, f"inference_{args.classification_task}_{args.testset}.csv")
    if not os.path.exists(inference_file):
        raise FileNotFoundError(f"Inference file not found: {inference_file}")
    
    df_inference = pd.read_csv(inference_file)
    df_test = df.merge(df_inference[['img_path', 'labelCluster']], on='img_path', how='inner')
    
    if len(df_test) == 0:
        raise ValueError("No matching data found after merging test dataset with inference results")
    
    # Load training data to get number of tasks
    df_dataset = pd.read_csv(args.dataset_path)
    cluster_file_path = args.inference_path.replace('clustering_updated/inference_results', 'clustering/output')
    cluster_file = os.path.join(cluster_file_path, f"clustering_result_{args.classification_task}_{args.testset}.csv")
    
    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Clustering file not found: {cluster_file}")
    
    df_cluster = pd.read_csv(cluster_file)
    df_train = df_dataset.merge(df_cluster[['dataset', 'img_path', 'labelCluster']], on=['dataset', 'img_path'], how='inner')
    train_val_df = df_train[df_train.dataset != args.testset].reset_index(drop=True)
    
    unique_train_clusters = sorted(train_val_df['labelCluster'].unique())
    num_tasks = len(unique_train_clusters) if args.multitask else 1
    print(f"{'Multi-task' if args.multitask else 'Single-task'} mode. Tasks: {num_tasks}")
    
    unique_test_clusters = sorted(df_test['labelCluster'].unique())
    print(f"Test clusters: {unique_test_clusters}")
    if max(unique_test_clusters) >= num_tasks:
        raise ValueError(f"Test clusters {unique_test_clusters} exceed num_tasks {num_tasks}")
    
    # Set up dataset and dataloader
    dataset = MultiTaskDataset(df_test, args.classification_task, crop_size=args.crop_size, is_training=False)
    
    # Set number of workers properly based on available CPU cores
    num_workers = min(args.num_workers, os.cpu_count() or 4)
    print(f"Using {num_workers} workers for data loading")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    # Load model
    model = UNIMultitask(num_tasks=num_tasks).to(device)
    
    if args.model_path:
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        model_path = find_best_model(args.models_dir, exp_name)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    # Use DataParallel for multi-GPU processing if available
    if torch.cuda.is_available() and args.num_gpus > 1:
        gpu_ids = list(range(min(torch.cuda.device_count(), args.num_gpus)))
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    # Optimize batch size if needed
    if args.batch_size <= 0:
        optimal_batch_size = find_optimal_batch_size(
            model, 
            device, 
            (3, args.crop_size, args.crop_size), 
            min_batch=32, 
            max_batch=args.max_batch_size
        )
        print(f"Using optimal batch size: {optimal_batch_size}")
        dataloader = DataLoader(
            dataset, 
            batch_size=optimal_batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True,
            drop_last=False
        )
    
    # Dictionary to store results across methods
    all_metrics = {}
    
    # Test using each inference method
    for method in args.inference_methods:
        print(f"\nTesting with method: {method}")
        all_preds, all_labels, img_paths = test_model(model, dataloader, device, method, num_tasks, use_amp=args.use_amp)
        
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_labels)
        all_metrics[method] = metrics
        
        # Log metrics
        metric_string = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Method: {method}, {metric_string}")
        
        # Log to wandb
        wandb.log({f"test_{k}_{method}": v for k, v in metrics.items()})
        
        # Plot ROC and PR curves
        roc_path, pr_path = plot_curves(all_preds, all_labels, method, exp_name, args.plots_dir)
        wandb.log({
            f"roc_curve_{method}": wandb.Image(roc_path),
            f"pr_curve_{method}": wandb.Image(pr_path)
        })
        
        # Save predictions if requested
        if img_paths: # and args.save_predictions
            predictions_path = save_predictions_to_csv(all_preds, all_labels, img_paths, method, exp_name, args.plots_dir)
            if predictions_path:
                wandb.save(predictions_path)
    
    # Compare methods if multiple methods were used
    if len(args.inference_methods) > 1:
        comparison_data = {}
        for metric in ['accuracy', 'f1', 'auc', 'precision', 'recall', 'specificity', 'ap']:
            comparison_data[metric] = {method: all_metrics[method][metric].item() if torch.is_tensor(all_metrics[method][metric]) else all_metrics[method][metric] for method in args.inference_methods}
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\nMethod Comparison:")
        print(comparison_df)
        
        # Save comparison to CSV
        comparison_path = os.path.join(args.plots_dir, f"{exp_name}_method_comparison.csv")
        comparison_df.to_csv(comparison_path)
        wandb.save(comparison_path)
        
        # Create comparison bar chart
        plt.figure(figsize=(12, 8))
        comparison_df.T.plot(kind='bar')  # Transpose for better visualization
        plt.title('Performance Metrics Across Inference Methods')
        plt.ylabel('Score')
        plt.xlabel('Metric')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        comparison_chart_path = os.path.join(args.plots_dir, f"{exp_name}_method_comparison.png")
        plt.savefig(comparison_chart_path, dpi=300)
        plt.close()
        wandb.log({"method_comparison": wandb.Image(comparison_chart_path)})
    
    wandb.finish()
    torch.cuda.empty_cache()
    print("Testing completed successfully.")

if __name__ == "__main__":
    main()
    