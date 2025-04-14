import argparse
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
import pandas as pd
import os
import re
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import wandb
from datasets import MultiTaskDataset
from models import UNIMultitask
from torch.cuda.amp import autocast

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
                        help='List of inference methods: cluster, weighted_voting, weighted_sum')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--max_batch_size', type=int, default=128, help='Maximum batch size to try')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                        help='Confidence threshold for dropping low-confidence task heads')
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

def find_best_model(models_dir, pattern):
    pattern = re.compile(pattern + r"_valacc(\d+\.\d+)_\d{8}_\d{6}\.pth")
    best_val_acc = 0.0
    best_model_path = None
    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            val_acc = float(match.group(1))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(models_dir, filename)
    if best_model_path is None:
        raise ValueError(f"No model found matching pattern {pattern.pattern}")
    return best_model_path

def test_model(model, dataloader, device, method, num_tasks, task_weights=None, optimal_thresholds=None, 
               temperature=None, confidence_threshold=0.5, use_amp=False):
    model.eval()
    all_preds = []
    all_labels = []
    
    # Default values if not provided
    if task_weights is None:
        task_weights = torch.ones(num_tasks, device=device)
    if optimal_thresholds is None:
        optimal_thresholds = torch.ones(num_tasks, device=device) * 0.5
    if temperature is None:
        temperature = torch.ones(num_tasks, device=device)
    
    with torch.no_grad():
        for images, labels, clusters in dataloader:
            images, labels, clusters = images.to(device), labels.to(device), clusters.to(device)
            
            if use_amp:
                with autocast():
                    outputs_list, cluster_probs = model(images)
                    if method == 'cluster':
                        unique_clusters = torch.unique(clusters)
                        if max(unique_clusters) >= num_tasks:
                            raise ValueError(f"Cluster index {max(unique_clusters)} exceeds num_tasks {num_tasks}")
                        
                        selected_outputs = torch.stack([outputs_list[cluster][i] for i, cluster in enumerate(clusters)])
                        preds = torch.sigmoid(selected_outputs.view(-1))
                        
                    elif method == 'weighted_voting':
                        outputs_tensor = torch.stack(outputs_list, dim=1).squeeze(-1)  # [batch_size, num_tasks]
                        probs = torch.sigmoid(outputs_tensor)
                        
                        # Apply adaptive thresholding for each task
                        binary_preds = torch.zeros_like(probs)
                        for task_id in range(num_tasks):
                            binary_preds[:, task_id] = (probs[:, task_id] > optimal_thresholds[task_id]).float()
                        
                        # Apply task head relevance weighting based on validation AUC
                        weighted_binary_preds = binary_preds * task_weights.unsqueeze(0)
                        
                        # Compute final prediction
                        weighted_sum = (weighted_binary_preds * probs).sum(dim=1)
                        weights_sum = (weighted_binary_preds * task_weights.unsqueeze(0)).sum(dim=1)
                        weights_sum = torch.clamp(weights_sum, min=1e-6)  # Avoid division by zero
                        preds = weighted_sum / weights_sum
                        
                    elif method == 'weighted_sum':
                        outputs_tensor = torch.stack(outputs_list, dim=1).squeeze(-1)  # [batch_size, num_tasks]
                        
                        # Apply temperature scaling to cluster probabilities
                        scaled_cluster_probs = cluster_probs / temperature.unsqueeze(0)
                        
                        # Drop low confidence task heads
                        mask = cluster_probs > confidence_threshold
                        # Ensure at least one task head is used
                        if not torch.any(mask, dim=1).all():
                            # If no head passes threshold for some samples, use the most confident head
                            max_conf_indices = torch.argmax(cluster_probs, dim=1)
                            for i in range(len(mask)):
                                mask[i, max_conf_indices[i]] = True
                        
                        # Scale probabilities and zero out low-confidence predictions
                        effective_probs = scaled_cluster_probs * mask.float()
                        # Normalize probabilities to sum to 1
                        sum_probs = effective_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
                        normalized_probs = effective_probs / sum_probs
                        
                        # Compute weighted sum of logits with the normalized, temperature-scaled probabilities
                        preds = torch.sigmoid((outputs_tensor * normalized_probs).sum(dim=1))
                    else:
                        raise ValueError(f"Unsupported inference method: {method}")
            else:
                outputs_list, cluster_probs = model(images)
                if method == 'cluster':
                    unique_clusters = torch.unique(clusters)
                    if max(unique_clusters) >= num_tasks:
                        raise ValueError(f"Cluster index {max(unique_clusters)} exceeds num_tasks {num_tasks}")
                    
                    selected_outputs = torch.stack([outputs_list[cluster][i] for i, cluster in enumerate(clusters)])
                    preds = torch.sigmoid(selected_outputs.view(-1))
                    
                elif method == 'weighted_voting':
                    outputs_tensor = torch.stack(outputs_list, dim=1).squeeze(-1)  # [batch_size, num_tasks]
                    probs = torch.sigmoid(outputs_tensor)
                    
                    # Apply adaptive thresholding for each task
                    binary_preds = torch.zeros_like(probs)
                    for task_id in range(num_tasks):
                        binary_preds[:, task_id] = (probs[:, task_id] > optimal_thresholds[task_id]).float()
                    
                    # Apply task head relevance weighting based on validation AUC
                    weighted_binary_preds = binary_preds * task_weights.unsqueeze(0)
                    
                    # Compute final prediction
                    weighted_sum = (weighted_binary_preds * probs).sum(dim=1)
                    weights_sum = (weighted_binary_preds * task_weights.unsqueeze(0)).sum(dim=1)
                    weights_sum = torch.clamp(weights_sum, min=1e-6)  # Avoid division by zero
                    preds = weighted_sum / weights_sum
                    
                elif method == 'weighted_sum':
                    outputs_tensor = torch.stack(outputs_list, dim=1).squeeze(-1)  # [batch_size, num_tasks]
                    
                    # Apply temperature scaling to cluster probabilities
                    scaled_cluster_probs = cluster_probs / temperature.unsqueeze(0)
                    
                    # Drop low confidence task heads
                    mask = cluster_probs > confidence_threshold
                    # Ensure at least one task head is used
                    if not torch.any(mask, dim=1).all():
                        # If no head passes threshold for some samples, use the most confident head
                        max_conf_indices = torch.argmax(cluster_probs, dim=1)
                        for i in range(len(mask)):
                            mask[i, max_conf_indices[i]] = True
                    
                    # Scale probabilities and zero out low-confidence predictions
                    effective_probs = scaled_cluster_probs * mask.float()
                    # Normalize probabilities to sum to 1
                    sum_probs = effective_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    normalized_probs = effective_probs / sum_probs
                    
                    # Compute weighted sum of logits with the normalized, temperature-scaled probabilities
                    preds = torch.sigmoid((outputs_tensor * normalized_probs).sum(dim=1))
                else:
                    raise ValueError(f"Unsupported inference method: {method}")
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds, all_labels

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.plots_dir, exist_ok=True)
    exp_name = f"UNI_{args.classification_task}_{args.testset}"
    wandb.init(project="pathology_multitask_test", name=f"{exp_name}_test")
    
    ALLOWED_METHODS = ['cluster', 'weighted_voting', 'weighted_sum']
    for method in args.inference_methods:
        if method not in ALLOWED_METHODS:
            raise ValueError(f"Invalid inference method: {method}. Allowed methods are {ALLOWED_METHODS}")
    
    # Load test data
    df = pd.read_csv(args.dataset_path)
    df = df[df.dataset == args.testset].reset_index(drop=True)
    inference_file = f"{args.inference_path}/inference_{args.classification_task}_{args.testset}.csv"
    df_inference = pd.read_csv(inference_file)
    df_test = df.merge(df_inference[['img_path', 'labelCluster']], on='img_path', how='inner')
    
    # Load training data for cluster information
    df_dataset = pd.read_csv(args.dataset_path)
    cluster_file = os.path.join(args.inference_path.replace('clustering_updated/inference_results', 'clustering/output'), 
                               f"clustering_result_{args.classification_task}_{args.testset}.csv")
    df_cluster = pd.read_csv(cluster_file)
    df_train = df_dataset.merge(df_cluster[['dataset', 'img_path', 'labelCluster']], on=['dataset', 'img_path'], how='inner')
    train_val_df = df_train[df_train.dataset != args.testset].reset_index(drop=True)
    
    # Determine number of tasks
    unique_train_clusters = sorted(train_val_df['labelCluster'].unique())
    num_tasks = len(unique_train_clusters) if args.multitask else 1
    print(f"{'Multi-task' if args.multitask else 'Single-task'} mode. Tasks: {num_tasks}")
    
    # Check test clusters
    unique_test_clusters = sorted(df_test['labelCluster'].unique())
    print(f"Test clusters: {unique_test_clusters}")
    if max(unique_test_clusters) >= num_tasks:
        raise ValueError(f"Test clusters {unique_test_clusters} exceed num_tasks {num_tasks}")
    
    # Create test dataset and dataloader
    test_dataset = MultiTaskDataset(df_test, args.classification_task, crop_size=args.crop_size, is_training=False)
    
    # Find best model
    best_model_path = find_best_model(args.models_dir, exp_name)
    print(f"Loading best model from: {best_model_path}")
    
    # Load model with metadata
    checkpoint = torch.load(best_model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        metadata = checkpoint.get('metadata', {})
        task_aucs = torch.tensor(metadata.get('task_aucs', [1.0] * num_tasks), device=device)
        temperature = torch.tensor(metadata.get('temperature', [1.0] * num_tasks), device=device)
        optimal_thresholds = torch.tensor(metadata.get('optimal_thresholds', [0.5] * num_tasks), device=device)
        
        print(f"Loaded model metadata:")
        print(f"- Task AUCs: {task_aucs.cpu().numpy()}")
        print(f"- Temperature values: {temperature.cpu().numpy()}")
        print(f"- Optimal thresholds: {optimal_thresholds.cpu().numpy()}")
    else:
        model_state = checkpoint
        task_aucs = torch.ones(num_tasks, device=device)
        temperature = torch.ones(num_tasks, device=device)
        optimal_thresholds = torch.ones(num_tasks, device=device) * 0.5
        print("No metadata found, using default values for task weights and temperature")
    
    # Initialize model
    model = UNIMultitask(num_tasks=num_tasks).to(device)
    model.load_state_dict(model_state)
    
    # Setup DataParallel if multiple GPUs are available
    if torch.cuda.is_available() and args.num_gpus > 1:
        gpu_ids = list(range(min(torch.cuda.device_count(), args.num_gpus)))
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    # Find optimal batch size
    optimal_batch_size = args.batch_size
    if args.batch_size <= 0:
        optimal_batch_size = find_optimal_batch_size(model, device, (3, args.crop_size, args.crop_size), 
                                                   min_batch=32, max_batch=args.max_batch_size)
        print(f"Using optimal batch size: {optimal_batch_size}")
    
    # Create dataloader with optimal batch size
    test_dataloader = DataLoader(test_dataset, batch_size=optimal_batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=True)
    
    # Create task_weights from task_aucs
    task_weights = task_aucs.clone()
    task_weights = task_weights / task_weights.sum() * num_tasks  # Normalize but keep scale
    
    # Test with different inference methods
    results = {}
    for method in args.inference_methods:
        print(f"Testing with method: {method}")
        
        all_preds, all_labels = test_model(
            model, test_dataloader, device, method, num_tasks,
            task_weights=task_weights,
            optimal_thresholds=optimal_thresholds, 
            temperature=temperature,
            confidence_threshold=args.confidence_threshold,
            use_amp=args.use_amp
        )
        
        # Calculate metrics
        acc = torchmetrics.functional.accuracy(all_preds > 0.5, all_labels.int(), task='binary')
        f1 = torchmetrics.functional.f1_score(all_preds > 0.5, all_labels.int(), task='binary')
        precision = torchmetrics.functional.precision(all_preds > 0.5, all_labels.int(), task='binary')
        recall = torchmetrics.functional.recall(all_preds > 0.5, all_labels.int(), task='binary')
        auc_score = torchmetrics.functional.auroc(all_preds, all_labels.int(), task='binary')
        
        print(f"Method: {method}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC: {auc_score:.4f}")
        
        # Log metrics
        results[method] = {
            'accuracy': acc.item(),
            'f1': f1.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'auc': auc_score.item(),
            'predictions': all_preds.numpy(),
            'labels': all_labels.numpy()
        }
        
        wandb.log({
            f"test_accuracy_{method}": acc,
            f"test_f1_{method}": f1,
            f"test_precision_{method}": precision,
            f"test_recall_{method}": recall,
            f"test_auc_{method}": auc_score
        })
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_labels.numpy(), all_preds.numpy())
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {method}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        roc_path = os.path.join(args.plots_dir, f"{exp_name}_{method}_roc.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        wandb.log({f"roc_curve_{method}": wandb.Image(roc_path)})
        
        # Plot Precision-Recall curve
        precision_values, recall_values, _ = precision_recall_curve(all_labels.numpy(), all_preds.numpy())
        pr_auc = auc(recall_values, precision_values)
        plt.figure(figsize=(10, 8))
        plt.plot(recall_values, precision_values, label=f'PR AUC = {pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {method}')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        pr_path = os.path.join(args.plots_dir, f"{exp_name}_{method}_pr_curve.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        wandb.log({f"pr_curve_{method}": wandb.Image(pr_path)})
    
    # Compare methods
    if len(args.inference_methods) > 1:
        plt.figure(figsize=(12, 10))
        for method in args.inference_methods:
            fpr, tpr, _ = roc_curve(results[method]['labels'], results[method]['predictions'])
            plt.plot(fpr, tpr, label=f'{method} (AUC = {results[method]["auc"]:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        compare_path = os.path.join(args.plots_dir, f"{exp_name}_methods_comparison.png")
        plt.savefig(compare_path, dpi=300, bbox_inches='tight')
        plt.close()
        wandb.log({"methods_comparison": wandb.Image(compare_path)})
        
        # Bar chart of metrics
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
        methods = args.inference_methods
        x = np.arange(len(metrics))
        width = 0.8 / len(methods)
        
        plt.figure(figsize=(14, 8))
        for i, method in enumerate(methods):
            values = [results[method][metric] for metric in metrics]
            plt.bar(x + i*width - (len(methods)-1)*width/2, values, width, label=method)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        metrics_path = os.path.join(args.plots_dir, f"{exp_name}_metrics_comparison.png")
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        wandb.log({"metrics_comparison": wandb.Image(metrics_path)})
    
    wandb.finish()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()