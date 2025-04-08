import argparse
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
import pandas as pd
import os
import re
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import wandb
from datasets import MultiTaskDataset
from models import UNIMultitask

def parse_arguments():
    parser = argparse.ArgumentParser(description='MultiTask Learning Model Testing')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--inference_path', type=str, required=True, help='Path to inference results')
    parser.add_argument('--models_dir', type=str, default='./models', help='Directory with trained models')
    parser.add_argument('--plots_dir', type=str, default='./plots', help='Directory to save plots')
    parser.add_argument('--classification_task', type=str, default='tumor', choices=['tumor', 'TIL'], help='Classification task')
    parser.add_argument('--testset', type=str, required=True, help='Test dataset name')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size for images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--multitask', action='store_true', help='Enable multitask learning')
    parser.add_argument('--inference_methods', type=str, nargs='+', default=['cluster'], 
                        help='List of inference methods: cluster, weighted_voting, weighted_sum')
    return parser.parse_args()

def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def test_model(model, dataloader, device, method, num_tasks):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels, clusters in dataloader:
            images, labels, clusters = images.to(device), labels.to(device), clusters.to(device)
            outputs, cluster_probs = model(images)
            if method == 'cluster':
                unique_clusters = torch.unique(clusters)
                if max(unique_clusters) >= num_tasks:
                    raise ValueError(f"Cluster index {max(unique_clusters)} exceeds num_tasks {num_tasks}")
                selected_outputs = torch.stack([outputs[cluster][i] for i, cluster in enumerate(clusters)])
                preds = torch.sigmoid(selected_outputs.view(-1))
            elif method == 'weighted_voting':
                outputs_tensor = torch.stack(outputs, dim=1).squeeze(-1)
                probs = torch.sigmoid(outputs_tensor)
                binary_preds = (probs > 0.5).float()
                preds = (binary_preds * probs).sum(dim=1) / probs.sum(dim=1)
            elif method == 'weighted_sum':
                outputs_tensor = torch.stack(outputs, dim=1).squeeze(-1)
                preds = torch.sigmoid((outputs_tensor * cluster_probs).sum(dim=1))
            else:
                raise ValueError(f"Unsupported inference method: {method}")
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds, all_labels

def main():
    args = parse_arguments()
    
    ALLOWED_METHODS = ['cluster', 'weighted_voting', 'weighted_sum']
    for method in args.inference_methods:
        if method not in ALLOWED_METHODS:
            raise ValueError(f"Invalid inference method: {method}. Allowed methods are {ALLOWED_METHODS}")
    
    device = setup_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.plots_dir, exist_ok=True)
    exp_name = f"UNI_{args.classification_task}_{args.testset}"

    wandb.init(project="pathology_multitask_test", name=f"{exp_name}_test")

    # Load test data with cluster labels
    df = pd.read_csv(args.dataset_path)
    df = df[df.dataset == args.testset].reset_index(drop=True)
    inference_file = f"{args.inference_path}/inference_{args.classification_task}_{args.testset}.csv"
    df_inference = pd.read_csv(inference_file)
    df_test = df.merge(df_inference[['img_path', 'labelCluster']], on='img_path', how='inner')

    # Load training data to get the number of tasks (clusters) used during training
    df_dataset = pd.read_csv(args.dataset_path)
    cluster_file = os.path.join(args.inference_path.replace('clustering_updated/inference_results', 'clustering/output'), 
                               f"clustering_result_{args.classification_task}_{args.testset}.csv")
    df_cluster = pd.read_csv(cluster_file)
    df_train = df_dataset.merge(df_cluster[['dataset', 'img_path', 'labelCluster']], 
                                on=['dataset', 'img_path'], 
                                how='inner')
    train_val_df = df_train[df_train.dataset != args.testset].reset_index(drop=True)
    
    # Use the number of clusters from training data
    unique_train_clusters = sorted(train_val_df['labelCluster'].unique())
    num_tasks = len(unique_train_clusters) if args.multitask else 1
    if args.multitask:
        print(f"Multi-task mode enabled. Number of tasks: {num_tasks}, Unique clusters from training: {unique_train_clusters}")
    else:
        print("Single-task mode enabled. Number of tasks: 1")

    # Check test data clusters
    unique_test_clusters = sorted(df_test['labelCluster'].unique())
    print(f"Test data clusters: {unique_test_clusters}")
    if max(unique_test_clusters) >= num_tasks:
        raise ValueError(f"Test data contains cluster indices {unique_test_clusters} that exceed num_tasks {num_tasks}")

    dataset = MultiTaskDataset(df_test, args.classification_task, crop_size=args.crop_size)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=min(8, os.cpu_count()), pin_memory=True)

    # Load best model
    model = UNIMultitask(num_tasks=num_tasks)
    best_model_path = find_best_model(args.models_dir, exp_name)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # Test each inference method
    for method in args.inference_methods:
        print(f"Testing with method: {method}")
        all_preds, all_labels = test_model(model, dataloader, device, method, num_tasks)

        acc = torchmetrics.functional.accuracy(all_preds > 0.5, all_labels.int(), task='binary')
        f1 = torchmetrics.functional.f1_score(all_preds > 0.5, all_labels.int(), task='binary')
        auc = torchmetrics.functional.auroc(all_preds, all_labels.int(), task='binary')
        print(f"Method: {method}, Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        wandb.log({
            f"test_accuracy_{method}": acc,
            f"test_f1_{method}": f1,
            f"test_auc_{method}": auc
        })

        fpr, tpr, _ = roc_curve(all_labels.cpu(), all_preds.cpu())
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {method}')
        plt.legend()
        roc_path = os.path.join(args.plots_dir, f"{exp_name}_{method}_roc.png")
        plt.savefig(roc_path)
        plt.close()
        wandb.log({f"roc_curve_{method}": wandb.Image(roc_path)})

    wandb.finish()

if __name__ == "__main__":
    main()
    