from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import torchmetrics
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets import MultiTaskDataset
from models import MultiTaskResNet50, MultiTaskResNet18, UNIMultitask, MultiTaskEfficientNet
import argparse
import glob
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='MultiTask Learning Model Testing')
    
    # Dataset and task parameters
    parser.add_argument('--classification-task', type=str, default='tumor', 
                        help='Classification task: tumor or TIL')
    parser.add_argument('--testset', type=str, default='ocelot', 
                        help='Dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)')
    parser.add_argument('--dataset-path', type=str, default='dataset/tumor_dataset.csv',
                        help='Path to the full dataset CSV file')
    parser.add_argument('--clustering-path', type=str, default='clustering/output',
                        help='Path to the clustering results directory')
    parser.add_argument('--inference-path', type=str, default='clustering_updated/inference_results',
                        help='Path to the inference results directory')
    
    # Model parameters
    parser.add_argument('--multitask', type=bool, default=True, 
                        help="Enable multitask model")
    parser.add_argument('--test-method', type=str, default='cluster', 
                        choices=['cluster', 'mv'],
                        help='Testing method: cluster (cluster-based) or mv (majority vote)')
    parser.add_argument('--model', type=str, default="ResNet18", 
                        help="Backbone: ResNet18, ResNet50, or EfficientNet")
    
    # Testing parameters
    parser.add_argument('--batch-size', type=int, default=64, 
                        help="Batch size for testing")
    parser.add_argument('--sample', type=float, default=0.9,
                        help="Sample size for testing (if subsampling is needed)")
    parser.add_argument('--crop-size', type=int, default=48,
                        help="Size of image crops")
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Output parameters
    parser.add_argument('--models-dir', type=str, default='saved_models',
                        help="Directory containing trained models")
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help="Directory to save test results")
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help="Directory to save plots")
    parser.add_argument('--project-name', type=str, 
                        default=None,
                        help="Project name for experiment tracking")
    
    return parser.parse_args()

def setup_device():
    """Set up the device for testing."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using {torch.cuda.device_count()} GPU(s)")
        return device
    else:
        print("Using CPU")
        return torch.device("cpu")

def generate_experiment_name(args):
    """Generate a unique experiment name based on parameters."""
    multitask = "Multitask" if args.multitask else "Single"
    method = "" if not args.multitask else f'_{args.test_method}'
    
    return f"{args.classification_task}_{args.testset}_{multitask}{method}"

def plot_confusion_matrix(cm, class_names, output_path=None):
    """Plot confusion matrix."""
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
    
    df_cm = pd.DataFrame(cm_normalized, class_names, class_names)
    print(df_cm)
    
    if output_path:
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap='flare', fmt='.2f')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(output_path)
        plt.close()

def test_by_cluster(model, dataloader, device, args):
    """Test model using cluster-based approach."""
    model.eval()
    
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    roc_auc_metric = torchmetrics.AUROC(task="binary").to(device)
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():
        for images, labels, clusters in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = torch.zeros(len(images), dtype=torch.float32).to(device)
            
            for cluster in np.unique(clusters):
                cluster_indices = np.where(clusters == cluster)[0]
                cluster_images = images[cluster_indices]
                
                if isinstance(model, nn.DataParallel):
                    cluster_outputs = model.module(cluster_images)[cluster]
                else:
                    cluster_outputs = model(cluster_images)[cluster]
                    
                outputs[cluster_indices] = torch.sigmoid(cluster_outputs).squeeze()

            acc_metric((outputs > 0.5).int(), labels)
            uar_metric((outputs > 0.5).int(), labels)
            f1_metric((outputs > 0.5).int(), labels)
            roc_auc_metric(outputs, labels)
            confusion_matrix((outputs > 0.5).int(), labels)

    metrics_dict = {
        'Accuracy_test': acc_metric.compute().item(),
        'UAR_test': uar_metric.compute().item(),
        'F1_test': f1_metric.compute().item(),
        'AUC_ROC_test': roc_auc_metric.compute().item(),
    }

    save_results(metrics_dict, "cluster based", args)
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def test_by_mv(model, dataloader, device, args, num_tasks):
    """Test model using majority vote approach."""
    model.eval()
    
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    roc_auc_metric = torchmetrics.AUROC(task="binary").to(device)
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            if isinstance(model, nn.DataParallel):
                batch_predictions = model.module(images)
            else:
                batch_predictions = model(images)
                
            binary_predictions = torch.tensor(
                [[1.0 if torch.sigmoid(value) > 0.5 else 0.0 for value in sublist] 
                  for sublist in batch_predictions], 
                device=device
            )

            outputs, _ = torch.mode(binary_predictions, dim=0)

            acc_metric(outputs, labels)
            uar_metric(outputs, labels)
            f1_metric(outputs, labels)
            roc_auc_metric(outputs, labels)
            confusion_matrix(outputs, labels)

    metrics_dict = {
        'Accuracy_test': acc_metric.compute().item(),
        'UAR_test': uar_metric.compute().item(),
        'F1_test': f1_metric.compute().item(),
        'AUC_ROC_test': roc_auc_metric.compute().item(),
    }

    save_results(metrics_dict, "majority vote", args)
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def save_results(metrics_dict, method, args):
    """Save test results to JSON file."""
    results = {
        "model": args.model,
        "task": args.classification_task,
        "testset": args.testset,
        "augmented": "No",
        "method": method,
        "num_tasks": num_tasks,
        "crop_size": args.crop_size,
        "test_datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    results.update(metrics_dict)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "test_results_no_aug.json"), "a") as f:
        f.write(json.dumps(results) + '\n')

def load_model(args, num_tasks, device):
    """Load the trained model."""
    multitask = "Multitask" if args.multitask else "Single"
    
    model_pattern = f'{args.models_dir}/no-aug-{args.crop_size}_{multitask}_{args.model}_{args.classification_task}_{args.testset}*'
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found matching pattern: {model_pattern}")
    
    state_dict = torch.load(model_files[0], map_location=device)

    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks)
    elif args.model == "ResNet18":
        model = MultiTaskResNet18(num_tasks=num_tasks, retrain=True)
    elif args.model == "EfficientNet":
        model = MultiTaskEfficientNet(num_tasks=num_tasks)
    else:
        model = UNIMultitask(num_tasks=num_tasks)

    model.load_state_dict(state_dict)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    return model

def main():
    args = parse_arguments()
    device = setup_device()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    exp_name = generate_experiment_name(args)
    print(f"Experiment: {exp_name}")
    
    if args.project_name is None:
        args.project_name = f"FULL-ColorBasedMultitask-Test-{args.crop_size}-{args.model}-no-aug"
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Load test data
    df = pd.read_csv(args.dataset_path)
    df = df[df.dataset == args.testset].reset_index()

    if args.sample < 1.0:
        stratifier = "label"
        df, _ = train_test_split(df, train_size=args.sample, stratify=df[stratifier], random_state=args.seed)

    # Load inference results
    inference_file_pattern = f"{args.inference_path}/inference_{args.classification_task}_{args.testset}.csv"
    inference_files = glob.glob(inference_file_pattern)
    if not inference_files:
        raise FileNotFoundError(f"No inference file found matching pattern: {inference_file_pattern}")
    inference_file = inference_files[0]  # Use the first match
    df_inference = pd.read_csv(inference_file)
    
    # Join test data with inference results
    df_test = df.merge(df_inference[['img_path', 'labelCluster']], 
                      left_on='img_path', 
                      right_on='img_path', 
                      how='inner')
    if df_test.empty:
        raise ValueError(f"No matching images found between test dataset and inference results for {args.testset}")

    # Create test dataset and dataloader
    dataset = MultiTaskDataset(df_test, args.classification_task, crop_size=args.crop_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=True, 
        num_workers=2, 
        pin_memory=True
    )

    # Load training data with clusters (for num_tasks calculation)
    df_train = pd.read_csv(f"{args.clustering_path}/clustering_result_{args.classification_task}_{args.testset}.csv")
    
    # Determine number of tasks
    global num_tasks
    num_tasks = 1
    if args.multitask:
        num_tasks = len(df_train.labelCluster.unique())
    
    # Load model
    model = load_model(args, num_tasks, device)
    
    class_names = ["non-tumor", "tumor"] if args.classification_task == "tumor" else ["non-TIL", "TIL"]

    # Test the model
    if args.multitask:
        if args.test_method == "cluster":
            test_metrics_dict, cm = test_by_cluster(model, dataloader, device, args)
        else:  # args.test_method == "mv"
            test_metrics_dict, cm = test_by_mv(model, dataloader, device, args, num_tasks)
    else:
        test_metrics_dict, cm = test_by_mv(model, dataloader, device, args, num_tasks)

    # Plot confusion matrix (uncomment if needed)
    # plot_path = os.path.join(args.plots_dir, f"{exp_name}.png")
    # plot_confusion_matrix(cm, class_names, plot_path)
    
    print("\nTest Results:")
    for metric, value in test_metrics_dict.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
    