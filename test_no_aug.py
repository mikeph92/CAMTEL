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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='MultiTask Learning Model Testing')
    
    # Dataset and task parameters
    parser.add_argument('--classification-task', type=str, default='tumor', 
                        help='Classification task: tumor or TIL')
    parser.add_argument('--testset', type=str, default='ocelot', 
                        help='Dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)')
    parser.add_argument('--dataset-path', type=str, default='dataset/full_dataset.csv',
                        help='Path to the full dataset CSV file')
    parser.add_argument('--clustering-path', type=str, default='clustering/output',
                        help='Path to the clustering results directory')
    parser.add_argument('--statistics-path', type=str, default='clustering/output/img_statistics.csv',
                        help='Path to the image statistics CSV file')
    
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
    """Plot confusion matrix.
    
    Args:
        cm: The confusion matrix to plot
        class_names: Names of the classes
        output_path: Path to save the plot (optional)
    """
    # Normalize the confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
    
    df_cm = pd.DataFrame(cm_normalized, class_names, class_names)
    print(df_cm)
    
    # Save visualization if path is provided
    if output_path:
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap='flare', fmt='.2f')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(output_path)
        plt.close()

def predict_cluster(df_train):
    """Train a Random Forest classifier to predict clusters.
    
    Args:
        df_train: DataFrame containing training data with cluster labels
        
    Returns:
        Scaler and trained classifier
    """
    X_train = df_train[["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]].values
    y_train = df_train.labelCluster

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Initialize and train the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return scaler, clf

def test_by_cluster(model, dataloader, df_test, device, args):
    """Test model using cluster-based approach.
    
    Args:
        model: The model to test
        dataloader: DataLoader for test data
        df_test: DataFrame with test data and cluster labels
        device: Device for testing
        args: Command line arguments
        
    Returns:
        Dictionary of metrics and confusion matrix
    """
    model.eval()
    
    # Initialize metrics
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    roc_auc_metric = torchmetrics.AUROC(task="binary").to(device)
    
    # Initialize confusion matrix
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():
        for images, labels, img_names in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            # Get cluster for each image in batch
            clusters = np.array([df_test.loc[df_test.img_name == img_name, 'labelCluster'].iloc[0] 
                                for img_name in img_names])

            # Initialize outputs
            outputs = torch.zeros(len(images), dtype=torch.float32).to(device)
            
            # Process each cluster
            for cluster in np.unique(clusters):
                # Select indices for each cluster
                cluster_indices = np.where(clusters == cluster)[0]
                
                # Perform batch prediction for images in this cluster
                cluster_images = images[cluster_indices]
                
                # Handle DataParallel model
                if isinstance(model, nn.DataParallel):
                    cluster_outputs = model.module(cluster_images)[cluster]
                else:
                    cluster_outputs = model(cluster_images)[cluster]
                    
                outputs[cluster_indices] = torch.sigmoid(cluster_outputs).squeeze()

            # Calculate metrics
            acc_metric((outputs > 0.5).int(), labels)
            uar_metric((outputs > 0.5).int(), labels)
            f1_metric((outputs > 0.5).int(), labels)
            roc_auc_metric(outputs, labels)  # ROC-AUC uses raw sigmoid values
            confusion_matrix((outputs > 0.5).int(), labels)

    # Calculate test metrics
    metrics_dict = {
        'Accuracy_test': acc_metric.compute().item(),
        'UAR_test': uar_metric.compute().item(),
        'F1_test': f1_metric.compute().item(),
        'AUC_ROC_test': roc_auc_metric.compute().item(),
    }

    # Save results
    save_results(metrics_dict, "cluster based", args)

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def test_by_mv(model, dataloader, device, args, num_tasks):
    """Test model using majority vote approach.
    
    Args:
        model: The model to test
        dataloader: DataLoader for test data
        device: Device for testing
        args: Command line arguments
        num_tasks: Number of tasks/heads
        
    Returns:
        Dictionary of metrics and confusion matrix
    """
    model.eval()
    
    # Initialize metrics
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    roc_auc_metric = torchmetrics.AUROC(task="binary").to(device)
    
    # Initialize confusion matrix
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            # Handle DataParallel model
            if isinstance(model, nn.DataParallel):
                batch_predictions = model.module(images)
            else:
                batch_predictions = model(images)
                
            # Convert to binary predictions
            binary_predictions = torch.tensor(
                [[1.0 if torch.sigmoid(value) > 0.5 else 0.0 for value in sublist] 
                  for sublist in batch_predictions], 
                device=device
            )

            # Get majority vote
            outputs, _ = torch.mode(binary_predictions, dim=0)

            # Calculate metrics
            acc_metric(outputs, labels)
            uar_metric(outputs, labels)
            f1_metric(outputs, labels)
            roc_auc_metric(outputs, labels)
            confusion_matrix(outputs, labels)

    # Calculate test metrics
    metrics_dict = {
        'Accuracy_test': acc_metric.compute().item(),
        'UAR_test': uar_metric.compute().item(),
        'F1_test': f1_metric.compute().item(),
        'AUC_ROC_test': roc_auc_metric.compute().item(),
    }

    # Save results
    save_results(metrics_dict, "majority vote", args)

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def save_results(metrics_dict, method, args):
    """Save test results to JSON file.
    
    Args:
        metrics_dict: Dictionary of metrics
        method: Testing method
        args: Command line arguments
    """
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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write results to JSON file
    with open(os.path.join(args.output_dir, "test_results_no_aug.json"), "a") as f:
        f.write(json.dumps(results) + '\n')

def load_model(args, num_tasks, device):
    """Load the trained model.
    
    Args:
        args: Command line arguments
        num_tasks: Number of tasks/heads
        device: Device for testing
        
    Returns:
        Loaded model
    """
    multitask = "Multitask" if args.multitask else "Single"
    
    # Find model file
    model_pattern = f'{args.models_dir}/no-aug-{args.crop_size}_{multitask}_{args.model}_{args.classification_task}_{args.testset}*'
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found matching pattern: {model_pattern}")
    
    # Load state dictionary
    state_dict = torch.load(model_files[0], map_location=device)

    # Create model
    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks)
    elif args.model == "ResNet18":
        model = MultiTaskResNet18(num_tasks=num_tasks, retrain=True)
    elif args.model == "EfficientNet":
        model = MultiTaskEfficientNet(num_tasks=num_tasks)
    else:
        model = UNIMultitask(num_tasks=num_tasks)

    # Load weights
    model.load_state_dict(state_dict)
    
    # Wrap with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    return model

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup device
    device = setup_device()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Generate experiment name
    exp_name = generate_experiment_name(args)
    print(f"Experiment: {exp_name}")
    
    # Set project name if not provided
    if args.project_name is None:
        args.project_name = f"FULL-ColorBasedMultitask-Test-{args.crop_size}-{args.model}-no-aug"
    
    # Initialize experiment tracking (wandb)
    # run = wandb.init(project=args.project_name, name=exp_name)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Load test data
    df = pd.read_csv(args.dataset_path)
    df = df[df.dataset == args.testset].reset_index()

    # Optional: Subsample test data
    if args.sample < 1.0:
        stratifier = "labelTumor" if args.classification_task == "tumor" else "labelTIL"
        df, _ = train_test_split(df, train_size=args.sample, stratify=df[stratifier], random_state=args.seed)

    # Create test dataset and dataloader
    dataset = MultiTaskDataset(df, args.classification_task, crop_size=args.crop_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=True, 
        num_workers=2, 
        pin_memory=True
    )

    # Load training data with clusters
    df_train = pd.read_csv(f"{args.clustering_path}/clustering_result_{args.classification_task}_{args.testset}.csv")
    
    # Determine number of tasks
    global num_tasks
    num_tasks = 1
    if args.multitask:
        num_tasks = len(df_train.labelCluster.unique())
    
    # Load model
    model = load_model(args, num_tasks, device)
    
    # Define class names
    class_names = ["non-tumor", "tumor"] if args.classification_task == "tumor" else ["non-TIL", "TIL"]

    # Test the model
    if args.multitask:
        if args.test_method == "cluster":
            # Train cluster predictor
            scaler, predictor = predict_cluster(df_train)

            # Preprocess test images
            df_statistics = pd.read_csv(args.statistics_path)
            df_test = df_statistics[df_statistics.dataset == args.testset].reset_index()
            test_data = df_test[["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]]
            
            # Predict clusters for test data
            clusters = predictor.predict(scaler.transform(test_data))
            df_test['labelCluster'] = clusters

            # Test using cluster-based approach
            test_metrics_dict, cm = test_by_cluster(model, dataloader, df_test, device, args)
        else:  # args.test_method == "mv"
            # Test using majority vote approach
            test_metrics_dict, cm = test_by_mv(model, dataloader, device, args, num_tasks)
    else:
        # Single-task model always uses majority vote
        test_metrics_dict, cm = test_by_mv(model, dataloader, device, args, num_tasks)

    # Log metrics (if using wandb)
    # wandb.log({**test_metrics_dict})

    # Plot confusion matrix
    # plot_path = os.path.join(args.plots_dir, f"{exp_name}.png")
    # plot_confusion_matrix(cm, class_names, plot_path)
    
    # Print results
    print("\nTest Results:")
    for metric, value in test_metrics_dict.items():
        print(f"{metric}: {value:.4f}")
    
    # Finish experiment tracking
    # run.finish()

if __name__ == '__main__':
    main()
