from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import torchmetrics
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from datasets import MultiTaskDataset
from models import MultiTaskResNet18, MultiTaskResNet50, UNIMultitask, MultiTaskEfficientNet
import torch.optim as optim
import argparse
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='MultiTask Learning for Pathology Image Classification')
    
    # Dataset and task parameters
    parser.add_argument('--classification-task', type=str, default='tumor', 
                        help='Classification task: tumor or TIL')
    parser.add_argument('--testset', type=str, default='ocelot', 
                        help='Dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)')
    parser.add_argument('--dataset-path', type=str, default='dataset/full_dataset.csv',
                        help='Path to the full dataset CSV file')
    parser.add_argument('--clustering-path', type=str, default='clustering/output',
                        help='Path to the clustering results directory')
    parser.add_argument('--sample-size', type=float, default=0.9,
                        help='Fraction of data to use for training')
    
    # Model parameters
    parser.add_argument('--multitask', type=bool, default=True, 
                        help="Enable multitask model")
    parser.add_argument('--retrain', type=bool, default=True, 
                        help="Enable retrain base model")
    parser.add_argument('--model', type=str, default="ResNet18", 
                        help="Backbone: ResNet18, ResNet50, or EfficientNet")
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, 
                        help="Batch size per GPU")
    parser.add_argument('--epochs', type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument('--train-split', type=float, default=0.8, 
                        help="Training/validation split ratio")
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument('--crop-size', type=int, default=96,
                        help="Size of image crops")
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='saved_models',
                        help="Directory to save trained models")
    parser.add_argument('--project-name', type=str, default='ColorBasedMultitask-full',
                        help="Project name for experiment tracking")
    
    return parser.parse_args()

def setup_device():
    """Set up the device for training."""
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return f"no-aug-{args.crop_size}_{multitask}_{args.model}_{args.classification_task}_{args.testset}_{timestamp}"

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix.
    
    Args:
        cm: The confusion matrix to plot
        class_names: Names of the classes
    """
    # Normalize the confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
    
    df_cm = pd.DataFrame(cm_normalized, class_names, class_names)
    print(df_cm)
    
    # Optional: Create and save visualization
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(df_cm, annot=True, cmap="Blues")
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title('Confusion Matrix')
    # plt.savefig(f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

def train_epoch(model, optimizer, weights, dataloaders, num_tasks, device):
    """Train the model for one epoch.
    
    Args:
        model: The model to train
        optimizer: The optimizer
        weights: Class weights for loss function
        dataloaders: Dictionary of dataloaders
        num_tasks: Number of tasks
        device: Device to train on
        
    Returns:
        Dictionary of training metrics
    """
    model.train()

    # Initialize metrics
    epoch_loss = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
        
    for task_idx in range(num_tasks):
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights[task_idx])

        for images, labels, _ in dataloaders[task_idx]['train']:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)[task_idx].squeeze()

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_loss(loss)
            acc_metric(outputs, labels)
            uar_metric(outputs, labels)    

    # Calculate epoch metrics
    metrics_dict = {
        'Loss_train': epoch_loss.compute().item(),
        'Accuracy_train': acc_metric.compute().item(),
        'UAR_train': uar_metric.compute().item(),
    }

    return metrics_dict

def val_epoch(model, weights, dataloaders, num_tasks, device):
    """Evaluate the model on the validation set.
    
    Args:
        model: The model to evaluate
        weights: Class weights for loss function
        dataloaders: Dictionary of dataloaders
        num_tasks: Number of tasks
        device: Device to evaluate on
        
    Returns:
        Dictionary of validation metrics and confusion matrix
    """
    model.eval()
    
    # Initialize metrics
    epoch_loss = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    
    # Initialize confusion matrix
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():
        for task_idx in range(num_tasks):
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights[task_idx])

            for images, labels, _ in dataloaders[task_idx]['val']:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)[task_idx].squeeze()
                loss = criterion(outputs, labels)

                # Accumulate metrics
                epoch_loss(loss)
                acc_metric(outputs, labels)
                uar_metric(outputs, labels)
            
                # Accumulate confusion matrix 
                confusion_matrix(outputs, labels)

    # Calculate validation metrics
    metrics_dict = {
        'Loss_val': epoch_loss.compute().item(),
        'Accuracy_val': acc_metric.compute().item(),
        'UAR_val': uar_metric.compute().item(),
    }

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def train_model(model, dataloaders, optimizer, weights, args, device, class_names):   
    """Train the model for multiple epochs.
    
    Args:
        model: The model to train
        dataloaders: Dictionary of dataloaders
        optimizer: The optimizer
        weights: Class weights for loss function
        args: Command line arguments
        device: Device to train on
        class_names: Names of the classes
    """
    model.to(device)
    num_tasks = len(weights)
    
    # Train by iterating over epochs
    for epoch in tq.tqdm(range(args.epochs), total=args.epochs, desc='Epochs'):
        train_metrics_dict = train_epoch(model, optimizer, weights, dataloaders, num_tasks, device)
                
        val_metrics_dict, cm = val_epoch(model, weights, dataloaders, num_tasks, device)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_metrics_dict['Loss_train']:.4f}, Train Acc: {train_metrics_dict['Accuracy_train']:.4f}")
        print(f"Val Loss: {val_metrics_dict['Loss_val']:.4f}, Val Acc: {val_metrics_dict['Accuracy_val']:.4f}")
        
        # Uncomment to log to wandb
        # wandb.log({**train_metrics_dict, **val_metrics_dict})

    # Plot confusion matrix from results of last val epoch
    plot_confusion_matrix(cm, class_names)

def read_data(args):
    """Read and preprocess the dataset.
    
    Args:
        args: Command line arguments
        
    Returns:
        Preprocessed DataFrame
    """
    # Read the full dataset
    df = pd.read_csv(args.dataset_path)

    # Read clustering results
    clustering_file = f"{args.clustering_path}/clustering_result_{args.classification_task}_{args.testset}.csv"
    df_clustering = pd.read_csv(clustering_file)

    # Extract image name from path
    df_clustering['img_name'] = df_clustering.apply(lambda row: os.path.basename(row['img_path'])[:-4], axis=1)

    # Merge datasets
    df_merged = pd.merge(df_clustering, df, left_on=['dataset', 'img_name'], 
                         right_on=['dataset', 'img_name'], how='left')
    
    # Filter only needed columns
    df_merged_filtered = df_merged[["dataset", "img_name", "centerX", "centerY", 
                                   "labelTIL", "labelTumor", "labelCluster"]]

    # Remove testset data
    df_merged_filtered = df_merged_filtered[df_merged_filtered.dataset != args.testset]
    
    # Remove rows with missing values
    df_merged_filtered.dropna(inplace=True)

    return df_merged_filtered

def create_datasets_and_weights(df, args, device):
    """Create datasets and class weights.
    
    Args:
        df: Preprocessed DataFrame
        args: Command line arguments
        device: Device for training
        
    Returns:
        List of datasets and weights
    """
    datasets = []
    weights = []

    # Determine number of tasks
    if args.multitask:
        num_tasks = len(df.labelCluster.unique())  # Number of clusters/heads
    else:
        num_tasks = 1  # Single head model

    for i in range(num_tasks):
        if args.multitask:
            df_filtered = df[df.labelCluster == i].reset_index()
        else:
            df_filtered = df.copy().reset_index()
        
        # Optional: Sample data
        if args.sample_size < 1.0:
            # Create a stratification key
            label_col = 'labelTumor' if args.classification_task == 'tumor' else 'labelTIL'
            df_filtered['stratify_key'] = df_filtered.apply(
                lambda row: f"{row[label_col]}_{row['dataset']}", axis=1
            )
            
            # Stratified sampling
            df_filtered, _ = train_test_split(
                df_filtered, 
                train_size=args.sample_size, 
                stratify=df_filtered['stratify_key'], 
                random_state=args.seed
            )
        
        # Create dataset
        dataset = MultiTaskDataset(df_filtered, args.classification_task, crop_size=args.crop_size)
        
        # Calculate class weights
        w = torch.tensor([dataset.pos_weight], dtype=torch.float32, device=device)

        datasets.append(dataset)
        weights.append(w)

    return datasets, weights, num_tasks

def create_dataloaders(datasets, args):
    """Create dataloaders for training and validation.
    
    Args:
        datasets: List of datasets
        args: Command line arguments
        
    Returns:
        Dictionary of dataloaders
    """
    dataloaders = {}
    
    for task_idx, dataset in enumerate(datasets):
        # Split into train and validation sets
        train_size = int(args.train_split * len(dataset))
        val_size = len(dataset) - train_size
        
        # Use fixed seed for reproducibility
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )        

        # Create dataloaders
        dataloaders[task_idx] = {
            'train': DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                drop_last=True, 
                num_workers=2, 
                pin_memory=True
            ),
            'val': DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                drop_last=True, 
                num_workers=2, 
                pin_memory=True
            )
        }
    
    return dataloaders

def create_model(args, num_tasks):
    """Create and configure the model.
    
    Args:
        args: Command line arguments
        num_tasks: Number of tasks
        
    Returns:
        Configured model
    """
    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks, retrain=args.retrain)
    elif args.model == "ResNet18":
        model = MultiTaskResNet18(num_tasks=num_tasks, retrain=args.retrain)
    elif args.model == "EfficientNet":
        model = MultiTaskEfficientNet(num_tasks=num_tasks)
    else:
        model = UNIMultitask(num_tasks=num_tasks)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    return model

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up device
    device = setup_device()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Generate experiment name
    exp_name = generate_experiment_name(args)
    print(f"Experiment: {exp_name}")
    
    # Initialize experiment tracking (wandb)
    # run = wandb.init(project=args.project_name, name=exp_name)
    
    # Read and preprocess data
    df = read_data(args)
    
    # Create datasets and weights
    datasets, weights, num_tasks = create_datasets_and_weights(df, args, device)
    
    # Create dataloaders
    dataloaders = create_dataloaders(datasets, args)
    
    # Create model
    model = create_model(args, num_tasks)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Define class names
    class_names = ["non-tumor", "tumor"] if args.classification_task == "tumor" else ["non-TIL", "TIL"]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    train_model(model, dataloaders, optimizer, weights, args, device, class_names)
    
    # Save model
    model_path = os.path.join(args.output_dir, f"{exp_name}.pth")
    
    # If using DataParallel, save the module
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")
    
    # Finish experiment tracking
    # run.finish()

if __name__ == '__main__':
    main()
