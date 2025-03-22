from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import argparse
from tqdm import tqdm
from datasets import RandStainNADataset
from models import MultiTaskResNet50, MultiTaskResNet18, UNIMultitask, MultiTaskEfficientNet
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryConfusionMatrix

parser = argparse.ArgumentParser(description='Multi-task Learning for Histopathology Images')
parser.add_argument('--classification-task', type=str, default='tumor', help='classification task: tumor or TIL') 
parser.add_argument('--testset', type=str, default='ocelot', help='dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)') 
parser.add_argument('--sample-size', type=float, default=0.9)
parser.add_argument('--multitask', type=bool, default=True, help="Enable use multitask model")
parser.add_argument('--crop-size', type=int, default=96)
parser.add_argument('--model', type=str, default="ResNet18", help="backbone ResNet18, ResNet50, or EfficientNet")
parser.add_argument('--batch-size', type=int, default=32, help="Batch size per GPU")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--train-split', type=float, default=0.8, help="Training/validation split ratio")
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

args = parser.parse_args()

# Create a timestamp for model naming
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
multitask_str = "Multitask" if args.multitask else "Single"
exp_name = f"{multitask_str}_{args.model}_{args.classification_task}_{args.testset}_{timestamp}"

# Setup for multi-GPU training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.device_count() > 1
if multi_gpu:
    print(f"Using {torch.cuda.device_count()} GPUs for training")

# Ensure output directories exist
os.makedirs('saved_models', exist_ok=True)

def train_epoch(model, optimizer, weights, dataloaders, num_tasks, device):
    model.train()

    # Initialize metrics
    epoch_loss = MeanMetric().to(device)
    acc_metric = BinaryAccuracy().to(device)
    uar_metric = BinaryRecall().to(device)
        
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
    '''
    Evaluate the model on the entire validation set.
    '''
    model.eval()
    
    # Initialize metrics
    epoch_loss = MeanMetric().to(device)
    acc_metric = BinaryAccuracy().to(device)
    uar_metric = BinaryRecall().to(device)
    confusion_matrix = BinaryConfusionMatrix().to(device)

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
                confusion_matrix(outputs, labels)
         
    # Calculate epoch metrics
    metrics_dict = {
        'Loss_val': epoch_loss.compute().item(),
        'Accuracy_val': acc_metric.compute().item(),
        'UAR_val': uar_metric.compute().item(),
    }

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def train_model(model, dataloaders, optimizer, weights, n_epochs, device):
    # Move model to device and wrap with DataParallel if multiple GPUs available
    model.to(device)
    if multi_gpu:
        model = nn.DataParallel(model)
    
    best_val_acc = 0.0
    
    # Train by iterating over epochs
    for epoch in tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
        train_metrics_dict = train_epoch(model, optimizer, weights, dataloaders, num_tasks, device)
        val_metrics_dict, _ = val_epoch(model, weights, dataloaders, num_tasks, device)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train: Loss {train_metrics_dict['Loss_train']:.4f}, Acc {train_metrics_dict['Accuracy_train']:.4f}, UAR {train_metrics_dict['UAR_train']:.4f}")
        print(f"Val: Loss {val_metrics_dict['Loss_val']:.4f}, Acc {val_metrics_dict['Accuracy_val']:.4f}, UAR {val_metrics_dict['UAR_val']:.4f}")
        
        # Save best model
        if val_metrics_dict['Accuracy_val'] > best_val_acc:
            best_val_acc = val_metrics_dict['Accuracy_val']
            torch.save(model.state_dict(), f'saved_models/pretrained_{exp_name}_best.pth')
    
    # Save final model
    torch.save(model.state_dict(), f'saved_models/pretrained_{exp_name}_final.pth')
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

def read_data(task, testset):
    df = pd.read_csv("dataset/full_dataset.csv")
    df_clustering = pd.read_csv(f"clustering/output/clustering_result_{task}_{testset}.csv")
    
    # Extract image name from path
    df_clustering['img_name'] = df_clustering['img_path'].apply(lambda x: os.path.basename(x)[:-4])
    
    # Merge dataframes
    df_merged = pd.merge(df_clustering, df, on=['dataset', 'img_name'], how='left')
    
    # Select only needed columns
    df_merged_filtered = df_merged[["dataset", "img_name", "centerX", "centerY", "labelTIL", "labelTumor", "labelCluster"]]
    
    return df_merged_filtered

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Read and preprocess data
    df = read_data(args.classification_task, args.testset)
    
    # Remove test dataset
    df = df[df.dataset != args.testset]
    df.dropna(inplace=True)
    
    # Determine stratification column based on task
    stratifier = "labelTumor" if args.classification_task == "tumor" else "labelTIL"
    
    datasets = []
    weights = []
    
    # Determine number of tasks
    if not args.multitask:
        num_tasks = 1
        
        # Create dataset for single task
        sample, _ = train_test_split(df, train_size=args.sample_size, stratify=df[stratifier], random_state=args.seed)
        img_saved_path = f'/home/michael/data/Augmented/{args.classification_task}_{args.testset}/single'
        dataset = RandStainNADataset(sample.reset_index(), task=args.classification_task, 
                                    testset=args.testset, saved_path=img_saved_path, crop_size=args.crop_size)
        datasets.append(dataset)
        w = torch.tensor([dataset.pos_weight], dtype=torch.float32, device=device)
        weights.append(w)
    else:
        # Create datasets for multi-task model based on clusters
        unique_clusters = df.labelCluster.unique()
        num_tasks = len(unique_clusters)
        img_saved_path = f'/home/michael/data/Augmented/{args.classification_task}_{args.testset}/multi'
        
        for i in range(num_tasks):
            df_filtered = df[df.labelCluster == i].reset_index()
            if len(df_filtered) == 0:
                continue
                
            sample, _ = train_test_split(df_filtered, train_size=args.sample_size, stratify=df_filtered[stratifier], random_state=args.seed)
            
            dataset = RandStainNADataset(sample.reset_index(), task=args.classification_task, 
                                        testset=args.testset, saved_path=img_saved_path, crop_size=args.crop_size)
            w = torch.tensor([dataset.pos_weight], dtype=torch.float32, device=device)
            
            datasets.append(dataset)
            weights.append(w)
        
        # Update num_tasks to actual number of datasets created
        num_tasks = len(datasets)

    # Select model architecture
    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks, retrain=True)
    elif args.model == "ResNet18":
        model = MultiTaskResNet18(num_tasks=num_tasks, retrain=True)
    elif args.model == "EfficientNet":
        model = MultiTaskEfficientNet(num_tasks=num_tasks)
    else:
        model = UNIMultitask(num_tasks=num_tasks)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Adjust batch size for multi-GPU
    batch_size = args.batch_size * (torch.cuda.device_count() if multi_gpu else 1)
    
    # Create data loaders
    dataloaders = {}
    for task_idx, dataset in enumerate(datasets):
        train_size = int(args.train_split * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        dataloaders[task_idx] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=4, pin_memory=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            drop_last=True, num_workers=4, pin_memory=True)
        }
    
    # Train the model
    train_model(model, dataloaders, optimizer, weights, args.epochs, device)
    
    print(f"Model saved as saved_models/{exp_name}_final.pth")
