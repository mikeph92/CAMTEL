from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
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

parser = argparse.ArgumentParser(description='clustering')
parser.add_argument('--classification-task', type=str, default='tumor', help='classification task: tumor or TIL') 
parser.add_argument('--testset', type=str, default='ocelot', help='dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)') 
parser.add_argument('--sample-size', type=float, default='0.9')
parser.add_argument('--retrain', type=bool, default=True, help="Enable retrain base model")
parser.add_argument('--multitask', type=bool, default=True, help="Enable use multitask model")
parser.add_argument('--crop-size', type=int, default=48)
parser.add_argument('--model', type=str, default="ResNet18", help="backbone ResNet18 or ResNet50")


args = parser.parse_args()

# initiate wandb
project_name = "ColorBasedMultitask-full"
multitask = "Multitask" if args.multitask else "Single"
retrain =  "Retrained" if args.retrain else "Pretrained"
exp_name = f"FULL-{args.crop_size}_{multitask}_no-aug_{args.model}_{args.classification_task}_{args.testset}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run = wandb.init(project=project_name, name=exp_name)

# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    index = 1 #if args.model == "ResNet18" else 0
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)

def plot_confusion_matrix(cm, class_names):
    '''
        cm: the confusion matrix that we wish to plot
        class_names: the names of the classes 
    '''

    # this normalizes the confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
    
    df_cm = pd.DataFrame(cm_normalized, class_names, class_names)
    print(df_cm)
    

def train_epoch(model, optimizer, weights, dataloaders,  num_tasks, device):

    model.train()

    # Initialize metrics
    epoch_loss = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
        
    for task_idx in range(num_tasks):
        criterion = nn.BCEWithLogitsLoss(pos_weight = weights[task_idx])

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

    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Loss_train': epoch_loss.compute(),
        'Accuracy_train': acc_metric.compute(),
        'UAR_train': uar_metric.compute(),
    }

    return metrics_dict

def val_epoch(model, weights, dataloaders, num_tasks, device):
    '''
    Evaluate the model on the entire validation set.
    '''
    model.eval()
    
    # Initialize metrics
    epoch_loss = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    

    
    # initialize a confusion matrix torchmetrics object
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():
        for task_idx in range(num_tasks):
            criterion = nn.BCEWithLogitsLoss(pos_weight = weights[task_idx])

            for images, labels, _ in dataloaders[task_idx]['val']:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)[task_idx].squeeze()
                loss = criterion(outputs, labels)

                # Accumulate metrics
                epoch_loss(loss)
                acc_metric(outputs, labels)
                uar_metric(outputs, labels)
            
                # acculmate confusion matrix 
                confusion_matrix(outputs, labels)

         
    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Loss_val': epoch_loss.compute(),
        'Accuracy_val': acc_metric.compute(),
        'UAR_val': uar_metric.compute(),
    }

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm
    


def train_model(model, dataloaders, optimizer, weights, n_epochs, device, class_names):   
    model.to(device)
    # Train by iterating over epochs
    for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
        train_metrics_dict = train_epoch(model, optimizer, weights, dataloaders, num_tasks, device)
                
        val_metrics_dict, cm = val_epoch(model, weights, dataloaders, num_tasks, device)
        wandb.log({**train_metrics_dict, **val_metrics_dict})
        

    # Plot confusion matrix from results of last val epoch
    # plot_confusion_matrix(cm, class_names)


def read_data(task, testset):
    df = pd.read_csv("dataset/full_dataset.csv")

    df_clustering = pd.read_csv(f"clustering/output/clustering_result_{task}_{testset}.csv")

    df_clustering['img_name'] = df_clustering.apply(lambda row: os.path.basename(row['img_path'])[:-4], axis = 1)

    df_merged = pd.merge(df_clustering, df, left_on=['dataset', 'img_name'], right_on=['dataset', 'img_name'], how='left')
    df_merged_filtered = df_merged[["dataset", "img_name", "centerX", "centerY", "labelTIL", "labelTumor","labelCluster"]]

    return df_merged_filtered


if __name__ == '__main__':

    df = read_data(args.classification_task, args.testset)

    df = df[df.dataset != args.testset]
    df.dropna(inplace=True)

    # # Create a new column combining multiple stratification features
    # if args.classification_task == 'tumor':
    #     df['stratify_key'] = df.apply(lambda row: f"{row['labelTumor']}_{row['dataset']}", axis = 1)
    # else:
    #     df['stratify_key'] = df.apply(lambda row: f"{row['labelTIL']}_{row['dataset']}", axis = 1)

    datasets = []
    weights = []

    num_tasks = 1                # use with single head model
    df_filtered = df.copy().reset_index()

    if args.multitask:    # else use multihead model
        num_tasks  = len(df.labelCluster.unique())  #number of clustes in dataset and number of heads in multitask model


    for i in range(num_tasks):
        if args.multitask:
            df_filtered = df[df.labelCluster == i].reset_index()
        
        # sample,_ = train_test_split(df_filtered, train_size=args.sample_size, stratify=df_filtered['stratify_key'], random_state=11)
        
        dataset = MultiTaskDataset(df_filtered, args.classification_task, crop_size = args.crop_size)
        w = torch.tensor([dataset.pos_weight], dtype=torch.float32, device=device)

        datasets.append(dataset)
        weights.append(w)


    # Instantiate the model
    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks, retrain = True)
    elif args.model == "ResNet18":
        model = MultiTaskResNet18(num_tasks=num_tasks, retrain = True)
    elif args.model == "EfficientNet":
        model = MultiTaskEfficientNet(num_tasks=num_tasks)
    else:
        model = UNIMultitask(num_tasks=num_tasks)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 64
    n_epochs = 10
    class_names = ["non-tumor", "tumor"] if args.classification_task == "tumor" else ["non-TIL", "TIL"]
    train_split = 0.8
    
    
    dataloaders = {}
    for task_idx, dataset in enumerate(datasets):
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = generator)        

        dataloaders[task_idx] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 2, pin_memory = True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 2, pin_memory = True)
        }
    
    train_model(model, dataloaders, optimizer, weights, n_epochs, device, class_names)
    
    # Saving the model state dictionary locally
    torch.save(model.state_dict(), f'saved_models/{exp_name}.pth')
    run.finish()
