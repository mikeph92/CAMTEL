from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from datasets import RandStainNADataset
from models import MultiTaskResNet18, MultiTaskResNet50, UNIMultitask
import torch.optim as optim
import argparse
import json
import os
import pandas as pd
from skimage.color import rgb2hed, hed2rgb
from PIL import Image


parser = argparse.ArgumentParser(description='clustering')
parser.add_argument('--classification-task', type=str, default='tumor', help='classification task: tumor or TIL') 
parser.add_argument('--testset', type=str, default='ocelot', help='dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)') 
parser.add_argument('--crop-size', type=int, default=48)
parser.add_argument('--model', type=str, default="ResNet18", help="backbone ResNet18, ResNet50, UNI")


args = parser.parse_args()

# initiate wandb
project_name = "Test-base-normalization"
exp_name = f"BASE-NORMALIZATION-{args.crop_size}_no-aug_{args.model}_{args.classification_task}_{args.testset}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run = wandb.init(project=project_name, name=exp_name)

# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    index = 1 if args.model == "ResNet18" else 0
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
    


def train_model(model, dataloaders, optimizer, weights, n_epochs, device):   
    model.to(device)
    # Train by iterating over epochs
    for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
        train_metrics_dict = train_epoch(model, optimizer, weights, dataloaders, num_tasks, device)
                
        val_metrics_dict, _ = val_epoch(model, weights, dataloaders, num_tasks, device)
        wandb.log({**train_metrics_dict, **val_metrics_dict})
        

    # Plot confusion matrix from results of last val epoch
    # plot_confusion_matrix(cm, class_names)

def test_by_mv(model, dataloader, device):    
    # Initialize metrics
    acc_metric = torchmetrics.classification.BinaryAccuracy()
    uar_metric = torchmetrics.classification.BinaryRecall()
    f1_metric = torchmetrics.F1Score(task="binary")
    roc_auc_metric = torchmetrics.AUROC(task="binary")
    
    # initialize a confusion matrix torchmetrics object
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix()

    with torch.no_grad():

        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = torch.sigmoid(model(images).squeeze())
            preds = [1.0 if torch.sigmoid(value) > 0.5 else 0.0 for value in outputs]

            # Accumulate metrics
            acc_metric(preds, labels)
            uar_metric(preds, labels)
            f1_metric(preds, labels)
            roc_auc_metric(preds, labels)
        
            # acculmate confusion matrix 
            confusion_matrix(preds, labels)

         
    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Accuracy_test': acc_metric.compute().item(),
        'UAR_test': uar_metric.compute().item(),
        'F1_test': f1_metric.compute().item(),
        'AUC_ROC_test': roc_auc_metric.compute().item(),
    }

    #write results into json file
    results = {
        "task": args.classification_task,
        "testset": args.testset,
        "augmented": "No",
        "method": "base normalization",
        "num_tasks": num_tasks
    }
    results.update(metrics_dict)
    with open("outputs/test_result.json", "a") as f:
        json.dump(results, f, indent=4)

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def get_path(dataset, img_name):
    path_dict = {
        "ocelot": "/home/michael/data/ProcessedHistology/Ocelot/inputs/original",
        "lizard": "/home/michael/data/ProcessedHistology/Lizard/inputs/original",
        "pannuke": "/home/michael/data/ProcessedHistology/PanNuke/inputs/original",
        "nucls": "/home/michael/data/ProcessedHistology/NuCLS/inputs/original",
        "cptacCoad": "/home/michael/data/ProcessedHistology/CPTAC-COAD/inputs/original",
        "tcgaBrca": "/home/michael/data/ProcessedHistology/TCGA-BRCA/inputs/original"
    }

    return f"{path_dict[dataset]}/{img_name}.tif"

def compute_training_profile(train_paths):
    """
    Compute mean and standard deviation profiles from the training dataset.
    """
    means = []
    stds = []

    for path in train_paths:
        img = np.array(Image.open(path).convert("RGB"))
        hed = rgb2hed(img)
        for i in range(3):  # For H, E, D channels
            means.append(hed[..., i].mean())
            stds.append(hed[..., i].std())

    return np.mean(means), np.mean(stds)

def macenko_normalize(img, mean, std):
    """
    Apply Macenko normalization using the computed mean and std.
    """
    hed = rgb2hed(img)
    for i in range(3):
        hed[..., i] = (hed[..., i] - hed[..., i].mean()) / hed[..., i].std()  # Standardize
        hed[..., i] = hed[..., i] * std + mean  # Normalize
    return np.clip(hed2rgb(hed), 0, 1) * 255

if __name__ == '__main__':

    df = pd.read_csv(f"clustering/output/clustering_result_{args.classification_task}_{args.testset}.csv")
    df['img_name'] = df.apply(lambda row: os.path.basename(df['img_path'])[:-4], axis = 1)
    df.dropna(inplace=True)


    save_dir = f'/home/michael/data/Normalization/{args.classification_task}_{args.testset}'
    os.makedirs(save_dir, exist_ok=True)
    
    #normalize training images
    img_paths = df['img_path'].unique()
    mean, std = compute_training_profile(img_paths)
    for img_path in img_paths:
        img = np.array(Image.open(img_path).convert("RGB"))
        img = macenko_normalize(img, mean, std).astype(np.uint8)
        img = Image.fromarray(img)
        img_name = os.path.basename(img_path)
        img.save(f'{save_dir}/{img_name}')

    datasets = []
    weights = []

    num_tasks = 1                # use with single head model

    for i in range(num_tasks):
                
        dataset = RandStainNADataset(df, task = args.classification_task, 
                                            testset = args.testset, saved_path = save_dir, crop_size = args.crop_size)
        w = torch.tensor([dataset.pos_weight], dtype=torch.float32, device=device)

        datasets.append(dataset)
        weights.append(w)


    batch_size = 64
    n_epochs = 10
    class_names = ["non-tumor", "tumor"] if args.classification_task == "tumor" else ["non-TIL", "TIL"]
    train_split = 0.8
    
    # Instantiate the model
    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks, retrain = True)
    elif args.model == "ResNet18":
        model = MultiTaskResNet18(num_tasks=num_tasks, retrain = True)
    else:
        model = UNIMultitask(num_tasks=num_tasks)
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)  

    dataloaders = {}
    for task_idx, dataset in enumerate(datasets):
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(i)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = generator)        

        dataloaders[task_idx] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 2, pin_memory = True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 2, pin_memory = True)
        }
    
    train_model(model, dataloaders, optimizer, weights, n_epochs, device)
    
    # Saving the model state dictionary locally
    model_path = f'saved_models/{i}_{exp_name}.pth'
    torch.save(model.state_dict(), model_path)
    model.eval()
    model.to(device)
    
    # test time
    df_test = pd.read_csv("dataset/full_dataset.csv")
    df_test = df_test[df_test.dataset == args.testset].reset_index()

    #normalize test images
    img_paths = df_test.apply(lambda row: get_path(row['dataset'],row['img_name']), axis = 1)
    for img_path in img_paths.unique():
        img = np.array(Image.open(img_path).convert("RGB"))
        img = macenko_normalize(img, mean, std).astype(np.uint8)
        img = Image.fromarray(img)
        img_name = os.path.basename(img_path)
        img.save(f'{save_dir}/{img_name}')

    dataset_test = RandStainNADataset(df_test, task = args.classification_task, 
                                            testset = args.testset, saved_path = save_dir, crop_size = args.crop_size)
    batch_size = 64
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 2, pin_memory = True)

    test_metrics_dict, cm = test_by_mv(model, dataloader_test, device)
    wandb.log({**test_metrics_dict})
        
    # Plot confusion matrix from results of last val epoch
    plot_confusion_matrix(cm, class_names)

    run.finish()
