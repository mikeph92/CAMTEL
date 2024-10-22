from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from datasets import MultiTaskDatasetRandStainNA
from models import MultiTaskResNet50, MultiTaskResNet18
import torch.optim as optim
import argparse
import glob
import os
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import seaborn as sn
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sys

parser = argparse.ArgumentParser(description='clustering')
parser.add_argument('--classification-task', type=str, default='tumor', help='classification task: tumor or TIL') 
parser.add_argument('--testset', type=str, default='ocelot', help='dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)') 
parser.add_argument('--multitask', type=bool, default=True, help="Enable use multitask model")
parser.add_argument('--test-method', type=str, default='cluster', help='') 
parser.add_argument('--sample', type=float, default='0.9')
parser.add_argument('--crop-size', type=int, default=48)
parser.add_argument('--model', type=str, default="ResNet18", help="backbone ResNet18 or ResNet50")

args = parser.parse_args()

# initiate wandb
project_name = f"FULL-ColorBasedMultitask-Test-{args.crop_size}-{args.model}-RandStainNA"
multitask = "Multitask" if args.multitask else "Single"
method = "" if not args.multitask else f'_{args.test_method}'
exp_name = f"{args.classification_task}_{args.testset}_{multitask}{method}"
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
    # ax = sn.heatmap(df_cm, annot=True, cmap='flare', fmt='.2f')

    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('True')

    # plt.savefig(f'plots/{exp_name}.png')

#     return image_data
def predict_cluster(df):
    X_train = df[["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]]
    y_train = df.labelCluster

    # Feature scaling (not always necessary for Random Forest, but can be good practice)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)
    return scaler, clf

def calculate_optimal_threshold(labels, outputs):
    fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), outputs.cpu().numpy())
    optimal_idx = np.argmax(tpr - fpr)  # Youden's index
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def test_by_cluster(model, dataloader, df_test, device):
    '''
    Evaluate the model on the entire test set.
    '''
    model.eval()
    
    # Initialize metrics
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    roc_auc_metric = torchmetrics.AUROC(task="binary").to(device)
    
    # initialize a confusion matrix torchmetrics object
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    all_outputs = []
    all_labels = []

    with torch.no_grad():

        for images, labels, img_names in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            clusters = np.array([df_test.loc[df_test.img_name == img_name, 'labelCluster'].iloc[0] for img_name in img_names])


            outputs = torch.zeros(len(images), dtype=torch.float32).to(device)
            for cluster in np.unique(clusters):
                # Select indices for each cluster
                cluster_indices = np.where(clusters == cluster)[0]
                
                # Perform batch prediction for images in this cluster
                cluster_images = images[cluster_indices]
                cluster_outputs = model(cluster_images)[cluster]
                outputs[cluster_indices] = cluster_outputs.squeeze()

            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_outputs = torch.tensor(np.concatenate(all_outputs))
        all_labels = torch.tensor(np.concatenate(all_labels))

        optimal_threshold = calculate_optimal_threshold(all_labels, all_outputs)

        if outputs.device != device:
            outputs = outputs.to(device)
        if final_labels.device != device:
            final_labels = final_labels.to(device)
        optimal_threshold = torch.tensor(optimal_threshold, device=device)
        
        acc_metric((all_outputs > optimal_threshold).int(), all_labels)
        uar_metric((all_outputs > optimal_threshold).int(), all_labels)
        f1_metric((all_outputs > optimal_threshold).int(), all_labels)
        roc_auc_metric(all_outputs, all_labels)  # ROC-AUC uses raw sigmoid values
        confusion_matrix((all_outputs > optimal_threshold).int(), all_labels)

         
    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Accuracy_test': acc_metric.compute(),
        'UAR_test': uar_metric.compute(),
        'F1_test': f1_metric.compute(),
        'AUC_ROC_test': roc_auc_metric.compute(),
    }

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def predict_with_model_and_labels(model, dataloader, task_index, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No need to compute gradients for inference
        for images, labels, _ in dataloader:  # Get both images and labels
            images = images.to(device)  # Move images to the GPU
            labels = labels.to(device)  # Move labels to the GPU for accuracy calculation
            
            outputs = model(images)[task_index]  # Assuming outputs are probabilities (sigmoid or softmax)
            
            # Apply threshold of 0.5 to convert probabilities to class predictions
            preds = (outputs > 0.5).int().squeeze()  # Binary: 1 if prob > 0.5 else 0
            
            all_preds.append(preds.cpu())  # Move predictions back to the CPU and store
            all_labels.append(labels.cpu())  # Move labels back to the CPU and store

    # Concatenate all predictions and labels into a single tensor (m,)
    return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)


def test_by_mv(model, dataloaders, device):
    model.eval()
    
    # Initialize metrics
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    roc_auc_metric = torchmetrics.AUROC(task="binary").to(device)
    
    # initialize a confusion matrix torchmetrics object
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    # List to store predictions from all n dataloaders
    all_predictions = []
    final_labels = None  # To store the true labels once

    # Iterate over all n dataloaders and get the predictions and labels
    for i, dataloader in enumerate(dataloaders):
        preds, labels = predict_with_model_and_labels(model, dataloader, i, device)
        
        all_predictions.append(preds)
        
        if final_labels is None:
            final_labels = labels  # Save the labels only once (labels are the same for all dataloaders)

    # Stack the predictions from all dataloaders into a tensor of shape (n, m)
    stacked_predictions = torch.stack(all_predictions, dim=0)  # Shape: (n, m)

    # Use torch.mode to find the most frequent prediction for each row (axis=0 for mode across loaders)
    outputs, _ = torch.mode(stacked_predictions, dim=0)

    if outputs.device != device:
        outputs = outputs.to(device)
    if final_labels.device != device:
        final_labels = final_labels.to(device)
        
    # Accumulate metrics
    acc_metric(outputs, final_labels)
    uar_metric(outputs, final_labels)
    f1_metric(outputs, final_labels)
    roc_auc_metric(outputs, final_labels)
    confusion_matrix(outputs, final_labels)

         
    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Accuracy_test': acc_metric.compute(),
        'UAR_test': uar_metric.compute(),
        'F1_test': f1_metric.compute(),
        'AUC_ROC_test': roc_auc_metric.compute(),
    }

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def dataset_by_cluster(df, df_train, num_tasks):
    scaler, predictor = predict_cluster(df_train)

    # preprocess for test images
    df_statistics = pd.read_csv('clustering/output/img_statistics.csv')
    df_test = df_statistics[df_statistics.dataset == args.testset].reset_index()
    test_data = df_test[["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]]
    clusters = predictor.predict(scaler.transform(test_data))
    df_test['labelCluster'] = clusters

    df_merged = pd.merge(df, df_test, left_on=['dataset', 'img_name'], right_on=['dataset', 'img_name'], how='left')
    df_merged_filtered = df_merged[["dataset", "img_name", "centerX", "centerY", "labelTIL", "labelTumor","labelCluster"]]

   
    datasets = []

    for i in range(num_tasks):
        df_filtered = df_merged_filtered[df_merged_filtered.labelCluster == i]
        cluster = i
        try:
            # sample,_ = train_test_split(df_filtered, train_size=args.sample, stratify=df_filtered[stratifier], random_state=7)
            dataset = MultiTaskDatasetRandStainNA(df_filtered.reset_index(), task = args.classification_task, 
                                              testset = args.testset, cluster = cluster, crop_size = args.crop_size)
            datasets.append(dataset) 
        except Exception:
            continue

    
    dataset = ConcatDataset(datasets)
    return dataset, df_test

def dataset_by_mv(df, num_tasks):
    datasets = []

    for i in range(num_tasks):
        cluster = i if args.multitask else None
        try:
            sample,_ = train_test_split(df, train_size=args.sample, stratify=df[stratifier], random_state=7)
            dataset = MultiTaskDatasetRandStainNA(sample.reset_index(), task = args.classification_task, 
                                              testset = args.testset, cluster = cluster, crop_size = args.crop_size)
            datasets.append(dataset) 
        except Exception:
            continue
    return datasets

if __name__ == '__main__':

    df = pd.read_csv("dataset/full_dataset.csv")
    df = df[df.dataset == args.testset].reset_index()

    stratifier = "labelTumor" if args.classification_task == "tumor" else "labelTIL"
    df_train = pd.read_csv(f"clustering/output/clustering_result_{args.classification_task}_{args.testset}.csv")

    
    num_tasks = 1

    if args.multitask:
        num_tasks  = len(df_train.labelCluster.unique())  #number of clustes in dataset and number of heads in multitask model

    batch_size = 96

    # load saved model
    model_files = glob.glob(f'saved_models/FULL-randstainna_{args.crop_size}_{multitask}_{args.model}_{args.classification_task}_{args.testset}*')
    state_dict = torch.load(model_files[0])

    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks)
    else:
        model = MultiTaskResNet18(num_tasks=num_tasks)

    model.load_state_dict(state_dict)
    model.to(device)
    

    if args.multitask:
        if args.test_method == "cluster":       #test by most similar cluster
            dataset, df_test = dataset_by_cluster(df, df_train, num_tasks)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 4, pin_memory = True)
            test_metrics_dict, cm = test_by_cluster(model, dataloader, df_test, device)
        elif args.test_method == "mv":          # majority vote
            datasets = dataset_by_mv(df, num_tasks)
            dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 4, pin_memory = True) for dataset in datasets]
            test_metrics_dict, cm = test_by_mv(model, dataloaders, device)
    else:
        datasets = dataset_by_mv(df, num_tasks)
        dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 4, pin_memory = True) for dataset in datasets]
        test_metrics_dict, cm = test_by_mv(model, dataloaders, device)


    wandb.log({**test_metrics_dict})
        

    # Plot confusion matrix from results of last val epoch
    class_names = ["non-tumor", "tumor"] if args.classification_task == "tumor" else ["non-TIL", "TIL"]
    plot_confusion_matrix(cm, class_names)
    
    run.finish()
