from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from datasets import MultiTaskDataset
from models import MultiTaskResNet50,  MultiTaskResNet18, UNIMultitask, MultiTaskEfficientNet
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
import json

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
project_name = f"FULL-ColorBasedMultitask-Test-{args.crop_size}-{args.model}-no-aug"
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

def predict_cluster(df):
    X_train = df[["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]].values
    y_train = df.labelCluster

    # Feature scaling (not always necessary for Random Forest, but can be good practice)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=11)

    # Train the model
    clf.fit(X_train, y_train)
    return scaler, clf

# def calculate_optimal_threshold(labels, outputs):
#     fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), outputs.cpu().numpy())
#     optimal_idx = np.argmax(tpr - fpr)  # Youden's index
#     optimal_threshold = thresholds[optimal_idx]
#     return optimal_threshold

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

    # all_outputs = []
    # all_labels = []

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
                outputs[cluster_indices] = torch.sigmoid(cluster_outputs).squeeze()

 
            acc_metric((outputs > 0.5).int(), labels)
            uar_metric((outputs > 0.5).int(), labels)
            f1_metric((outputs > 0.5).int(), labels)
            roc_auc_metric(outputs, labels)  # ROC-AUC uses raw sigmoid values
            confusion_matrix((outputs > 0.5).int(), labels)

         
    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Accuracy_test': acc_metric.compute().item(),
        'UAR_test': uar_metric.compute().item(),
        'F1_test': f1_metric.compute().item(),
        'AUC_ROC_test': roc_auc_metric.compute().item(),
    }

    #write results into json file
    results = {
        "model": args.model,
        "task": args.classification_task,
        "testset": args.testset,
        "augmented": "No",
        "method": "cluster based",
        "num_tasks": num_tasks
    }
    results.update(metrics_dict)
    with open("outputs/test_result.json", "a") as f:
        f.write(json.dumps(results) + '\n')

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def test_by_mv(model, dataloader, device):
    model.eval()
    
    # Initialize metrics
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    roc_auc_metric = torchmetrics.AUROC(task="binary").to(device)
    
    # initialize a confusion matrix torchmetrics object
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():

        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            batch_predictions = model(images)
            binary_predictions = torch.tensor([[1.0 if torch.sigmoid(value) > 0.5 else 0.0 for value in sublist] for sublist in batch_predictions],  device = device)

            outputs, _ = torch.mode(binary_predictions, dim=0)

            # Accumulate metrics
            acc_metric(outputs, labels)
            uar_metric(outputs, labels)
            f1_metric(outputs, labels)
            roc_auc_metric(outputs, labels)
        
            # acculmate confusion matrix 
            confusion_matrix(outputs, labels)

         
    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Accuracy_test': acc_metric.compute().item(),
        'UAR_test': uar_metric.compute().item(),
        'F1_test': f1_metric.compute().item(),
        'AUC_ROC_test': roc_auc_metric.compute().item(),
    }

    #write results into json file
    results = {
        "model": args.model,
        "task": args.classification_task,
        "testset": args.testset,
        "augmented": "No",
        "method": "majority vote",
        "num_tasks": num_tasks
    }
    results.update(metrics_dict)
    with open("outputs/test_result.json", "a") as f:
<<<<<<< HEAD
        json.dump(results, f, indent=4)
=======
        f.write(json.dumps(results) + '\n')
>>>>>>> 25b4d6e04f80025271abdfca98ccba651c16a2cc

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm


if __name__ == '__main__':

    df = pd.read_csv("dataset/full_dataset.csv")
    df = df[df.dataset == args.testset].reset_index()

    # stratifier = "labelTumor" if args.classification_task == "tumor" else "labelTIL"
    # sample,_ = train_test_split(df, train_size=args.sample, stratify=df[stratifier], random_state=11)

    df_train = pd.read_csv(f"clustering/output/clustering_result_{args.classification_task}_{args.testset}.csv")


    dataset = MultiTaskDataset(df, args.classification_task, crop_size = args.crop_size)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 2, pin_memory = True)

    num_tasks = 1
    if args.multitask:
        num_tasks  = len(df_train.labelCluster.unique())  #number of clustes in dataset and number of heads in multitask model

    # load saved model
    model_files = glob.glob(f'saved_models/FULL-{args.crop_size}_{multitask}_no-aug_{args.model}_{args.classification_task}_{args.testset}*')
    state_dict = torch.load(model_files[0])

    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks)
    elif args.model == "ResNet18":
        model = MultiTaskResNet18(num_tasks=num_tasks, retrain = True)
    elif args.model == "EfficientNet":
        model = MultiTaskEfficientNet(num_tasks=num_tasks)
    else:
        model = UNIMultitask(num_tasks=num_tasks)

    model.load_state_dict(state_dict)
    model.to(device)
    
    class_names = ["non-tumor", "tumor"] if args.classification_task == "tumor" else ["non-TIL", "TIL"]

    if args.multitask:
        if args.test_method == "cluster":       #test by most similar cluster
            scaler, predictor = predict_cluster(df_train)

            # preprocess for test images
            df_statistics = pd.read_csv('clustering/output/img_statistics.csv')
            df_test = df_statistics[df_statistics.dataset == args.testset].reset_index()
            test_data = df_test[["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]]
            clusters = predictor.predict(scaler.transform(test_data))
            df_test['labelCluster'] = clusters

            test_metrics_dict, cm = test_by_cluster(model, dataloader, df_test, device)
        elif args.test_method == "mv":          # majority vote
            test_metrics_dict, cm = test_by_mv(model, dataloader, device)
    else:
        test_metrics_dict, cm = test_by_mv(model, dataloader, device)


    wandb.log({**test_metrics_dict})
        

    # Plot confusion matrix from results of last val epoch
    plot_confusion_matrix(cm, class_names)
    
    run.finish()
