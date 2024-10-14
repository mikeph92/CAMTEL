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
from models import MultiTaskEfficientNet, MultiTaskResNet50,  MultiTaskResNet18
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

parser = argparse.ArgumentParser(description='clustering')
parser.add_argument('--classification-task', type=str, default='tumor', help='classification task: tumor or TIL') 
parser.add_argument('--testset', type=str, default='ocelot', help='dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)') 
parser.add_argument('--multitask', type=bool, default=True, help="Enable use multitask model")
parser.add_argument('--test-method', type=str, default='cluster', help='') 
parser.add_argument('--sample', type=float, default='0.5')
parser.add_argument('--crop-size', type=int, default=64)
parser.add_argument('--model', type=str, default="ResNet18", help="backbone ResNet18 or ResNet50")


args = parser.parse_args()

# initiate wandb
project_name = f"ColorBasedMultitask-Test-{args.crop_size}-{args.model}-no-aug"
multitask = "Multitask" if args.multitask else "Single"
method = "" if not args.multitask else f'_{args.test_method}'
exp_name = f"{args.classification_task}_{args.testset}_{multitask}{method}"
run = wandb.init(project=project_name, name=exp_name)

# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
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

# def reverse_normalize(tensor):
#     mean = torch.tensor([0.485, 0.456, 0.406])
#     std = torch.tensor([0.229, 0.224, 0.225])
#     # If the tensor has been normalized, we reverse it
#     # tensor.mul_(std).add_(mean) would re-normalize, so we reverse it:
#     for t, m, s in zip(tensor, mean, std):
#         t.mul_(s).add_(m)
#     return tensor


# def preprocess_image(path):
#     img = Image.open(path).convert('LAB')
#     lab_img = np.array(img)
    
#     l, a, b = lab_img[:,:,0], lab_img[:,:,1], lab_img[:,:,2]
#     l_mean, l_std = np.mean(l), np.std(l)
#     a_mean, a_std = np.mean(a), np.std(a)
#     b_mean, b_std = np.mean(b), np.std(b)
#     image_data = [l_mean, l_std, a_mean, a_std, b_mean, b_std]

#     return image_data
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

def test_by_cluster(model, dataloader, cluster_predictor, df_test, device):
    '''
    Evaluate the model on the entire test set.
    '''
    model.eval()
    
    # Initialize metrics
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    
    # initialize a confusion matrix torchmetrics object
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():

        for images, labels, img_names in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = []
            for idx, img in enumerate(images):
                img_data = df_test[df_test.img_name == img_names[idx]][["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]]
                img_data = img_data.values.reshape(1,-1)
                
                scaler = cluster_predictor[0]
                predictor = cluster_predictor[1]

                cluster = predictor.predict(scaler.transform(img_data))[0]

                output = model(img.unsqueeze(0))[cluster]
                predict = 1.0 if output > 0.5 else 0.0
                outputs.append(predict) 

            outputs = torch.tensor(outputs, dtype=torch.float32).to(device)
            # Accumulate metrics
            acc_metric(outputs, labels)
            uar_metric(outputs, labels)
        
            # acculmate confusion matrix 
            confusion_matrix(outputs, labels)

         
    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Accuracy_test': acc_metric.compute(),
        'UAR_test': uar_metric.compute(),
    }

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def test_by_mv(model, dataloader, device):
    model.eval()
    
    # Initialize metrics
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    uar_metric = torchmetrics.classification.BinaryRecall().to(device)
    
    # initialize a confusion matrix torchmetrics object
    confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(device)

    with torch.no_grad():

        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = []
            for img in images:
                predictions = model(img.unsqueeze(0))
                binary_predictions = [1.0 if output > 0.5 else 0.0 for output in predictions]

                final_prediction = torch.mode(torch.tensor(binary_predictions)).values.item()
                outputs.append(final_prediction) 

            outputs = torch.tensor(outputs, dtype=torch.float32).to(device)
            # Accumulate metrics
            acc_metric(outputs, labels)
            uar_metric(outputs, labels)
        
            # acculmate confusion matrix 
            confusion_matrix(outputs, labels)

         
    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Accuracy_test': acc_metric.compute(),
        'UAR_test': uar_metric.compute(),
    }

    # Compute the confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm


if __name__ == '__main__':

    df = pd.read_csv("dataset/full_dataset.csv")
    df = df[df.dataset == args.testset].reset_index()

    # stratifier = "labelTumor" if args.classification_task == "tumor" else "labelTIL"
    # sample,_ = train_test_split(df, train_size=args.sample, stratify=df[stratifier], random_state=11)

    df_train = pd.read_csv(f"clustering/output/clustering_result_{args.classification_task}_{args.testset}.csv")
    cluster_predictor = predict_cluster(df_train)
    df_test = pd.read_csv('clustering/output/img_statistics.csv')

    dataset = MultiTaskDataset(df, args.classification_task, crop_size = args.crop_size)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    num_tasks = 1
    if args.multitask:
        num_tasks  = len(df_train.labelCluster.unique())  #number of clustes in dataset and number of heads in multitask model

    # load saved model
    model_files = glob.glob(f'saved_models/{args.crop_size}_{multitask}_no-aug_{args.model}_{args.classification_task}_{args.testset}*')
    state_dict = torch.load(model_files[0])

    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks)
    else:
        model = MultiTaskResNet18(num_tasks=num_tasks)

    model.load_state_dict(state_dict)
    model.to(device)
    
    class_names = ["non-tumor", "tumor"] if args.classification_task == "tumor" else ["non-TIL", "TIL"]

    if args.multitask:
        if args.test_method == "cluster":       #test by most similar cluster
            test_metrics_dict, cm = test_by_cluster(model, dataloader, cluster_predictor, df_test, device)
        elif args.test_method == "mv":          # majority vote
            test_metrics_dict, cm = test_by_mv(model, dataloader, device)
    else:
        test_metrics_dict, cm = test_by_mv(model, dataloader, device)


    wandb.log({**test_metrics_dict})
        

    # Plot confusion matrix from results of last val epoch
    plot_confusion_matrix(cm, class_names)
    
    run.finish()
