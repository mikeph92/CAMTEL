from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import os
import json
import glob
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pandas as pd
from datasets import RandStainNADataset, get_path
from models import MultiTaskResNet50, MultiTaskResNet18, UNIMultitask, MultiTaskEfficientNet
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryConfusionMatrix
from torchmetrics import F1Score, AUROC
from randstainna import RandStainNA

parser = argparse.ArgumentParser(description='Inference for Histopathology Image Classification')
parser.add_argument('--classification-task', type=str, default='tumor', help='classification task: tumor or TIL') 
parser.add_argument('--testset', type=str, default='ocelot', help='dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)') 
parser.add_argument('--multitask', type=bool, default=True, help="Enable use multitask model")
parser.add_argument('--test-method', type=str, default='cluster', help='Testing method: cluster or mv (majority vote)') 
parser.add_argument('--sample', type=float, default=0.5, help='Sample size for testing')
parser.add_argument('--crop-size', type=int, default=96, help='Size of image crops')
parser.add_argument('--model', type=str, default="ResNet18", help="backbone ResNet18, ResNet50, or EfficientNet")
parser.add_argument('--batch-size', type=int, default=32, help="Batch size per GPU")
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
parser.add_argument('--output-dir', type=str, default="outputs", help="Directory to save results")

args = parser.parse_args()

# Create timestamp for result naming
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
multitask_str = "Multitask" if args.multitask else "Single"
method_str = "" if not args.multitask else f'_{args.test_method}'
exp_name = f"pretrained_{args.classification_task}_{args.testset}_{multitask_str}{method_str}_{timestamp}"

# Setup for multi-GPU testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.device_count() > 1
if multi_gpu:
    print(f"Using {torch.cuda.device_count()} GPUs for testing")

# Ensure output directories exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs(f'/home/michael/data/Augmented/{args.classification_task}_{args.testset}', exist_ok=True)

def print_confusion_matrix(cm, class_names):
    '''
    Print the confusion matrix as a normalized percentage
    '''
    # Normalize the confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
    
    df_cm = pd.DataFrame(cm_normalized, class_names, class_names)
    print("Normalized Confusion Matrix:")
    print(df_cm)
    
    # Also print raw counts
    print("\nRaw Confusion Matrix:")
    df_raw = pd.DataFrame(cm, class_names, class_names)
    print(df_raw)

def predict_cluster(df):
    """
    Train a Random Forest classifier to predict cluster assignments
    based on color statistics
    """
    X_train = df[["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]]
    y_train = df.labelCluster

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Initialize and train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    clf.fit(X_train, y_train)
    
    return scaler, clf

def test_by_cluster(model, dataloader, df_test, device):
    '''
    Evaluate the model on the test set using cluster-based approach
    '''
    model.eval()
    
    # Initialize metrics
    acc_metric = BinaryAccuracy().to(device)
    uar_metric = BinaryRecall().to(device)
    f1_metric = F1Score(task="binary").to(device)
    roc_auc_metric = AUROC(task="binary").to(device)
    confusion_matrix = BinaryConfusionMatrix().to(device)

    print("Starting cluster-based testing...")
    with torch.no_grad():
        for images, labels, img_names in tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device).float()

            # Get cluster assignments for each image
            clusters = np.array([df_test.loc[df_test.img_name == img_name, 'labelCluster'].iloc[0] 
                                for img_name in img_names])

            # Initialize outputs
            outputs = torch.zeros(len(images), dtype=torch.float32).to(device)
            
            # Process images by cluster
            for cluster in np.unique(clusters):
                # Select indices for each cluster
                cluster_indices = np.where(clusters == cluster)[0]
                if len(cluster_indices) == 0:
                    continue
                    
                # Perform batch prediction for images in this cluster
                cluster_images = images[cluster_indices]
                cluster_outputs = model(cluster_images)[cluster]
                outputs[cluster_indices] = torch.sigmoid(cluster_outputs).squeeze()

            # Update metrics
            acc_metric((outputs > 0.5).int(), labels)
            uar_metric((outputs > 0.5).int(), labels)
            f1_metric((outputs > 0.5).int(), labels)
            roc_auc_metric(outputs, labels)
            confusion_matrix((outputs > 0.5).int(), labels)

    # Calculate metrics
    metrics_dict = {
        'Accuracy': acc_metric.compute().item(),
        'UAR': uar_metric.compute().item(),
        'F1': f1_metric.compute().item(),
        'AUC_ROC': roc_auc_metric.compute().item(),
    }

    # Save results to JSON
    results = {
        "model": args.model,
        "task": args.classification_task,
        "testset": args.testset,
        "augmented": "Yes",
        "method": "cluster based",
        "num_tasks": num_tasks,
        "timestamp": timestamp
    }
    results.update(metrics_dict)
    
    with open(f"{args.output_dir}/test_result_{exp_name}.json", "w") as f:
        json.dump(results, f, indent=4)

    # Log results to console
    print(f"Test Results:")
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def predict_with_model_and_labels(model, dataloader, task_index, device):
    """
    Get predictions from a specific task head
    """
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc=f"Predicting task {task_index}"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)[task_index]
            # Add unsqueeze to ensure proper dimensions if batch size is 1
            # Only squeeze the second dimension and ensure we keep at least 1 dimension
            preds = torch.sigmoid(outputs).round()
            if preds.dim() > 1:
                preds = preds.squeeze(1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Check if all tensors are at least 1D before concatenating
    all_preds = [p if p.dim() > 0 else p.unsqueeze(0) for p in all_preds]
    all_labels = [l if l.dim() > 0 else l.unsqueeze(0) for l in all_labels]
    
    # Concatenate all predictions and labels
    return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)

def test_by_mv(model, dataloaders, device):
    """
    Test using majority voting from all task heads
    """
    model.eval()
    
    # Initialize metrics
    acc_metric = BinaryAccuracy().to(device)
    uar_metric = BinaryRecall().to(device)
    f1_metric = F1Score(task="binary").to(device)
    roc_auc_metric = AUROC(task="binary").to(device)
    confusion_matrix = BinaryConfusionMatrix().to(device)

    print("Starting majority voting-based testing...")
    
    # Get predictions from all task heads
    all_predictions = []
    final_labels = None
    
    for i, dataloader in enumerate(dataloaders):
        print(f"Processing task head {i+1}/{len(dataloaders)}")
        preds, labels = predict_with_model_and_labels(model, dataloader, i, device)
        all_predictions.append(preds)
        
        if final_labels is None:
            final_labels = labels

    # Stack predictions for majority voting
    stacked_predictions = torch.stack(all_predictions, dim=0)
    
    # Use majority voting to determine final prediction
    outputs, _ = torch.mode(stacked_predictions, dim=0)

    # Move tensors to correct device
    outputs = outputs.to(device)
    final_labels = final_labels.to(device)
    
    # Update metrics
    acc_metric(outputs, final_labels)
    uar_metric(outputs, final_labels)
    f1_metric(outputs, final_labels)
    roc_auc_metric(outputs, final_labels)
    confusion_matrix(outputs, final_labels)

    # Calculate metrics
    metrics_dict = {
        'Accuracy': acc_metric.compute().item(),
        'UAR': uar_metric.compute().item(),
        'F1': f1_metric.compute().item(),
        'AUC_ROC': roc_auc_metric.compute().item(),
    }

    # Save results to JSON
    results = {
        "model": args.model,
        "task": args.classification_task,
        "testset": args.testset,
        "augmented": "Yes",
        "method": "majority vote",
        "num_tasks": num_tasks,
        "timestamp": timestamp
    }
    results.update(metrics_dict)
    
    with open(f"{args.output_dir}/test_result_{exp_name}.json", "w") as f:
        json.dump(results, f, indent=4)

    # Log results to console
    print(f"Test Results:")
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix.compute().cpu().numpy()

    return metrics_dict, cm

def augmenting_images(df, saved_path, cluster=None):
    """
    Apply RandStainNA augmentation to images
    """
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    
    # Choose appropriate YAML configuration
    if cluster is not None:
        yaml_path = f"/home/michael/CAMTEL/yaml_config/{args.classification_task}_{args.testset}_{cluster}.yaml"
    else:
        yaml_path = f"/home/michael/CAMTEL/yaml_config/{args.classification_task}_{args.testset}.yaml"
    
    # Create RandStainNA augmenter
    try:
        randstainna = RandStainNA(
            yaml_file=yaml_path,
            std_hyper=0.0,
            distribution='normal', 
            probability=1.0,
            is_training=False,
        )
    except Exception as e:
        print(f"Error loading RandStainNA configuration: {e}")
        return

    # Process each image
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Augmenting images {'for cluster '+str(cluster) if cluster is not None else ''}"):
        output_path = f"{saved_path}/{row['img_name']}.tif"
        
        # Skip if file already exists
        if os.path.isfile(output_path):
            continue
            
        try:
            img_path = get_path(row['dataset'], row['img_name'], "original")
            img = Image.open(img_path).convert('RGB')
            new_img = randstainna(img)
            new_img.save(output_path)
        except Exception as e:
            print(f"Error processing image {row['img_name']}: {e}")

def dataset_by_cluster(df, df_train, num_tasks):
    """
    Create dataset using cluster-based approach
    """
    print("Creating dataset using cluster-based approach...")
    
    # Train cluster predictor
    scaler, predictor = predict_cluster(df_train)

    # Load test image statistics
    df_statistics = pd.read_csv('clustering/output/img_statistics.csv')
    df_test = df_statistics[df_statistics.dataset == args.testset].reset_index()
    
    # Predict clusters for test images
    test_data = df_test[["l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"]]
    clusters = predictor.predict(scaler.transform(test_data))
    df_test['labelCluster'] = clusters

    # Create augmented images
    saved_path = f'/home/michael/data/Augmented/{args.classification_task}_{args.testset}/test_multi'
    
    # Augment images for each cluster
    for i in range(num_tasks):
        df_filtered = df_test[df_test.labelCluster == i].reset_index()
        if len(df_filtered) > 0:
            augmenting_images(df_filtered, saved_path, cluster=i)

    # Create dataset
    dataset = RandStainNADataset(df, task=args.classification_task, 
                                testset=args.testset, saved_path=saved_path, 
                                crop_size=args.crop_size)
    
    return dataset, df_test

def dataset_by_mv(df, num_tasks):
    """
    Create datasets for majority voting approach
    """
    print("Creating datasets for majority voting approach...")
    
    datasets = []
    stratifier = "labelTumor" if args.classification_task == "tumor" else "labelTIL"
    
    if num_tasks == 1:
        # Single task case
        saved_path = f'/home/michael/data/Augmented/{args.classification_task}_{args.testset}/test_single'
        augmenting_images(df, saved_path)

        try:
            sample, _ = train_test_split(df, train_size=args.sample, stratify=df[stratifier], random_state=args.seed)
            dataset = RandStainNADataset(sample.reset_index(), task=args.classification_task, 
                                        testset=args.testset, saved_path=saved_path, 
                                        crop_size=args.crop_size)
            datasets.append(dataset)
        except Exception as e:
            print(f"Error creating dataset: {e}")
    else:
        # Multi-task case
        for i in range(num_tasks):
            saved_path = f'/home/michael/data/Augmented/{args.classification_task}_{args.testset}/test_multi_{i}'
            augmenting_images(df, saved_path, cluster=i)
            
            try:
                sample, _ = train_test_split(df, train_size=args.sample, stratify=df[stratifier], random_state=args.seed)
                dataset = RandStainNADataset(sample.reset_index(), task=args.classification_task, 
                                           testset=args.testset, saved_path=saved_path, 
                                           crop_size=args.crop_size)
                datasets.append(dataset)
            except Exception as e:
                print(f"Error creating dataset for cluster {i}: {e}")
                continue
    
    return datasets

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("dataset/full_dataset.csv")
    df = df[df.dataset == args.testset].reset_index()

    stratifier = "labelTumor" if args.classification_task == "tumor" else "labelTIL"
    df_train = pd.read_csv(f"clustering/output/clustering_result_{args.classification_task}_{args.testset}.csv")
    
    # Determine number of tasks
    num_tasks = 1
    if args.multitask:
        num_tasks = len(df_train.labelCluster.unique())
        print(f"Using {num_tasks} tasks (clusters)")

    # Adjust batch size for multi-GPU
    batch_size = args.batch_size * (torch.cuda.device_count() if multi_gpu else 1)

    # Find and load saved model
    print("Loading model...")
    model_pattern = f'saved_models/pretrained_{multitask_str}_{args.model}_{args.classification_task}_{args.testset}_????????_??????_best.pth'
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No model found matching pattern: {model_pattern}")
    
    # Use the first matching model file
    model_path = model_files[0]
    print(f"Using model: {model_path}")
    state_dict = torch.load(model_path)

    # Initialize model
    if args.model == "ResNet50":
        model = MultiTaskResNet50(num_tasks=num_tasks)
    elif args.model == "ResNet18":
        model = MultiTaskResNet18(num_tasks=num_tasks, retrain=True)
    elif args.model == "EfficientNet":
        model = MultiTaskEfficientNet(num_tasks=num_tasks)
    else:
        model = UNIMultitask(num_tasks=num_tasks)

    # Load model weights
    try:
        # Move model to device and wrap with DataParallel if multiple GPUs available
        model.to(device)
    
        if multi_gpu:
            model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
    except Exception as e:
        with open('outputs/error.txt', 'w') as f:
            f.write(str(e) + '\n')
        raise e
    
    
    # Test model using appropriate method
    if args.multitask:
        if args.test_method == "cluster":
            # Test by most similar cluster
            dataset, df_test = dataset_by_cluster(df, df_train, num_tasks)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                   drop_last=False, num_workers=4, pin_memory=True)
            test_metrics_dict, cm = test_by_cluster(model, dataloader, df_test, device)
        elif args.test_method == "mv":
            # Test by majority vote
            datasets = dataset_by_mv(df, num_tasks)
            dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                     drop_last=False, num_workers=4, pin_memory=True) 
                          for dataset in datasets]
            test_metrics_dict, cm = test_by_mv(model, dataloaders, device)
        else:
            raise ValueError(f"Unknown test method: {args.test_method}")
    else:
        # Single task model
        datasets = dataset_by_mv(df, num_tasks)
        dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                 drop_last=False, num_workers=4, pin_memory=True) 
                      for dataset in datasets]
        test_metrics_dict, cm = test_by_mv(model, dataloaders, device)

    # Print confusion matrix
    class_names = ["non-tumor", "tumor"] if args.classification_task == "tumor" else ["non-TIL", "TIL"]
    print_confusion_matrix(cm, class_names)
    
    print(f"Testing completed. Results saved to {args.output_dir}/test_result_{exp_name}.json")
