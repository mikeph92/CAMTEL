import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from randstainna import RandStainNA

def get_path(dataset, img_name, datatype):
    path_dict = {
        ("ocelot", "original"): "/home/michael/data/ProcessedHistology/Ocelot/inputs/original",
        ("lizard", "original"): "/home/michael/data/ProcessedHistology/Lizard/inputs/original",
        ("pannuke", "original"): "/home/michael/data/ProcessedHistology/PanNuke/inputs/original",
        ("nucls", "original"): "/home/michael/data/ProcessedHistology/NuCLS/inputs/original",
        ("cptacCoad", "original"): "/home/michael/data/ProcessedHistology/CPTAC-COAD/inputs/original",
        ("tcgaBrca", "original"): "/home/michael/data/ProcessedHistology/TCGA-BRCA/inputs/original"
    }
    return f"{path_dict[(dataset, datatype)]}/{img_name}.tif"

class MultiTaskDataset(Dataset):
    def __init__(self, df, task, datatype='original', crop_size=224, is_training=True):
        self.datasets = df['dataset']
        self.img_paths = df['img_path']
        self.centerXs = df['centerX']
        self.centerYs = df['centerY']
        self.labels = df['label']
        self.clusters = df['labelCluster']
        self.crop_size = crop_size
        self.datatype = datatype
        self.is_training = is_training

        # Define augmentation transforms for training only
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.Pad(padding=int((224 - self.crop_size)/2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Only normalization for testing
            self.transform = transforms.Compose([
                transforms.Pad(padding=int((224 - self.crop_size)/2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        full_img = Image.open(img_path)
        if self.datasets[idx] == "lizard":
            crop_size = int(self.crop_size / 2)
            crop_box = (self.centerXs[idx] - crop_size / 2, self.centerYs[idx] - crop_size / 2,
                        self.centerXs[idx] + crop_size / 2, self.centerYs[idx] + crop_size / 2)
            image = full_img.crop(crop_box)
            image = image.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        else:
            crop_box = (self.centerXs[idx] - self.crop_size / 2, self.centerYs[idx] - self.crop_size / 2,
                        self.centerXs[idx] + self.crop_size / 2, self.centerYs[idx] + self.crop_size / 2)
            image = full_img.crop(crop_box)
        image = self.transform(image)
        label = self.labels[idx]
        cluster = self.clusters[idx]
        return image, label, cluster

class MultiTaskDatasetRandStainNA(Dataset):
    def __init__(self, df, task, testset, cluster, datatype='original', crop_size=224, is_training=True):
        self.datasets = df['dataset']
        self.img_names = df['img_name']
        self.centerXs = df['centerX']
        self.centerYs = df['centerY']
        self.labels = df['labelTumor'] if task == "tumor" else df['labelTIL']
        self.crop_size = crop_size
        self.datatype = datatype
        self.is_training = is_training
        
        cluster_text = f"_{cluster}" if cluster is not None else ""
        
        # Choose transforms based on whether this is training or testing
        base_transforms = [
            RandStainNA(yaml_file=f'yaml_config/{task}_{testset}{cluster_text}.yaml', std_hyper=0, probability=1.0, distribution='normal')
        ]
        
        if self.is_training:
            # Add augmentations for training
            base_transforms.extend([
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])
        
        # Common transforms for both training and testing
        base_transforms.extend([
            transforms.Pad(padding=int((224 - self.crop_size)/2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        dataset = self.datasets[idx]
        img_path = get_path(dataset, img_name, self.datatype)
        full_img = Image.open(img_path)
        if dataset == "lizard":
            crop_size = int(self.crop_size / 2)
            crop_box = (self.centerXs[idx] - crop_size / 2, self.centerYs[idx] - crop_size / 2,
                        self.centerXs[idx] + crop_size / 2, self.centerYs[idx] + crop_size / 2)
            image = full_img.crop(crop_box)
            image = image.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        else:
            crop_box = (self.centerXs[idx] - self.crop_size / 2, self.centerYs[idx] - self.crop_size / 2,
                        self.centerXs[idx] + self.crop_size / 2, self.centerYs[idx] + self.crop_size / 2)
            image = full_img.crop(crop_box)
        image = self.transform(image.convert('RGB'))
        label = self.labels[idx]
        return image, label, img_name

class RandStainNADataset(Dataset):
    def __init__(self, df, task, testset, saved_path, crop_size=224, is_training=True):
        self.datasets = df['dataset']
        self.img_names = df['img_name']
        self.centerXs = df['centerX']
        self.centerYs = df['centerY']
        self.labels = df['labelTumor'] if task == "tumor" else df['labelTIL']
        self.crop_size = crop_size
        self.task = task
        self.testset = testset
        self.saved_path = saved_path
        self.is_training = is_training
        
        # Choose transforms based on whether this is training or testing
        transforms_list = []
        
        if self.is_training:
            # Add augmentations for training
            transforms_list.extend([
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])
        
        # Common transforms for both training and testing
        transforms_list.extend([
            transforms.Pad(padding=int((224 - self.crop_size)/2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = f'{self.saved_path}/{img_name}.tif'
        full_img = Image.open(img_path)
        if self.datasets[idx] == "lizard":
            crop_size = int(self.crop_size / 2)
            crop_box = (self.centerXs[idx] - crop_size / 2, self.centerYs[idx] - crop_size / 2,
                        self.centerXs[idx] + crop_size / 2, self.centerYs[idx] + crop_size / 2)
            image = full_img.crop(crop_box)
            image = image.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        else:
            crop_box = (self.centerXs[idx] - self.crop_size / 2, self.centerYs[idx] - self.crop_size / 2,
                        self.centerXs[idx] + self.crop_size / 2, self.centerYs[idx] + self.crop_size / 2)
            image = full_img.crop(crop_box)
        image = self.transform(image)
        label = self.labels[idx]
        return image, label, img_name
    