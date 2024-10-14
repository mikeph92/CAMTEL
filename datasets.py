import collections
import csv
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
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

    return f"{path_dict[(dataset,datatype)]}/{img_name}.tif"

# MultiTask Dataset class
class MultiTaskDataset(Dataset):
    def __init__(self, df, task, datatype = 'original', crop_size = 224):
        self.datasets = df['dataset']
        self.img_names = df['img_name']
        self.centerXs = df['centerX']
        self.centerYs = df['centerY']
        self.labels = df['labelTumor'] if task == "tumor" else df['labelTIL']
        self.crop_size = crop_size

        self.datatype = datatype

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Pad(padding = int((224 - self.crop_size)/2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.pos_weight = sum(self.labels == 0)/sum(self.labels == 1) if sum(self.labels == 1) != 0 else float('inf')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = get_path(self.datasets[idx], img_name,self.datatype)
        full_img = Image.open(img_path)
        if self.datasets[idx] == "lizard":
            crop_size = int(self.crop_size/2)           #lizard dataset has x20 magnification instead of x40
            crop_box = (self.centerXs[idx]-crop_size/2,self.centerYs[idx]-crop_size/2, self.centerXs[idx]+crop_size/2, self.centerYs[idx]+crop_size/2)
            image = full_img.crop(crop_box)
            image = image.resize((self.crop_size, self.crop_size), Image.LANCZOS)       #resize into standard size to fix with other datasets
        else:
            crop_box = (self.centerXs[idx]-self.crop_size/2,self.centerYs[idx]-self.crop_size/2, self.centerXs[idx]+self.crop_size/2, self.centerYs[idx]+self.crop_size/2)
            image = full_img.crop(crop_box)

        image = self.transform(image)

        label = self.labels[idx]
        return image, label, img_name
    

# MultiTask Dataset class
class MultiTaskDatasetRandStainNA(Dataset):
    def __init__(self, df, task, testset, cluster, datatype = 'original', crop_size = 224):
        self.datasets = df['dataset']
        self.img_names = df['img_name']
        self.centerXs = df['centerX']
        self.centerYs = df['centerY']
        self.labels = df['labelTumor'] if task == "tumor" else df['labelTIL']
        self.crop_size = crop_size

        self.datatype = datatype
        if cluster == None:
            cluster_text = ""
        else:
            cluster_text = f"_{cluster}"

        self.transform = transforms.Compose([
            RandStainNA(yaml_file=f'yaml_config/{task}_{testset}{cluster_text}.yaml', std_hyper=0, probability=1.0,
                               distribution='normal'),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Pad(padding = int((224 - self.crop_size)/2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.pos_weight = sum(self.labels == 0)/sum(self.labels == 1) if sum(self.labels == 1) != 0 else float('inf')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        dataset = self.datasets[idx]
        img_path = get_path(dataset, img_name,self.datatype)
        full_img = Image.open(img_path)
        if dataset == "lizard":
            crop_size = int(self.crop_size/2)           #lizard dataset has x20 magnification instead of x40
            crop_box = (self.centerXs[idx]-crop_size/2,self.centerYs[idx]-crop_size/2, self.centerXs[idx]+crop_size/2, self.centerYs[idx]+crop_size/2)
            image = full_img.crop(crop_box)
            image = image.resize((self.crop_size, self.crop_size), Image.LANCZOS)       #resize into standard size to fix with other datasets
        else:
            crop_box = (self.centerXs[idx]-self.crop_size/2,self.centerYs[idx]-self.crop_size/2, self.centerXs[idx]+self.crop_size/2, self.centerYs[idx]+self.crop_size/2)
            image = full_img.crop(crop_box)

        image = self.transform(image.convert('RGB'))

        label = self.labels[idx]
        
        return image, label, img_name, dataset

