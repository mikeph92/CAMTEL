
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.utils.data as data
import numpy as np
import pandas as pd
from PIL import Image
import argparse

from timm.data.transforms import RandStainNA_Attention


parser = argparse.ArgumentParser(description='norm&jitter dataset lab statistics')

parser.add_argument('--color-space', type=str, default='LAB', choices=['LAB', 'HED', 'HSV'],
                    help='dataset statistics color space')
parser.add_argument('--classification-task', default='tumor', help='tumor or TIL')
parser.add_argument('--testset', default='ocelot', help='ocelot, pannuke or nucls for tumor, lizard, nucls, cptacCoad or tcgaBrca for TIL')


def _parse_args():
    args = parser.parse_args()

    return args

def _setup_randstainna(classification_task, testset, color_space, cluster):
    color_jitter = {}
    color_jitter['brightness'] = 0.35
    color_jitter['contrast'] = 0.5
    color_jitter['saturation'] = 0
    color_jitter['hue'] = 0
    color_jitter['p'] = 1.0

    randstainna = {}
    randstainna['bg_yaml_file'] = f'./statistics/{classification_task}_{testset}/bg/{color_space}_{cluster}.yaml'
    randstainna['fg_yaml_file'] = f'./statistics/{classification_task}_{testset}/fg/{color_space}_{cluster}.yaml'
    randstainna['std_hyper'] = 0.5
    randstainna['probability'] = [0.3, 0.3, 0.4]
    randstainna['distribution'] = 'normal'

    randstainna_attention = {}
    randstainna_attention['color_jitter'] = color_jitter
    randstainna_attention['randstainna'] = randstainna
    randstainna_attention['fg'] = 'randstainna'
    randstainna_attention['bg'] = 'randstainna'

    return randstainna_attention


if __name__ == '__main__':
    args = _parse_args()

    input_path = f'../clustering/output/clustering_result_{args.classification_task}_{args.testset}.csv'
    df = pd.read_csv(input_path)
    clusters = np.append(df['labelCluster'].unique(),  'full') 
    # print(f'clusters: {clusters}')

    for _,row in df.iterrows():
        path_img = row['img_path']
        path_cam = path_img.replace('/nfs/datasets/public/ProcessedHistology/', '/nfs/users/michael/histoPath/')
        cluster = row['labelCluster']


        img = Image.open(path_img).convert('RGB')
        cam = Image.open(path_cam).convert('L')

        for cluster in clusters:
            save_path = path_img.replace('/nfs/datasets/public/ProcessedHistology/', f'/nfs/users/michael/augmented/{cluster}/')
            dir_path = os.path.dirname(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            randstainna = _setup_randstainna(args.classification_task, args.testset, args.color_space, cluster)

            aug_img,_,_,_,_,_ = RandStainNA_Attention(fg=randstainna['fg'], bg=randstainna['bg'], 
                                                color_jitter=randstainna['color_jitter'], 
                                                randstainna=randstainna['randstainna'], seg=cam)(img)
            aug_img.save(save_path)


