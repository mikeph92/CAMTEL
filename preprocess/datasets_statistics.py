import os
import cv2
import numpy as np
import time
import argparse
import yaml
import json
import random
import copy
from skimage import color
from fitter import Fitter 
import pandas as pd

parser = argparse.ArgumentParser(description='norm&jitter dataset lab statistics')

parser.add_argument('--methods', type=str, default='Reinhard',
                    help='colornorm_methods')
parser.add_argument('--color-space', type=str, default='LAB', choices=['LAB', 'HED', 'HSV'], 
                    help='dataset statistics color space')
parser.add_argument('--random', action='store_true', default=True,
                    help='random shuffle sample')
parser.add_argument('--classification-task', default='tumor', help='tumor or TIL')
parser.add_argument('--testset', default='ocelot', help='ocelot, pannuke or nucls for tumor, lizard, nucls, cptacCoad or tcgaBrca for TIL')


def _parse_args():
    args = parser.parse_args()

    return args


def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg, std)


if __name__ == '__main__':

    args = _parse_args()

    
    input_path = f'../clustering/output/clustering_result_{args.classification_task}_{args.testset}.csv'
    df = pd.read_csv(input_path)

    # for i in df.labelCluster.unique():
    labL_avg_List = []
    labA_avg_List = []
    labB_avg_List = []
    labL_std_List = []
    labA_std_List = []
    labB_std_List = []

    df_filtered = df.copy() #[df.labelCluster == i]

    for _,row in df_filtered.iterrows():
        path_img = row['img_path']
        img = cv2.imread(path_img)
        try:  # debug
            if args.color_space == "LAB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            elif args.color_space == "HED":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = color.rgb2hed(img)
            elif args.color_space == "HSV":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            else:
                print("wrong color space: {}!!".format(args.color_space))
            img_avg, img_std = getavgstd(img)
        except:
            continue
            print(path_img)
        labL_avg_List.append(img_avg[0])
        labA_avg_List.append(img_avg[1])
        labB_avg_List.append(img_avg[2])
        labL_std_List.append(img_std[0])
        labA_std_List.append(img_std[1])
        labB_std_List.append(img_std[2])
    # t2 = time.time()
    # print(t2 - t1)
    l_avg_mean = np.mean(labL_avg_List).item()
    l_avg_std = np.std(labL_avg_List).item()
    l_std_mean = np.mean(labL_std_List).item()
    l_std_std = np.std(labL_std_List).item()
    a_avg_mean = np.mean(labA_avg_List).item()
    a_avg_std = np.std(labA_avg_List).item()
    a_std_mean = np.mean(labA_std_List).item()
    a_std_std = np.std(labA_std_List).item()
    b_avg_mean = np.mean(labB_avg_List).item()
    b_avg_std = np.std(labB_avg_List).item()
    b_std_mean = np.mean(labB_std_List).item()
    b_std_std = np.std(labB_std_List).item()

    std_avg_list = [labL_avg_List, labL_std_List, labA_avg_List, labA_std_List, labB_avg_List, labB_std_List]
    distribution = []
    for std_avg in std_avg_list:
        f = Fitter(std_avg, distributions=['norm', 'laplace'])
        f.fit()
        distribution.append(list(f.get_best(method='sumsquare_error').keys())[0]) 
        
    yaml_dict_lab = {
        'random': args.random,
        'color_space': 'LAB',
        'methods': args.methods,
        'L': {  # lab-L/hed-H
            'avg': {
                'mean': round(l_avg_mean, 3),
                'std': round(l_avg_std, 3),
                'distribution': distribution[0]
            },
            'std': {
                'mean': round(l_std_mean, 3),
                'std': round(l_std_std, 3),
                'distribution': distribution[1]
            }
        },
        'A': {  # lab-A/hed-E
            'avg': {
                'mean': round(a_avg_mean, 3),
                'std': round(a_avg_std, 3),
                'distribution': distribution[2]
            },
            'std': {
                'mean': round(a_std_mean, 3),
                'std': round(a_std_std, 3),
                'distribution': distribution[3]
            }
        },
        'B': {  # lab-B/hed-D
            'avg': {
                'mean': round(b_avg_mean, 3),
                'std': round(b_avg_std, 3),
                'distribution': distribution[4]
            },
            'std': {
                'mean': round(b_std_mean, 3),
                'std': round(b_std_std, 3),
                'distribution': distribution[5]
            }
        }
    }
    yaml_save_path = f'../yaml_config/{args.classification_task}_{args.testset}.yaml' #_{i}.yaml'
    with open(yaml_save_path, 'w') as f:
        yaml.dump(yaml_dict_lab, f)
        print('The dataset lab statistics has been saved in {}'.format(yaml_save_path))