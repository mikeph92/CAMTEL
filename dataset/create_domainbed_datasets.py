import pandas as pd
import os
from PIL import Image
import random

CUT_SIZE = 96
TASK = 'tumor'

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
def __main__():
    df = pd.read_csv('full_dataset.csv')
    #df.columns = ['dataset', 'img_name', 'centerX', 'centerY', 'labelTumor', 'labelTIL']

    save_dir = f'/home/michael/data/domainbed/{CUT_SIZE}'
    os.makedirs(save_dir, exist_ok=True)

    targetCol = 'labelTumor' if TASK == 'tumor' else 'labelTIL'
    datasets = ['ocelot', 'pannuke', 'nucls'] if TASK == 'tumor' else ['lizard', 'cptacCoad', 'tcgaBrca']

    EXTRACT_SAMPLES = True
    if not EXTRACT_SAMPLES:
        #extract all images
        for dataset in datasets:
            print(f"Processing {dataset}...")
            dataset_dir = save_dir + '/' + dataset
            os.makedirs(dataset_dir, exist_ok=True)

            df_dataset = df[df.dataset == dataset]
            for _, row in df_dataset.iterrows():
                img_path = get_path(row['dataset'], row['img_name'])
                img = Image.open(img_path)
                centerX, centerY = row['centerX'], row['centerY']
                left = max(centerX - CUT_SIZE // 2, 0)
                top = max(centerY - CUT_SIZE // 2, 0)
                right = left + CUT_SIZE
                bottom = top + CUT_SIZE
                img_cropped = img.crop((left, top, right, bottom))
                
                label_dir = f"{dataset_dir}/{int(row[targetCol])}"
                os.makedirs(label_dir, exist_ok=True)
                img_cropped.save(f"{label_dir}/{row['img_name']}_{row['centerX']}_{row['centerY']}.tif")

    else:
        #extract samples
        n = 20  # Number of images to select per label

        for dataset in datasets:
            print(f"Selecting samples for {dataset}...")
            dataset_dir = save_dir + '/' + dataset
            sample_dir = f"{save_dir}/samples/{dataset}"
            os.makedirs(sample_dir, exist_ok=True)

            df_dataset = df[df.dataset == dataset]
            for label in df_dataset[targetCol].unique():
                df_label = df_dataset[df_dataset[targetCol] == label]
                sampled_rows = df_label.sample(n=min(n, len(df_label)), random_state=42)

                label_sample_dir = f"{sample_dir}/{int(label)}"
                os.makedirs(label_sample_dir, exist_ok=True)

                for _, row in sampled_rows.iterrows():
                    img_path = get_path(row['dataset'], row['img_name'])
                    img = Image.open(img_path)
                    centerX, centerY = row['centerX'], row['centerY']
                    left = max(centerX - CUT_SIZE // 2, 0)
                    top = max(centerY - CUT_SIZE // 2, 0)
                    right = left + CUT_SIZE
                    bottom = top + CUT_SIZE
                    img_cropped = img.crop((left, top, right, bottom))
                    img_cropped.save(f"{label_sample_dir}/{row['img_name']}_{row['centerX']}_{row['centerY']}.tif")

if __name__ == '__main__':
    __main__()
