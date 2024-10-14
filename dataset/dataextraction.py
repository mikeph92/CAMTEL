# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import pickle


def get_datasets(datasets = ["ocelot", "pannuke", "nucls", "lizard", "cptacCoad", "tcgaBrca"]):
    full_datasets  =  {
        "ocelot": [["/nfs/datasets/public/ProcessedHistology/Ocelot/cells/tumour_cell/"],
                   [],
                   ["/nfs/datasets/public/ProcessedHistology/Ocelot/cells/background_cell/"]],
        "pannuke": [["/nfs/datasets/public/ProcessedHistology/PanNuke/cells/Neoplastic/"],
                    [],
                    ["/nfs/datasets/public/ProcessedHistology/PanNuke/cells/Connective/",
                     "/nfs/datasets/public/ProcessedHistology/PanNuke/cells/Dead/",
                     "/nfs/datasets/public/ProcessedHistology/PanNuke/cells/Inflammatory/",
                     "/nfs/datasets/public/ProcessedHistology/PanNuke/cells/Non-Neoplastic Epithelial/"]],
        "nucls":  [["/nfs/datasets/public/ProcessedHistology/NuCLS/cells/tumor/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/correction_tumor/"],
                   ["/nfs/datasets/public/ProcessedHistology/NuCLS/cells/correction_lymphocyte/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/lymphocyte/"],
                   ["/nfs/datasets/public/ProcessedHistology/NuCLS/cells/apoptotic_body/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/correction_apoptotic_body/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/correction_fibroblast/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/correction_macrophage/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/correction_mitotic_figure/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/correction_plasma_cell/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/ductal_epithelium/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/eosinophil/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/fibroblast/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/macrophage/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/mitotic_figure/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/myoepithelium/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/neutrophil/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/plasma_cell/",
                    "/nfs/datasets/public/ProcessedHistology/NuCLS/cells/vascular_endothelium/"]],
        "lizard": [[],
                   ["/nfs/datasets/public/ProcessedHistology/Lizard/cells/Lymphocyte/"],
                   ["/nfs/datasets/public/ProcessedHistology/Lizard/cells/Connective/",
                    "/nfs/datasets/public/ProcessedHistology/Lizard/cells/Eosinophil/",
                    "/nfs/datasets/public/ProcessedHistology/Lizard/cells/Neutrophil/",
                    "/nfs/datasets/public/ProcessedHistology/Lizard/cells/Plasma/"]],
        "cptacCoad": [[],
                      ["/nfs/datasets/public/ProcessedHistology/CPTAC-COAD/cells/Lymphocytes/"],
                      []],
        "tcgaBrca": [[],
                     ["/nfs/datasets/public/ProcessedHistology/TCGA-BRCA/cells/TILs/"],
                     []]
    }
    result = {key: full_datasets[key] for key in datasets}
    return result


def read_single_dir(dataset, dir_path, result = [], dir_type = "other"):
    for sub_folder in os.listdir(dir_path):
        sub_folder_path = os.path.join(dir_path, sub_folder)
        for file_name in os.listdir(sub_folder_path):
            img_name = file_name[:-4]  #remove file extension .npy, .pkl
            extension = file_name[-3:]
            file_path = os.path.join(sub_folder_path, file_name)

            locs = []

            #read location data from files, then extract location of center points and write to final result
            if extension == "npy":
                locs = np.load(file_path)
                
                for loc in locs:
                    if len(loc) == 2:   #points
                        locX = loc[0]
                        locY = loc[1]
                    elif  len(loc) == 4:    #boxes
                        locX = int((loc[0]+loc[2])/2)
                        locY = int((loc[1]+loc[3])/2)
                    result.append({
                        "dataset": dataset,
                        "img_name": img_name, 
                        "centerX": locX,
                        "centerY": locY,
                        "labelTIL": 1 if dir_type == "til" else 0, 
                        "labelTumor": 1 if dir_type == "tumor" else 0})
            elif extension == "pkl":
                with open(file_path, 'rb') as file:
                    locs = pickle.load(file)
                for loc in locs:
                    center_loc = np.mean(loc, axis = 0)
                    result.append({
                        "dataset": dataset,
                        "img_name": img_name, 
                        "centerX": int(center_loc[0]),
                        "centerY": int(center_loc[1]),
                        "labelTIL": 1 if dir_type == "til" else 0, 
                        "labelTumor": 1 if dir_type == "tumor" else 0})
    return result

def read_dataset(dataset, tumor_dirs = [],  til_dirs = [], other_dirs = []):
    result = []  

    #read lists of files of tumor cells
    for dir_path in tumor_dirs:
        read_single_dir(dataset, dir_path, result = result, dir_type="tumor")

    #read lists of files of TILs
    for dir_path in til_dirs:
        read_single_dir(dataset, dir_path, result = result, dir_type="til")
    
    #read lists of files of other cells
    for dir_path in other_dirs:
        read_single_dir(dataset, dir_path, result = result)

    return result
    
def write_to_csv(data, filename = "full_dataset.csv"):
    df = pd.DataFrame(data, columns=["dataset", "img_name", "centerX", "centerY", "labelTIL", "labelTumor"])
    df.to_csv(filename, mode='a', header=False, index=False)

def main():
    datasets = get_datasets()
    for key in datasets.keys():
        data = read_dataset(key, datasets[key][0], datasets[key][1], datasets[key][2])
        write_to_csv(data)

if __name__ == "__main__":
    main()
    



































        
