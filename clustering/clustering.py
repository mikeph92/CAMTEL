import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib

matplotlib.use('Agg')




Image.MAX_IMAGE_PIXELS = None

# Function to select 5 out of 6 folders
def select_folders():
    folder_paths = {"lizard" : "/nfs/datasets/public/ProcessedHistology/Lizard/inputs/original/",
                    "ocelot" : "/nfs/datasets/public/ProcessedHistology/Ocelot/inputs/original/",
                    "pannuke" : "/nfs/datasets/public/ProcessedHistology/PanNuke/inputs/original/",
                    "cptacCoad" : "/nfs/datasets/public/ProcessedHistology/CPTAC-COAD/inputs/original/",
                    "tcgaBrca" : "/nfs/datasets/public/ProcessedHistology/TCGA-BRCA/inputs/original/",
                    "nucls" : "/nfs/datasets/public/ProcessedHistology/NuCLS/inputs/original/"
                    }
    return folder_paths
    
def get_path(dataset, img_name):
    path_dict = {
        ("ocelot"): "/home/michael/data/ProcessedHistology/Ocelot/inputs/original",
        ("lizard"): "/home/michael/data/ProcessedHistology/Lizard/inputs/original",
        ("pannuke"): "/home/michael/data/ProcessedHistology/PanNuke/inputs/original",
        ("nucls"): "/home/michael/data/ProcessedHistology/NuCLS/inputs/original",
        ("cptacCoad"): "/home/michael/data/ProcessedHistology/CPTAC-COAD/inputs/original",
        ("tcgaBrca"): "/home/michael/data/ProcessedHistology/TCGA-BRCA/inputs/original"
    }

    return f"{path_dict[(dataset)]}/{img_name}.tif"

# Function to convert images to LAB space and calculate mean and deviation
def process_images(folder_paths):
    image_data = []
    for dataset in folder_paths.keys():
        for filename in os.listdir(folder_paths[dataset]):
            if filename.endswith(".tif"):
                img_path = os.path.join(folder_paths[dataset], filename)
                img = Image.open(img_path).convert('LAB')
                lab_img = np.array(img)
                l, a, b = lab_img[:,:,0], lab_img[:,:,1], lab_img[:,:,2]
                l_mean, l_std = np.mean(l), np.std(l)
                a_mean, a_std = np.mean(a), np.std(a)
                b_mean, b_std = np.mean(b), np.std(b)
                image_data.append([dataset,img_path, l_mean, l_std, a_mean, a_std, b_mean, b_std])
    df = pd.DataFrame(image_data, columns=["dataset", "img_path", "l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std"])
    df.to_csv("./output/img_statistics.csv", index=None)

# Function to reduce dimensions using t-SNE
def reduce_dimensions(data, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_data = tsne.fit_transform(data)
    return reduced_data

# Function to cluster images using KMeans
def cluster_images(image_data, n_clusters=10):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(image_data)
    reduced_data = reduce_dimensions(scaled_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(reduced_data)
    centroids = kmeans.cluster_centers_
    return labels, reduced_data, centroids
    
# Function to save clusters plot as an image
def save_clusters_plot(labels, reduced_data, filename='./output/clusters_plot.png'):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_data = reduced_data[labels == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}', s=20)
    
    plt.title('KMeans Clustering with t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

# Function to save clustering results to CSV
def save_to_csv(image_paths, image_data, labels, filename='./output/clustering_results.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    combined_data = np.hstack((image_paths, image_data, labels.reshape(-1,1)))
    df = pd.DataFrame(combined_data, columns=["dataset", "img_path", "l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std", "labelCluster"])
    df.to_csv(filename, index=  None)

#Function to read the calculated image data from csv file
def read_from_csv(classification_task, test_set,filename='./output/img_statistics.csv'):
    df = pd.read_csv(filename)
    if classification_task == "tumor":
        df = df[df.dataset.isin(["pannuke", "nucls", "ocelot"])]
    else:
        df = df[df.dataset.isin(["lizard", "cptacCoad", "tcgaBrca", "nucls"])]
    
    df = df[df.dataset != test_set]
    df['img_path'] = df.apply(lambda row: get_path(row['dataset'], row['img_name']), axis = 1)
    image_paths = df[["dataset","img_path"]]
    image_data =  df.drop(columns = ["dataset","img_path", 'img_name'])
    return image_data, image_paths

# Function to find and plot images closest to each cluster centroid
def plot_closest_images_to_centroids(image_paths, image_data, labels, centroids, n_clusters=10, filename='./output/closest_images.png'):
    closest, _ = pairwise_distances_argmin_min(centroids, image_data)  # Find closest images
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(closest):
        img_path = image_paths.iloc[idx]['img_path']
        img = Image.open(img_path)
        plt.subplot(1, n_clusters, i + 1)
        plt.imshow(img)
        plt.title(f'Cluster {i}')
        plt.axis('off')
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()



# Main function
def main(classification_task, test_set):
    calculated_data_available = True

    if not calculated_data_available:
        folder_paths = select_folders()
        process_images(folder_paths)
    
    image_data, image_paths = read_from_csv(classification_task, test_set)
    
    best_n_clusters = 0
    best_silhouette = -1
    best_label = []
    best_centroids = []


    for i in range(5,7):
        labels, reduced_data, centroids = cluster_images(image_data, n_clusters=i)
        score = silhouette_score(reduced_data, labels)
        if score > best_silhouette:
            best_silhouette = score
            best_n_clusters = i
            best_label = labels
            best_centroids = centroids

    save_clusters_plot(best_label, reduced_data, filename=f"./output/cluster of for LAB.png")
    plot_closest_images_to_centroids(image_paths, image_data, best_label, best_centroids, n_clusters=best_n_clusters, filename=f"./output/closest_images.png")
    # save_to_csv(image_paths, image_data, best_label, filename=f"./output/clustering_result_{classification_task}_{test_set}.csv")
    # with open(f'output/centroids_{classification_task}_{test_set}.txt', 'w') as f:
    #     for item in best_centroids:
    #         f.write(f"{item}\n")  # Writing each value on a new line    
    # print((f'Silhouette Score for {best_n_clusters} clusters: {score}\n'))


parser = argparse.ArgumentParser(description='clustering')
parser.add_argument('--classification-task', type=str, default='tumor', help='classification task: tumor or TIL') 
parser.add_argument('--testset', type=str, default='ocelot', help='dataset used for testing: ocelot, pannuke, nucls (tumor) or lizard, cptacCoad, tcgaBrca, nucls (TIL)') 

if __name__ == "__main__":
    args = parser.parse_args()
    classification_task = args.classification_task   # "tumor" or "TIL"
    test_set = args.testset        # "ocelot" or "lizard", "pannuke", "cptacCoad", "tcgaBrca", "nucls"          
    main(classification_task = classification_task, test_set = test_set)
