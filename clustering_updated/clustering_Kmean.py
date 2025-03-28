import numpy as np
import pandas as pd
from skimage import io, color
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Function to load an image
def load_image(image_path):
    """
    Load a preprocessed .tif image.
    Returns the image in RGB format as a numpy array.
    """
    img = io.imread(image_path)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img

# Step 2: Function to compute 3D LAB histogram
def compute_lab_histogram(image, bins_per_channel=8):
    """
    Compute a 3D histogram of the image in LAB color space.
    Returns a flattened histogram (512 elements for 8 bins per channel).
    """
    lab_image = color.rgb2lab(image)
    lab_pixels = lab_image.reshape(-1, 3)
    hist, _ = np.histogramdd(
        lab_pixels,
        bins=(bins_per_channel, bins_per_channel, bins_per_channel),
        range=((0, 100), (-128, 127), (-128, 127))
    )
    hist = hist / hist.sum()
    hist_flat = hist.flatten()
    return hist_flat

# Step 3: Load images and extract features sequentially
def extract_features_from_dataset(csv_path, test_set, bins_per_channel=8):
    """
    Load images from a CSV file, filter out the test set, and compute LAB histograms.
    Returns a numpy array of features and a list of image paths.
    """
    df = pd.read_csv(csv_path)
    df = df[df['dataset'] != test_set]
    image_paths = df['img_path'].tolist()
    
    features = []
    valid_image_paths = []
    
    for img_path in image_paths:
        try:
            img = load_image(img_path)
            hist = compute_lab_histogram(img, bins_per_channel)
            features.append(hist)
            valid_image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    if not features:
        raise ValueError("No valid features extracted from the dataset.")
    
    return np.array(features), valid_image_paths

# Step 4: Clustering pipeline using scikit-learn K-Means
def cluster_images(features, min_clusters=4, max_clusters=10):
    """
    Apply UMAP and K-Means to cluster the features using CPU.
    Evaluates k from min_clusters to max_clusters and selects the best based on Silhouette Score.
    Returns the UMAP model, K-Means model, UMAP embeddings, and cluster labels for the best k.
    """
    print("Running UMAP on CPU...")
    umap_model = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    umap_embeddings = umap_model.fit_transform(features)
    
    best_k = min_clusters
    best_silhouette = -1
    best_labels = None
    best_model = None
    inertias = []
    
    print(f"Running K-Means for k = {min_clusters} to {max_clusters}...")
    for k in range(min_clusters, max_clusters + 1):
        kmeans_model = KMeans(
            n_clusters=k,
            random_state=42,
            max_iter=300
        )
        cluster_labels = kmeans_model.fit_predict(umap_embeddings)
        
        # Compute Silhouette Score
        sil_score = silhouette_score(umap_embeddings, cluster_labels)
        inertia = kmeans_model.inertia_
        inertias.append(inertia)
        
        print(f"k={k}: Silhouette Score = {sil_score:.3f}, Inertia = {inertia:.3f}")
        
        # Update best model if silhouette score improves
        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_k = k
            best_labels = cluster_labels
            best_model = kmeans_model
    
    print(f"Best k = {best_k} with Silhouette Score = {best_silhouette:.3f}")
    
    # Plot Inertia for elbow method
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), inertias, marker='o')
    plt.title("Inertia vs. Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.savefig("inertia_plot.png")
    
    return umap_model, best_model, umap_embeddings, best_labels

# Step 5: Visualize the clusters
def visualize_clusters(umap_embeddings, cluster_labels, task, test_set, output_path="clusters.png"):
    """
    Visualize the UMAP embeddings with cluster labels.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=umap_embeddings[:, 0],
        y=umap_embeddings[:, 1],
        hue=cluster_labels,
        palette="deep",
        legend="full",
        s=50
    )
    plt.title(f"K-Means Clustering with UMAP ({task.upper()} - Excluding {test_set})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(output_path)
    plt.show()

# Step 6: Save the models and parameters
def save_models_and_params(umap_model, kmeans_model, bins_per_channel, umap_embeddings, cluster_labels, image_paths, task, test_set):
    """
    Save the UMAP model, K-Means model, and other parameters for later use.
    """
    joblib.dump(umap_model, f"umap_model_{task}_{test_set}.joblib")
    joblib.dump(kmeans_model, f"kmeans_model_{task}_{test_set}.joblib")
    params = {
        "bins_per_channel": bins_per_channel
    }
    joblib.dump(params, f"clustering_params_{task}_{test_set}.joblib")
    results_df = pd.DataFrame({
        "image_path": image_paths,
        "cluster_label": cluster_labels,
        "umap_dim1": umap_embeddings[:, 0],
        "umap_dim2": umap_embeddings[:, 1]
    })
    results_df.to_csv(f"clustering_results_{task}_{test_set}.csv", index=False)
    print(f"Models and parameters for {task} (excluding {test_set}) saved successfully.")

# Main function to run the pipeline and save the models
def main(task):
    bins_per_channel = 8
    min_clusters = 4
    max_clusters = 10
    
    if task.lower() == 'tumor':
        csv_path = "/home/michael/CAMTEL/dataset/tumor_dataset.csv"
        test_sets = ['ocelot', 'pannuke', 'nucls']
    elif task.lower() == 'til':
        csv_path = "/home/michael/CAMTEL/dataset/TIL_dataset.csv"
        test_sets = ['lizard', 'nucls', 'cptacCoad', 'tcgaBrca']
    else:
        raise ValueError("Task must be 'tumor' or 'TIL'")

    for test_set in test_sets:
        print(f"\nProcessing {task} dataset, excluding test set: {test_set}")
        
        # Step 1: Extract features
        features, image_paths = extract_features_from_dataset(
            csv_path,
            test_set,
            bins_per_channel=bins_per_channel
        )
        
        if len(features) == 0:
            print(f"No images left after excluding {test_set}. Skipping...")
            continue
        
        # Step 2: Cluster the images
        umap_model, kmeans_model, umap_embeddings, cluster_labels = cluster_images(
            features,
            min_clusters=min_clusters,
            max_clusters=max_clusters
        )
        
        # Step 3: Visualize the clusters
        visualize_clusters(umap_embeddings, cluster_labels, task, test_set, f"clusters_{task}_{test_set}.png")
        
        # Step 4: Save the models and parameters
        save_models_and_params(
            umap_model,
            kmeans_model,
            bins_per_channel,
            umap_embeddings,
            cluster_labels,
            image_paths,
            task,
            test_set
        )

if __name__ == "__main__":
    for task in ['tumor']:  # Add 'TIL' if needed
        main(task)