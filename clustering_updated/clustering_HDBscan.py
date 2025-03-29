import numpy as np
import pandas as pd
from skimage import io, color
import umap
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 2: Function to load an image
def load_image(image_path):
    """
    Load a preprocessed .tif image.
    Returns the image in RGB format as a numpy array.
    """
    img = io.imread(image_path)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img

# Step 3: Function to compute 3D LAB histogram
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

# Step 4: Load images and extract features sequentially
def extract_features_from_dataset(csv_path, test_set, bins_per_channel=8):
    """
    Load images from a CSV file, filter out the test set, and compute LAB histograms sequentially.
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
    
    features = np.array(features)
    return features, valid_image_paths

# Step 5: Clustering pipeline using umap-learn and hdbscan (CPU-based)
def cluster_images(features, min_cluster_size=50, max_clusters=10):
    # [Unchanged code from previous version]
    print("Running UMAP on CPU...")
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    umap_embeddings = umap_model.fit_transform(features)
    
    print("Running HDBSCAN on CPU...")
    min_samples = max(2, min(1023, len(features) // 50))
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=max(min_cluster_size, len(features) // max_clusters),
        min_samples=min_samples,
        cluster_selection_epsilon=0.5,
        cluster_selection_method='eom'
    )
    cluster_labels = hdbscan_model.fit_predict(umap_embeddings)
    
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    print(f"Initial number of clusters formed: {n_clusters}")
    
    max_attempts = 10
    attempt = 0
    
    while n_clusters > max_clusters and attempt < max_attempts:
        print(f"Too many clusters ({n_clusters}). Adjusting parameters (attempt {attempt + 1}/{max_attempts})...")
        min_cluster_size = int(min_cluster_size * 1.5)
        min_samples = max(2, min(1023, len(features) // 50))
        
        if min_cluster_size >= len(features) // 2:
            print(f"min_cluster_size reached limit ({min_cluster_size}). Stopping adjustments.")
            break
        
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.5,
            cluster_selection_method='eom'
        )
        cluster_labels = hdbscan_model.fit_predict(umap_embeddings)
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        print(f"Number of clusters after adjustment: {n_clusters}")
        attempt += 1
    
    if n_clusters > max_clusters:
        print(f"Warning: Could not reduce clusters to {max_clusters} or fewer. Final number of clusters: {n_clusters}")
    else:
        print(f"Successfully reduced to {n_clusters} clusters (≤ {max_clusters}).")
    
    if len(np.unique(cluster_labels)) > 1:
        sil_score = silhouette_score(umap_embeddings, cluster_labels)
        print(f"Silhouette score: {sil_score:.3f}")
    else:
        print("Not enough clusters to compute silhouette score.")
    
    return umap_model, hdbscan_model, umap_embeddings, cluster_labels

# Step 6: Visualize the clusters (CPU-based)
def visualize_clusters(umap_embeddings, cluster_labels, task, test_set, output_path="clusters.png"):
    # [Unchanged code from previous version]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=umap_embeddings[:, 0],
        y=umap_embeddings[:, 1],
        hue=cluster_labels,
        palette="deep",
        legend="full",
        s=50
    )
    plt.title(f"HDBSCAN Clustering with UMAP ({task.upper()} - Excluding {test_set})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(output_path)
    plt.show()

# Step 7: Save the models and parameters
def save_models_and_params(umap_model, hdbscan_model, bins_per_channel, umap_embeddings, cluster_labels, image_paths, task, test_set):
    # [Unchanged code from previous version]
    joblib.dump(umap_model, f"umap_model_{task}_{test_set}.joblib")
    joblib.dump(hdbscan_model, f"hdbscan_model_{task}_{test_set}.joblib")
    params = {"bins_per_channel": bins_per_channel}
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
    # Parameters
    bins_per_channel = 8
    min_cluster_size = 50
    max_clusters = 10
    
    # Define dataset paths and test sets based on task
    if task.lower() == 'tumor':
        csv_path = "/home/michael/CAMTEL/dataset/tumor_dataset.csv"
        test_sets = ['ocelot', 'pannuke', 'nucls']
    elif task.lower() == 'til':
        csv_path = "/home/michael/CAMTEL/dataset/TIL_dataset.csv"
        test_sets = ['lizard', 'nucls', 'cptacCoad', 'tcgaBrca']
    else:
        raise ValueError("Task must be 'tumor' or 'TIL'")

    # Iterate over each test set
    for test_set in test_sets:
        print(f"\nProcessing {task} dataset, excluding test set: {test_set}")
        
        # Step 1: Extract features from the dataset (sequentially)
        features, image_paths = extract_features_from_dataset(
            csv_path,
            test_set,
            bins_per_channel=bins_per_channel
        )
        
        # Skip if no images remain after filtering
        if len(features) == 0:
            print(f"No images left after excluding {test_set}. Skipping...")
            continue
        
        # Step 2: Cluster the images (CPU-based)
        umap_model, hdbscan_model, umap_embeddings, cluster_labels = cluster_images(
            features,
            min_cluster_size=min_cluster_size,
            max_clusters=max_clusters
        )
        
        # Step 3: Visualize the clusters
        visualize_clusters(umap_embeddings, cluster_labels, task, test_set, f"clusters_{task}_{test_set}.png")
        
        # Step 4: Save the models and parameters
        save_models_and_params(
            umap_model,
            hdbscan_model,
            bins_per_channel,
            umap_embeddings,
            cluster_labels,
            image_paths,
            task,
            test_set
        )

if __name__ == "__main__":
    for task in ['tumor']:  # , 'TIL']:
        main(task)