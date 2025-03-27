import numpy as np
import pandas as pd
from skimage import io, color
import cuml
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dask.array as da
import dask.bag as db
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import rmm

# Step 1: Set up Dask with CUDA and configure RMM
def setup_dask_cluster():
    """
    Set up a Dask cluster with 2 GPUs and configure RMM for memory management.
    Returns a Dask client.
    """
    # Configure RMM to use a pool allocator with a maximum size (e.g., 50% of GPU memory)
    rmm.reinitialize(pool_allocator=True, initial_pool_size=8*1024*1024*1024)  # 8 GB pool
    
    cluster = LocalCUDACluster(
        n_workers=2,  # One worker per GPU
        threads_per_worker=1,
        memory_limit='16GB'  # Limit each worker to 16 GB of memory
    )
    client = Client(cluster)
    print("Dask cluster set up with 2 GPUs.")
    print(client)
    return client

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

# Step 4: Load images and extract features using Dask with batching
def extract_features_from_dataset(csv_path, test_set, bins_per_channel=8, client=None, batch_size=1000):
    """
    Load images from a CSV file, filter out the test set, and compute LAB histograms in parallel with batching.
    Returns a numpy array of features and a list of image paths.
    """
    df = pd.read_csv(csv_path)
    df = df[df['dataset'] != test_set]
    image_paths = df['img_path'].tolist()
    
    # Create a Dask bag from the image paths
    bag = db.from_sequence(image_paths, partition_size=batch_size)
    
    def process_image(img_path):
        try:
            img = load_image(img_path)
            hist = compute_lab_histogram(img, bins_per_channel)
            return (img_path, hist)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return (img_path, None)
    
    # Compute features in parallel
    results = bag.map(process_image).compute()
    
    # Filter out failed computations
    valid_results = [(img_path, hist) for img_path, hist in results if hist is not None]
    if not valid_results:
        raise ValueError("No valid features extracted from the dataset.")
    
    image_paths, features = zip(*valid_results)
    features = np.array(features)
    
    return features, list(image_paths)

# Step 5: Clustering pipeline using cuml (GPU-accelerated)
def cluster_images(features, min_cluster_size=50, max_clusters=10):
    """
    Apply UMAP and HDBSCAN to cluster the features using GPU acceleration.
    Ensures the number of clusters does not exceed max_clusters (10).
    Returns the UMAP model, HDBSCAN model, UMAP embeddings, and cluster labels.
    """
    print("Running UMAP on GPU...")
    umap_model = UMAP(
        n_components=2,
        n_neighbors=15,  # Reduced to lower memory usage
        min_dist=0.1,
        random_state=42
    )
    umap_embeddings = umap_model.fit_transform(features)
    
    print("Running HDBSCANceeds max_clusters...")
    # Initial parameters
    min_samples = max(2, min(1023, len(features) // 50))  # Ensure valid range
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(min_cluster_size, len(features) // max_clusters),
        min_samples=min_samples,
        cluster_selection_epsilon=0.5,
        cluster_selection_method='eom'
    )
    cluster_labels = hdbscan_model.fit_predict(umap_embeddings)
    
    # Convert to numpy for compatibility
    umap_embeddings = umap_embeddings.to_numpy()
    cluster_labels = cluster_labels.to_numpy()
    
    # Count the number of clusters (excluding noise points labeled -1)
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    print(f"Initial number of clusters formed: {n_clusters}")
    
    # Adjust parameters until n_clusters <= max_clusters
    max_attempts = 10  # Limit the number of adjustment attempts
    attempt = 0
    
    while n_clusters > max_clusters and attempt < max_attempts:
        print(f"Too many clusters ({n_clusters}). Adjusting parameters (attempt {attempt + 1}/{max_attempts})...")
        # Increase min_cluster_size to reduce the number of clusters
        min_cluster_size = int(min_cluster_size * 1.5)
        # Recalculate min_samples, keeping it in the valid range
        min_samples = max(2, min(1023, len(features) // 50))
        
        # Ensure min_cluster_size doesn't exceed a reasonable limit
        if min_cluster_size >= len(features) // 2:
            print(f"min_cluster_size reached limit ({min_cluster_size}). Stopping adjustments.")
            break
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.5,
            cluster_selection_method='eom'
        )
        cluster_labels = hdbscan_model.fit_predict(umap_embeddings)
        cluster_labels = cluster_labels.to_numpy()
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        print(f"Number of clusters after adjustment: {n_clusters}")
        attempt += 1
    
    # Final check and warning
    if n_clusters > max_clusters:
        print(f"Warning: Could not reduce clusters to {max_clusters} or fewer. Final number of clusters: {n_clusters}")
    else:
        print(f"Successfully reduced to {n_clusters} clusters (≤ {max_clusters}).")
    
    # Compute silhouette score if possible
    if len(np.unique(cluster_labels)) > 1:
        sil_score = silhouette_score(umap_embeddings, cluster_labels)
        print(f"Silhouette score: {sil_score:.3f}")
    else:
        print("Not enough clusters to compute silhouette score.")
    
    return umap_model, hdbscan_model, umap_embeddings, cluster_labels

# Step 6: Visualize the clusters (CPU-based)
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
    plt.title(f"HDBSCAN Clustering with UMAP ({task.upper()} - Excluding {test_set})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(output_path)
    plt.show()

# Step 7: Save the models and parameters
def save_models_and_params(umap_model, hdbscan_model, bins_per_channel, umap_embeddings, cluster_labels, image_paths, task, test_set):
    """
    Save the UMAP model, HDBSCAN model, and other parameters for later use.
    """
    joblib.dump(umap_model, f"umap_model_{task}_{test_set}.joblib")
    joblib.dump(hdbscan_model, f"hdbscan_model_{task}_{test_set}.joblib")
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
    # Set up Dask cluster with 2 GPUs
    client = setup_dask_cluster()
    
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
        
        # Step 1: Extract features from the dataset (parallelized with Dask)
        features, image_paths = extract_features_from_dataset(
            csv_path,
            test_set,
            bins_per_channel=bins_per_channel,
            client=client,
            batch_size=100  # Adjust based on GPU memory
        )
        
        # Skip if no images remain after filtering
        if len(features) == 0:
            print(f"No images left after excluding {test_set}. Skipping...")
            continue
        
        # Step 2: Cluster the images (GPU-accelerated)
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
    
    # Close the Dask client
    client.close()

if __name__ == "__main__":
    for task in ['tumor']: #, 'TIL']:
        main(task)
