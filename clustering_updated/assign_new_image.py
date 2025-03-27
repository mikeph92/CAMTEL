import numpy as np
from skimage import io, color
import joblib
import cuml
from cuml.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Step 3: Load the saved models and parameters
def load_models_and_params(task, test_set):
    """
    Load the saved UMAP model, HDBSCAN model, and parameters for the given task and test set.
    Returns the models, parameters, and original clustering results.
    """
    umap_model = joblib.load(f"umap_model_{task}_{test_set}.joblib")
    hdbscan_model = joblib.load(f"hdbscan_model_{task}_{test_set}.joblib")
    params = joblib.load(f"clustering_params_{task}_{test_set}.joblib")
    clustering_results = pd.read_csv(f"clustering_results_{task}_{test_set}.csv")
    return umap_model, hdbscan_model, params, clustering_results

# Step 4: Visualize the new image with the original clusters
def visualize_new_image_with_clusters(new_image_path, new_cluster_label, new_umap_embedding, clustering_results, task, test_set):
    """
    Visualize the new image in the UMAP space alongside the original clusters.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=clustering_results["umap_dim1"],
        y=clustering_results["umap_dim2"],
        hue=clustering_results["cluster_label"],
        palette="deep",
        legend="full",
        s=50
    )
    plt.scatter(
        new_umap_embedding[:, 0],
        new_umap_embedding[:, 1],
        c="red",
        marker="x",
        s=200,
        label=f"New Image (Cluster {new_cluster_label})"
    )
    plt.title(f"HDBSCAN Clustering with UMAP ({task.upper()} - Excluding {test_set})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.savefig(f"clusters_with_new_image_{task}_{test_set}_{new_image_path.split('/')[-1].split('.')[0]}.png")
    plt.show()

# Step 5: Standalone function to assign a new image to a cluster
def assign_new_image_to_cluster_standalone(new_image_path, task, test_set, visualize=True):
    """
    Assign a new image to an existing cluster using pre-trained UMAP and nearest-neighbor approach.
    
    Parameters:
    - new_image_path (str): Path to the new .tif image.
    - task (str): 'tumor' or 'TIL', determines which dataset and models to use.
    - test_set (str): The test set that was excluded during clustering (e.g., 'ocelot', 'nucls').
    - visualize (bool): Whether to generate and save a visualization of the new image in the UMAP space.
    
    Returns:
    - new_cluster_label (int): The cluster label assigned to the new image (-1 if an outlier).
    - new_umap_embedding (numpy array): The 2D UMAP embedding of the new image.
    """
    # Validate task and test_set
    if task.lower() == 'tumor':
        valid_test_sets = ['ocelot', 'pannuke', 'nucls']
    elif task.lower() == 'til':
        valid_test_sets = ['lizard', 'nucls', 'cptacCoad', 'tcgaBrca']
    else:
        raise ValueError("Task must be 'tumor' or 'TIL'")
    
    if test_set not in valid_test_sets:
        raise ValueError(f"Test set must be one of {valid_test_sets} for task {task}")

    # Load the saved models and parameters
    try:
        umap_model, hdbscan_model, params, clustering_results = load_models_and_params(task, test_set)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files for task '{task}' and test_set '{test_set}' not found. Ensure that train_and_save_clustering.py has been run first.") from e
    
    bins_per_channel = params["bins_per_channel"]
    
    # Process the new image
    print(f"Processing new image: {new_image_path}")
    try:
        img = load_image(new_image_path)
    except Exception as e:
        raise ValueError(f"Failed to load image {new_image_path}: {e}") from e
    
    hist = compute_lab_histogram(img, bins_per_channel)
    new_umap_embedding = umap_model.transform([hist])
    new_umap_embedding = new_umap_embedding.to_numpy()  # Convert to numpy
    
    # Use nearest neighbors to assign the new image to a cluster
    umap_embeddings = np.vstack((clustering_results["umap_dim1"].values, clustering_results["umap_dim2"].values)).T
    cluster_labels = clustering_results["cluster_label"].values
    
    # Fit a nearest neighbors model on the UMAP embeddings
    nn_model = NearestNeighbors(n_neighbors=1)
    nn_model.fit(umap_embeddings)
    
    # Find the nearest neighbor in the UMAP space
    distances, indices = nn_model.kneighbors(new_umap_embedding)
    nearest_idx = indices[0][0]
    new_cluster_label = cluster_labels[nearest_idx]
    
    print(f"New image assigned to cluster: {new_cluster_label}")
    if new_cluster_label == -1:
        print("The new image is an outlier and doesn't fit into any existing cluster.")
    
    # Visualize if requested
    if visualize:
        visualize_new_image_with_clusters(
            new_image_path,
            new_cluster_label,
            new_umap_embedding,
            clustering_results,
            task,
            test_set
        )
    
    return new_cluster_label, new_umap_embedding

# Example usage when running the script directly
if __name__ == "__main__":
    # Example parameters
    task = "tumor"
    test_set = "ocelot"
    new_image_path = "path/to/new/image.tif"
    
    # Call the standalone function
    new_cluster_label, new_umap_embedding = assign_new_image_to_cluster_standalone(
        new_image_path,
        task,
        test_set,
        visualize=True
    )
    