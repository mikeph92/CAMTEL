import os
import numpy as np
import pandas as pd
from skimage import io, color
from umap import UMAP
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from scipy.spatial.distance import cosine, correlation
from skimage.feature import graycomatrix, graycoprops
import datetime

# Constants
OUTPUT_DIR = "outputs"
MIN_CLUSTERS = 4
MAX_CLUSTERS = 10
UMAP_N_COMPONENTS = 10
UMAP_N_COMPONENTS_VIS = 2
BINS_PER_CHANNEL = 8

# Distance Metrics
def cosine_distance(X, Y):
    """Calculate cosine distance between two vectors."""
    return cosine(X, Y)

def correlation_distance(X, Y):
    """Calculate correlation distance between two vectors."""
    return correlation(X, Y)

# Utility Functions
def create_output_directories(task: str, test_set: str) -> tuple:
    """
    Create output directories for saving models, results, and images.
    Args:
        task (str): Task name (e.g., 'tumor' or 'til').
        test_set (str): Test set name (e.g., 'pannuke').

    Returns:
        tuple: Paths to results, models, and images directories.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(OUTPUT_DIR, f"{task}_{test_set}_{timestamp}")
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "models")
    images_dir = os.path.join(base_dir, "images")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    return results_dir, models_dir, images_dir

def create_feature_specific_directories(task: str, test_set: str, feature_type: str, 
                                      clustering_method: str, distance_metric: str) -> tuple:
    """
    Create output directories for specific feature type and clustering method.
    Args:
        task (str): Task name (e.g., 'tumor' or 'til').
        test_set (str): Test set name (e.g., 'pannuke').
        feature_type (str): Feature extraction method.
        clustering_method (str): Clustering method.
        distance_metric (str): Distance metric.

    Returns:
        tuple: Paths to results, models, and images directories.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(OUTPUT_DIR, f"{task}_{test_set}_{feature_type}_{clustering_method}_{distance_metric}_{timestamp}")
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "models")
    images_dir = os.path.join(base_dir, "images")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    return results_dir, models_dir, images_dir

# Feature Extraction Functions
def load_image(image_path: str) -> np.ndarray:
    """
    Load a preprocessed .tif image and convert to RGB format.
    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Image in RGB format as a numpy array.
    """
    img = io.imread(image_path)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img

def compute_non_uniform_histogram(image: np.ndarray, bins_per_channel: int = BINS_PER_CHANNEL) -> np.ndarray:
    """
    Compute a 3D histogram in LAB color space with non-uniform binning and log transformation.

    Args:
        image (np.ndarray): Input image in RGB format.
        bins_per_channel (int): Number of bins per channel.

    Returns:
        np.ndarray: Flattened histogram with log transformation applied.
    """
    lab_image = color.rgb2lab(image)
    lab_pixels = lab_image.reshape(-1, 3)

    # Define non-uniform bins based on data distribution (using quantiles)
    l_bins = np.percentile(lab_pixels[:, 0], np.linspace(0, 100, bins_per_channel + 1))
    a_bins = np.percentile(lab_pixels[:, 1], np.linspace(0, 100, bins_per_channel + 1))
    b_bins = np.percentile(lab_pixels[:, 2], np.linspace(0, 100, bins_per_channel + 1))

    hist, _ = np.histogramdd(
        lab_pixels,
        bins=(l_bins, a_bins, b_bins)
    )
    hist = hist / hist.sum()  # Normalize
    hist = np.log1p(hist)  # Apply log transformation
    return hist.flatten()

def extract_basic_lab_values(image: np.ndarray) -> np.ndarray:
    """
    Extract basic LAB color space values.
    
    Args:
        image (np.ndarray): Input image in RGB format.
        
    Returns:
        np.ndarray: Mean LAB values.
    """
    lab_image = color.rgb2lab(image)
    return np.mean(lab_image, axis=(0, 1))

def compute_color_moments(image: np.ndarray) -> np.ndarray:
    """
    Compute color moments (mean, std, skewness) for each channel in LAB color space.

    Args:
        image (np.ndarray): Input image in RGB format.

    Returns:
        np.ndarray: Feature vector with color moments and texture features.
    """
    lab_image = color.rgb2lab(image)
    lab_pixels = lab_image.reshape(-1, 3)

    moments = []
    for channel in range(3):
        mean = np.mean(lab_pixels[:, channel])
        std = np.std(lab_pixels[:, channel])
        skewness = stats.skew(lab_pixels[:, channel])
        # Higher order moments
        kurtosis = stats.kurtosis(lab_pixels[:, channel])
        moments.extend([mean, std, skewness, kurtosis])
    
    # Add texture features using GLCM
    gray = color.rgb2gray(image)
    gray = (gray * 255).astype(np.uint8)
    
    # Calculate GLCM
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    
    # Extract properties
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation_val = graycoprops(glcm, 'correlation').mean()
    
    moments.extend([contrast, dissimilarity, homogeneity, energy, correlation_val])

    return np.array(moments)

def compute_dominant_colors(image: np.ndarray, n_colors: int = 3) -> np.ndarray:
    """
    Extract dominant colors using K-Means clustering on LAB pixel values.

    Args:
        image (np.ndarray): Input image in RGB format.
        n_colors (int): Number of dominant colors to extract.

    Returns:
        np.ndarray: Feature vector of dominant colors and their proportions.
    """
    lab_image = color.rgb2lab(image)
    pixels = lab_image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # Compute proportions of each color
    proportions = np.bincount(labels, minlength=n_colors) / len(labels)

    # Concatenate centers and proportions
    features = np.concatenate([centers.flatten(), proportions])
    return features

def extract_features(
    image_paths: list,
    feature_type: str = "color_moments",
    bins_per_channel: int = BINS_PER_CHANNEL
) -> tuple:
    """
    Extract features from a list of images based on the specified feature type.

    Args:
        image_paths (list): List of image file paths.
        feature_type (str): Type of feature to extract ('histogram', 'lab', 'color_moments', 'dominant_colors').
        bins_per_channel (int): Number of bins per channel for histogram.

    Returns:
        tuple: Numpy array of features and list of valid image paths.
    """
    features = []
    valid_image_paths = []

    for img_path in image_paths:
        try:
            img = load_image(img_path)
            if feature_type == "histogram" or feature_type == "hist":
                hist = compute_non_uniform_histogram(img, bins_per_channel)
                features.append(hist)
            elif feature_type == "color_moments":
                moments = compute_color_moments(img)
                features.append(moments)
            elif feature_type == "lab":
                lab_values = extract_basic_lab_values(img)
                features.append(lab_values)
            elif feature_type == "dominant_colors":
                dominant = compute_dominant_colors(img)
                features.append(dominant)
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            valid_image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not features:
        raise ValueError("No valid features extracted from the dataset.")

    return np.array(features), valid_image_paths

# Dimensionality Reduction
def apply_dimensionality_reduction(features: np.ndarray, n_components: int = UMAP_N_COMPONENTS) -> np.ndarray:
    """
    Apply PCA to remove outliers and then UMAP with noise.
    
    Args:
        features (np.ndarray): Input features.
        n_components (int): Number of components for UMAP.
        
    Returns:
        np.ndarray: Reduced features.
    """
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA first to remove outliers
    n_pca_components = min(features_scaled.shape[0], features_scaled.shape[1])
    pca = PCA(n_components=n_pca_components)
    features_pca = pca.fit_transform(features_scaled)
    
    # Add noise to PCA results
    noise = np.random.normal(0, 0.01, features_pca.shape)
    features_pca_noisy = features_pca + noise
    
    # Apply UMAP
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric='cosine'  # Use cosine distance
    )
    embedding = reducer.fit_transform(features_pca_noisy)
    
    return embedding, reducer

# Clustering Functions
def cluster_with_kmeans(features: np.ndarray, min_clusters: int, max_clusters: int, metric: str = 'cosine') -> tuple:
    """
    Cluster features using K-Means and select the best number of clusters based on silhouette score.

    Args:
        features (np.ndarray): Input features.
        min_clusters (int): Minimum number of clusters.
        max_clusters (int): Maximum number of clusters.
        metric (str): Distance metric ('cosine' or 'correlation').

    Returns:
        tuple: Best K-Means model, cluster labels, and best number of clusters.
    """
    best_silhouette = -1
    best_model = None
    best_labels = None
    best_k = min_clusters

    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Calculate silhouette score
        sil_score = silhouette_score(features, labels, metric=metric)
        
        print(f"K-Means with k={k}: Silhouette Score = {sil_score:.3f}")

        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_model = kmeans
            best_labels = labels
            best_k = k

    print(f"Best k = {best_k} with Silhouette Score = {best_silhouette:.3f}")
    return best_model, best_labels, best_k

def cluster_with_hierarchical(features: np.ndarray, min_clusters: int, max_clusters: int, metric: str = 'cosine') -> tuple:
    """
    Cluster features using Hierarchical Clustering (Agglomerative) and select the best number 
    of clusters based on silhouette score.

    Args:
        features (np.ndarray): Input features.
        min_clusters (int): Minimum number of clusters.
        max_clusters (int): Maximum number of clusters.
        metric (str): Distance metric ('cosine' or 'correlation').

    Returns:
        tuple: Best Hierarchical Clustering model, cluster labels, and best number of clusters.
    """
    best_silhouette = -1
    best_model = None
    best_labels = None
    best_k = min_clusters

    for k in range(min_clusters, max_clusters + 1):
        # Note: for AgglomerativeClustering, 'metric' parameter is used instead of 'affinity' 
        # when using 'cosine' or 'correlation' with 'average' linkage
        model = AgglomerativeClustering(
            n_clusters=k, 
            linkage='average',
            metric=metric  # 'cosine' or 'correlation'
        )
        labels = model.fit_predict(features)
        
        # Calculate silhouette score
        sil_score = silhouette_score(features, labels, metric=metric)
        
        print(f"Hierarchical with k={k}, metric={metric}: Silhouette Score = {sil_score:.3f}")
        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_model = model
            best_labels = labels
            best_k = k
    print(f"Best k = {best_k} with Silhouette Score = {best_silhouette:.3f}")
    return best_model, best_labels, best_k
def cluster_with_gmm(features: np.ndarray, min_clusters: int, max_clusters: int) -> tuple:
    """
    Cluster features using Gaussian Mixture Model and select the best number of clusters based on BIC.

    Args:
        features (np.ndarray): Input features.
        min_clusters (int): Minimum number of clusters.
        max_clusters (int): Maximum number of clusters.

    Returns:
        tuple: Best GMM model, cluster labels, and best number of clusters.
    """
    best_bic = np.inf
    best_model = None
    best_labels = None
    best_k = min_clusters

    for k in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(features)
        bic = gmm.bic(features)

        print(f"GMM with k={k}: BIC = {bic:.3f}")

        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_labels = labels
            best_k = k

    print(f"Best k = {best_k} with BIC = {best_bic:.3f}")
    return best_model, best_labels, best_k


def cluster_images(
    features: np.ndarray,
    min_clusters: int,
    max_clusters: int,
    clustering_method: str = "gmm",
    distance_metric: str = "cosine"
) -> tuple:
    """
    Apply UMAP and clustering to the features.

    Args:
        features (np.ndarray): Input features.
        min_clusters (int): Minimum number of clusters.
        max_clusters (int): Maximum number of clusters.
        clustering_method (str): Clustering method ('gmm', 'hierarchical', or 'kmeans').
        distance_metric (str): Distance metric ('cosine' or 'correlation').

    Returns:
        tuple: UMAP model, clustering model, UMAP embeddings, cluster labels, and best k.
    """
    print("Applying PCA and then UMAP with noise for clustering...")
    umap_embeddings, umap_model = apply_dimensionality_reduction(features, UMAP_N_COMPONENTS)
    if clustering_method == "gmm":
        model, labels, best_k = cluster_with_gmm(umap_embeddings, min_clusters, max_clusters)
    elif clustering_method == "hierarchical":
        model, labels, best_k = cluster_with_hierarchical(umap_embeddings, min_clusters, max_clusters, distance_metric)
    elif clustering_method == "kmeans":
        model, labels, best_k = cluster_with_kmeans(umap_embeddings, min_clusters, max_clusters, distance_metric)
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")
    # Compute silhouette score
    if len(np.unique(labels)) > 1:  # Need at least 2 clusters for these metrics
        sil_score = silhouette_score(umap_embeddings, labels, metric=distance_metric)
        db_score = davies_bouldin_score(umap_embeddings, labels)
        ch_score = calinski_harabasz_score(umap_embeddings, labels)
        
        print(f"Evaluation Metrics:")
        print(f"Silhouette Score: {sil_score:.3f}")
        print(f"Davies-Bouldin Index: {db_score:.3f} (lower is better)")
        print(f"Calinski-Harabasz Index: {ch_score:.3f} (higher is better)")
    else:
        sil_score = db_score = ch_score = None
        print("Warning: Only one cluster found, cannot compute evaluation metrics")

    return umap_model, model, umap_embeddings, labels, best_k, (sil_score, db_score, ch_score)

# Visualization and Saving Functions
def visualize_clusters(
    umap_embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    task: str,
    test_set: str,
    output_path: str
) -> None:
    """
    Visualize the UMAP embeddings in 2D with cluster labels.

    Args:
        umap_embeddings_2d (np.ndarray): 2D UMAP embeddings.
        cluster_labels (np.ndarray): Cluster labels.
        task (str): Task name.
        test_set (str): Test set name.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=umap_embeddings_2d[:, 0],
        y=umap_embeddings_2d[:, 1],
        hue=cluster_labels,
        palette="deep",
        legend="full",
        s=50
    )
    plt.title(f"Clustering with UMAP ({task.upper()} - Excluding {test_set})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(output_path)
    plt.close()

def save_representative_images(
    features: np.ndarray,
    image_paths: list,
    cluster_labels: np.ndarray,
    images_dir: str
) -> None:
    """
    Save the 2 images nearest to each cluster's center.

    Args:
        features (np.ndarray): Features used for clustering.
        image_paths (list): List of image paths.
        cluster_labels (np.ndarray): Cluster labels.
        images_dir (str): Directory to save the images.
    """
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_features = features[cluster_indices]
        cluster_center = np.mean(cluster_features, axis=0)

        # Find the 2 closest images to the cluster center
        distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
        closest_indices = np.argsort(distances)[:2]
        closest_image_paths = [image_paths[cluster_indices[i]] for i in closest_indices]

        # Save the images
        cluster_dir = os.path.join(images_dir, f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)
        for i, img_path in enumerate(closest_image_paths):
            img = io.imread(img_path)
            output_path = os.path.join(cluster_dir, f"representative_{i}.png")
            io.imsave(output_path, img)

def save_results(
    umap_model: object,
    cluster_model: object,
    feature_type: str,
    umap_embeddings: np.ndarray,
    umap_embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    evaluation_metrics: tuple,
    clustering_method: str,
    distance_metric: str,
    image_paths: list,
    datasets: list,
    task: str,
    test_set: str,
    results_dir: str,
    models_dir: str,
    images_dir: str
) -> None:
    """
    Save models, parameters, clustering results, silhouette scores, and visualizations.

    Args:
        umap_model (object): Trained UMAP model.
        cluster_model (object): Trained clustering model.
        feature_type (str): Type of features used.
        umap_embeddings (np.ndarray): UMAP embeddings (10D).
        umap_embeddings_2d (np.ndarray): UMAP embeddings (2D for visualization).
        cluster_labels (np.ndarray): Cluster labels.
        evaluation_metrics (tuple): (Silhouette score, Davies-Bouldin index, Calinski-Harabasz index).
        clustering_method (str): Clustering method used.
        distance_metric (str): Distance metric used.
        image_paths (list): List of image paths.
        task (str): Task name.
        test_set (str): Test set name.
        results_dir (str): Directory for results.
        models_dir (str): Directory for models.
        images_dir (str): Directory for images.
    """
    # Save models
    joblib.dump(umap_model, os.path.join(models_dir, "umap_model.joblib"))
    joblib.dump(cluster_model, os.path.join(models_dir, "cluster_model.joblib"))
    params = {
        "feature_type": feature_type,
        "clustering_method": clustering_method,
        "distance_metric": distance_metric
    }
    joblib.dump(params, os.path.join(models_dir, "clustering_params.joblib"))



    # Save clustering results
    results_df = pd.DataFrame({
        "dataset": datasets,
        "img_path": image_paths,
        "labelCluster": cluster_labels
    })
    results_df.to_csv(os.path.join(results_dir, f"clustering_result_{task}_{test_set}.csv"), index=False)

    # Save evaluation metrics
    if evaluation_metrics[0] is not None:  # Check if metrics were calculated
        sil_score, db_score, ch_score = evaluation_metrics
        metrics_df = pd.DataFrame({
            "silhouette_score": [sil_score],
            "davies_bouldin_index": [db_score],
            "calinski_harabasz_index": [ch_score]
        })
        metrics_df.to_csv(os.path.join(results_dir, f"evaluation_metrics_{task}_{test_set}.csv"), index=False)

    # Visualize clusters
    visualize_clusters(
        umap_embeddings_2d,
        cluster_labels,
        task,
        test_set,
        os.path.join(images_dir, f"clusters_{task}_{test_set}.png")
    )

    # Save representative images
    save_representative_images(umap_embeddings, image_paths, cluster_labels, images_dir)

    print(f"Results for {task} (excluding {test_set}) with {feature_type} features, {clustering_method} clustering, and {distance_metric} distance saved successfully.")

# Main Pipeline
def process_dataset(
    csv_path: str,
    test_set: str,
    feature_type: str = "histogram",
    clustering_method: str = "gmm",
    distance_metric: str = "cosine",
    bins_per_channel: int = BINS_PER_CHANNEL,
    min_clusters: int = MIN_CLUSTERS,
    max_clusters: int = MAX_CLUSTERS
) -> None:
    """
    Process the dataset: extract features, cluster, and save results.
        distance_metric (str): Distance metric to use.
    Args:
        csv_path (str): Path to the CSV file containing image paths.
        test_set (str): Test set to exclude.
        feature_type (str): Type of feature to extract.
        clustering_method (str): Clustering method to use.
        bins_per_channel (int): Number of bins per channel for histogram.
        min_clusters (int): Minimum number of clusters.
        max_clusters (int): Maximum number of clusters.
    """
    # Load and filter dataset
    df = pd.read_csv(csv_path)
    df = df[df['dataset'] != test_set]
    df = df[['dataset', 'img_path']].drop_duplicates()
    datasets = df['dataset'].tolist()
    image_paths = df['img_path'].tolist()

    if not image_paths:
        print(f"No images left after excluding {test_set}. Skipping...")
        return

    # Extract features
    print(f"Extracting features using {feature_type}...")
    features, valid_image_paths = extract_features(
        image_paths,
        feature_type=feature_type,
        bins_per_channel=bins_per_channel
    )

    # Cluster images
    umap_model, cluster_model, umap_embeddings, cluster_labels, best_k, evaluation_metrics = cluster_images(
        features,
        min_clusters,
        max_clusters,
        clustering_method=clustering_method,
        distance_metric=distance_metric
    )

    # Reduce to 2D for visualization
    print("Running UMAP for visualization (2 dimensions)...")
    features_pca, _ = apply_dimensionality_reduction(features, min(features.shape[0], features.shape[1]))
    umap_model_2d = UMAP(
        n_components=UMAP_N_COMPONENTS_VIS,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric=distance_metric
    )
    umap_embeddings_2d = umap_model_2d.fit_transform(features_pca)
    feature_specific_results_dir, feature_specific_models_dir, feature_specific_images_dir = create_feature_specific_directories(
        task, test_set, feature_type, clustering_method, distance_metric
    )
    # Create output directories
    _, _, _ = create_output_directories(task, test_set)

    # Save results
    save_results(
        umap_model,
        cluster_model,
        feature_type,
        umap_embeddings,
        umap_embeddings_2d,
        cluster_labels,
        evaluation_metrics,
        clustering_method,
        distance_metric,
        valid_image_paths,
        datasets,
        task,
        test_set,
        feature_specific_results_dir,
        feature_specific_models_dir,
        feature_specific_images_dir
    )

def main(task: str) -> None:
    """
    Main function to run the clustering pipeline for a given task.

    Args:
        task (str): Task name ('tumor' or 'til').
    """
    if task.lower() == 'tumor':
        csv_path = "/home/michael/CAMTEL/dataset/tumor_dataset.csv"
        test_sets = [ 'pannuke', 'nucls'] #'ocelot'
    elif task.lower() == 'til':
        csv_path = "/home/michael/CAMTEL/dataset/TIL_dataset.csv"
        test_sets = ['lizard', 'nucls', 'cptacCoad', 'tcgaBrca']
    else:
        raise ValueError("Task must be 'tumor' or 'TIL'")

    feature_types = ['lab'] # 'histogram', 'lab', 'color_moments'
    clustering_methods = ['kmeans', 'gmm'] #  'hierarchical' if needed
    distance_metrics = ['cosine', 'correlation']
    
    for test_set in test_sets:
        for feature_type in feature_types:
            for clustering_method in clustering_methods:
                for distance_metric in distance_metrics:
                    print(f"\nProcessing {task} dataset, excluding test set: {test_set}")
                    print(f"Feature type: {feature_type}, Clustering method: {clustering_method}, Distance metric: {distance_metric}")
                    process_dataset(
                        csv_path,
                        test_set,
                        feature_type=feature_type,
                        clustering_method=clustering_method,
                        distance_metric=distance_metric,
                        bins_per_channel=BINS_PER_CHANNEL,
                        min_clusters=MIN_CLUSTERS,
                        max_clusters=MAX_CLUSTERS
                    )

if __name__ == "__main__":
    for task in ['tumor']:  # Add 'TIL' if needed
        main(task)
