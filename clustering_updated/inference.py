# inference.py
import os
import numpy as np
import pandas as pd
from skimage import io, color
import joblib
from scipy import stats
from scipy.spatial.distance import cosine, correlation, euclidean
from skimage.feature import graycomatrix, graycoprops

# Constants from original code
OUTPUT_DIR = "outputs"
BINS_PER_CHANNEL = 8

# Feature Extraction Functions (copied from original code)
def load_image(image_path: str) -> np.ndarray:
    """Load a preprocessed .tif image and convert to RGB format."""
    img = io.imread(image_path)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img

def compute_non_uniform_histogram(image: np.ndarray, bins_per_channel: int = BINS_PER_CHANNEL) -> np.ndarray:
    """Compute a 3D histogram in LAB color space with non-uniform binning and log transformation."""
    lab_image = color.rgb2lab(image)
    lab_pixels = lab_image.reshape(-1, 3)
    l_bins = np.percentile(lab_pixels[:, 0], np.linspace(0, 100, bins_per_channel + 1))
    a_bins = np.percentile(lab_pixels[:, 1], np.linspace(0, 100, bins_per_channel + 1))
    b_bins = np.percentile(lab_pixels[:, 2], np.linspace(0, 100, bins_per_channel + 1))
    hist, _ = np.histogramdd(lab_pixels, bins=(l_bins, a_bins, b_bins))
    hist = hist / hist.sum()  # Normalize
    hist = np.log1p(hist)  # Apply log transformation
    return hist.flatten()

def extract_basic_lab_values(image: np.ndarray) -> np.ndarray:
    """Extract basic LAB color space values."""
    lab_image = color.rgb2lab(image)
    return np.mean(lab_image, axis=(0, 1))

def compute_color_moments(image: np.ndarray) -> np.ndarray:
    """Compute color moments (mean, std, skewness) for each channel in LAB color space."""
    lab_image = color.rgb2lab(image)
    lab_pixels = lab_image.reshape(-1, 3)
    moments = []
    for channel in range(3):
        mean = np.mean(lab_pixels[:, channel])
        std = np.std(lab_pixels[:, channel])
        skewness = stats.skew(lab_pixels[:, channel])
        kurtosis = stats.kurtosis(lab_pixels[:, channel])
        moments.extend([mean, std, skewness, kurtosis])
    gray = color.rgb2gray(image)
    gray = (gray * 255).astype(np.uint8)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation_val = graycoprops(glcm, 'correlation').mean()
    moments.extend([contrast, dissimilarity, homogeneity, energy, correlation_val])
    return np.array(moments)

def compute_dominant_colors(image: np.ndarray, n_colors: int = 3) -> np.ndarray:
    """Extract dominant colors using K-Means clustering on LAB pixel values."""
    from sklearn.cluster import KMeans
    lab_image = color.rgb2lab(image)
    pixels = lab_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    proportions = np.bincount(labels, minlength=n_colors) / len(labels)
    features = np.concatenate([centers.flatten(), proportions])
    return features

def extract_features(
    image_paths: list,
    feature_type: str,
    bins_per_channel: int = BINS_PER_CHANNEL
) -> tuple:
    """Extract features from a list of images based on the specified feature type."""
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

def infer_clusters(
    task: str,
    test_set: str,
    clustering_method: str,
    feature_type: str = "color_moments",
    distance_metric: str = "euclidean",
    csv_path: str = "/home/michael/CAMTEL/dataset/tumor_dataset.csv"
) -> None:
    """
    Infer clusters for test images from tumor_dataset.csv and save results.
    
    Args:
        task (str): Task name ('tumor' or 'til')
        test_set (str): Test set name to filter (e.g., 'pannuke')
        clustering_method (str): Clustering method used in training ('kmeans', 'gmm', 'hierarchical')
        feature_type (str): Feature type used in training
        distance_metric (str): Distance metric used in training
        csv_path (str): Path to tumor_dataset.csv
    """
    # Load dataset and filter for test_set
    df = pd.read_csv(csv_path)
    test_df = df[df['dataset'] == test_set][['dataset', 'img_path']].drop_duplicates()
    image_paths = test_df['img_path'].tolist()
    
    if not image_paths:
        print(f"No images found for test set {test_set}. Skipping...")
        return
    
    # Find the most recent models directory matching the parameters
    base_pattern = f"{task}_{test_set}_{feature_type}_{clustering_method}_{distance_metric}"
    models_dir = None
    for dir_name in os.listdir(OUTPUT_DIR):
        if base_pattern in dir_name and os.path.isdir(os.path.join(OUTPUT_DIR, dir_name)):
            models_dir = os.path.join(OUTPUT_DIR, dir_name, "models")
            break  # Use the first match (could sort by timestamp if needed)
    
    if not models_dir or not os.path.exists(models_dir):
        raise FileNotFoundError(f"No models found for {base_pattern} in {OUTPUT_DIR}")
    
    # Load saved models
    umap_model = joblib.load(os.path.join(models_dir, "umap_model.joblib"))
    cluster_model = joblib.load(os.path.join(models_dir, "cluster_model.joblib"))
    params = joblib.load(os.path.join(models_dir, "clustering_params.joblib"))
    
    # Verify parameters match
    if params["feature_type"] != feature_type or params["clustering_method"] != clustering_method or params["distance_metric"] != distance_metric:
        raise ValueError(f"Parameter mismatch: trained with {params}, requested {feature_type}, {clustering_method}, {distance_metric}")
    
    # Extract features from test images
    print(f"Extracting features for {len(image_paths)} test images using {feature_type}...")
    features, valid_image_paths = extract_features(image_paths, feature_type=feature_type)
    
    # Transform features using UMAP
    print("Transforming features with UMAP...")
    umap_embeddings = umap_model.transform(features)
    
    # Predict clusters
    print(f"Predicting clusters using {clustering_method}...")
    if clustering_method == "gmm":
        cluster_labels = cluster_model.predict(umap_embeddings)
    elif clustering_method in ["kmeans", "hierarchical"]:
        # For hierarchical, we approximate by using distances to centroids if KMeans-like prediction isn't available
        if hasattr(cluster_model, "predict"):
            cluster_labels = cluster_model.predict(umap_embeddings)
        else:
            raise ValueError("Hierarchical clustering model does not support direct prediction")
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "dataset": test_df['dataset'].iloc[:len(valid_image_paths)].tolist(),
        "img_path": valid_image_paths,
        "labelCluster": cluster_labels
    })
    
    # Save results using the same directory structure
    base_dir = os.path.dirname(os.path.dirname(models_dir))  # Go up from models/ to base directory
    base_dir = os.path.dirname(os.path.dirname(base_dir))
    results_dir = os.path.join(base_dir, "inference_results")
    os.makedirs(results_dir, exist_ok=True)
    

    output_file = os.path.join(results_dir, f"inference_{task}_{test_set}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Inference results saved to {output_file}")

def main():
    # Example parameters
    task = "TIL" #"tumor"

    if task.lower() == 'tumor':
        csv_path = "/home/michael/CAMTEL/dataset/tumor_dataset.csv"
        test_sets = ['pannuke', 'nucls', 'ocelot'] # 'pannuke', 'nucls', 'ocelot'
    elif task.lower() == 'til':
        csv_path = "/home/michael/CAMTEL/dataset/TIL_dataset.csv"
        test_sets = ['lizard', 'nucls', 'cptac', 'austin']
    else:
        raise ValueError("Task must be 'tumor' or 'TIL'")
    
    # test_sets = ['lizard', 'nucls', 'cptac', 'austin'] #['pannuke', 'nucls', 'ocelot']
    clustering_methods = ['kmeans']
    feature_types = ['lab']
    distance_metrics = ['cosine']
    
    for test_set in test_sets:
        for clustering_method in clustering_methods:
            for feature_type in feature_types:
                for distance_metric in distance_metrics:
                    print(f"\nInferring clusters for {task}, test set: {test_set}, "
                          f"clustering: {clustering_method}, feature: {feature_type}, "
                          f"distance: {distance_metric}")
                    try:
                        infer_clusters(
                            task=task,
                            test_set=test_set,
                            clustering_method=clustering_method,
                            feature_type=feature_type,
                            distance_metric=distance_metric,
                            csv_path=csv_path
                        )
                    except Exception as e:
                        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
