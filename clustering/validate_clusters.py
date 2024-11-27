import matplotlib.pyplot as plt
from clustering import *
import numpy as np
import os
from PIL import Image

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a.astype(np.float32) - b.astype(np.float32))

# Load image data and perform clustering
image_data, image_paths = read_from_csv("tumor", "ocelot")
labels, reduced_data, centroids = cluster_images(image_data, n_clusters=6)

# Combine reduced_data with image paths for easy access
reduced_data = np.hstack((reduced_data, image_paths))

# Find the closest points to each centroid
closest_points = {}
num_closest = 1  # Number of closest points to find

for i, centroid in enumerate(centroids):
    cluster_points = reduced_data[labels == i][:, :-2]
    distances = np.array([euclidean_distance(point, centroid) for point in cluster_points])
    closest_indices = distances.argsort()[:num_closest]
    closest_points[i] = reduced_data[labels == i][closest_indices]

# Destination folder to save the plot
dest_folder = "./output/"
os.makedirs(dest_folder, exist_ok=True)

# Set up the plotting grid for all clusters in a single figure
n_clusters = len(centroids)
fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5))

for label, points in closest_points.items():
    for j, point in enumerate(points):
        img_path = point[-1]
        img = Image.open(img_path)
        
        # Plot image in the appropriate subplot
        ax = axes[label]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Cluster {label}')

# Save the combined plot as a single image
plt.tight_layout()
plt.savefig(os.path.join(dest_folder, "all_clusters_closest_images.png"))
plt.close()
