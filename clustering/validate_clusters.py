from clustering import *
import numpy as np
import os


# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a.astype(np.float32) - b.astype(np.float32))

image_data, image_paths = read_from_csv("./output/clustering_results.csv")
labels, reduced_data, centroids = cluster_images(image_data, n_clusters=6)


# image_paths_2d = np.reshape(image_paths, (len(image_paths), 1))
reduced_data = np.hstack((reduced_data, image_paths))

# Find the closest points to each centroid
closest_points = {}
num_closest = 2  # Number of closest points to find

# print(image_data[0])
print(reduced_data[[0,1]][:, :-2])

for i, centroid in enumerate(centroids):
    distances = np.array([euclidean_distance(point, centroid) for point in reduced_data[labels == i][:,:-2]])
    closest_indices = distances.argsort()[:num_closest]
    closest_points[i] = reduced_data[labels == i][closest_indices]

dest_folder = "./analysis_outputs/"
for label, points in closest_points.items():
    print(label)
    for point in points:
        os.system(f'cp "{point[-1]}" "{dest_folder}"')
        print(point[-1])
