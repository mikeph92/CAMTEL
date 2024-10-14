import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# List of file paths for the 16 images
file_paths = [
    './analysis_outputs/1-372_0_0_0_256_256.tif',
    './analysis_outputs/3-366_0_0_0_256_256.tif',
    './analysis_outputs/TCGA-GM-A2DH-DX1_0_67941_72004_68257_72320.tif',
    './analysis_outputs/TCGA-A7-A0DA-DX1_0_54329_24518_54643_24832.tif',
    './analysis_outputs/3-2660_0_0_0_256_256.tif',
    './analysis_outputs/2-2456_0_0_0_256_256.tif',
    './analysis_outputs/consep_6_0_0_500_500.tif',
    './analysis_outputs/2-1309_0_0_0_256_256.tif',
    './analysis_outputs/TCGA-BH-A0B3-DX1_0_107308_53366_107626_53684.tif',
    './analysis_outputs/1-1377_0_0_0_256_256.tif',
    './analysis_outputs/TCGA-A7-A6VV-DX1_0_62274_47190_62589_47506.tif',
    './analysis_outputs/179_0_49214_38961_50238_39985.tif'
]

# Read images from file paths
images = [Image.open(file_path) for file_path in file_paths]

# Number of groups
num_groups = 6

# Create a figure with subplots
fig, axes = plt.subplots(num_groups, 2, figsize=(10, 20))

# Plot each image in the corresponding subplot
for i in range(num_groups):
    for j in range(2):
        index = i * 2 + j
        axes[i, j].imshow(images[index], cmap='gray')
        axes[i, j].axis('off')
        axes[i, j].set_title(f'Group {i+1}, Image {j+1}')

# Adjust layout
plt.tight_layout()

# Save the plot to a PNG file
plt.savefig('./analysis_outputs/output_plot.png')
