import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


filenames = ["1-1119_0_0_0_256_256", 
             "1-1235_0_0_0_256_256",
             "3-1906_0_0_0_256_256",
             "2-1088_0_0_0_256_256",
             "1-0_0_0_0_256_256"]

for filename in filenames:
    # Create a list of image file paths
    image_paths = [
        f'/nfs/users/michael/augmented/0/PanNuke/inputs/original/{filename}.tif',
        f'/nfs/users/michael/augmented/1/PanNuke/inputs/original/{filename}.tif',
        f'/nfs/users/michael/augmented/2/PanNuke/inputs/original/{filename}.tif',
        f'/nfs/users/michael/augmented/3/PanNuke/inputs/original/{filename}.tif',
        f'/nfs/users/michael/augmented/4/PanNuke/inputs/original/{filename}.tif',
        f'/nfs/users/michael/augmented/5/PanNuke/inputs/original/{filename}.tif',
        f'/nfs/users/michael/augmented/full/PanNuke/inputs/original/{filename}.tif',
        f'/nfs/users/michael/histoPath/PanNuke/inputs/original/{filename}.tif',
        f'/nfs/datasets/public/ProcessedHistology/PanNuke/inputs/original/{filename}.tif'
    '']

    # Create a figure with 2 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Loop through the image paths and plot each image
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        ax = axes[i // 3, i % 3]
        ax.imshow(np.array(img))
        ax.axis('off')  # Hide the axis

    # Display the plot
    plt.savefig(f'show_aug_{filename}.png')