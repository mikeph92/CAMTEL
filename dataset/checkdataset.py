import os
import pandas as pd

# Read the dataset
csv_file = 'TIL_dataset.csv'
df = pd.read_csv(csv_file)

# Filter rows where dataset is 'Austin'
austin_data = df[df['dataset'] == 'austin']

# Get unique values for img_path
unique_img_paths = austin_data['img_path'].unique()

# Extract file names from img_path
file_names = [os.path.basename(path) for path in unique_img_paths]
print(file_names.__len__())

# Folder to check for images
# folder_path = r"D:\Study\Year 2 - sem 2\CSE5TSB\experiments\ProcessedHistology\Austin-CRC\OneDrive_1_11-04-2025\regions"
folder_path = "/home/michael/data/ProcessedHistology/Austin/regions/"
# folder_path = "/home/michael/data/ProcessedHistology/CPTAC-COAD/regions/"

# Check if all images exist in the folder
missing_files = []
for file_name in file_names:
    if not os.path.exists(os.path.join(folder_path, file_name)):
        missing_files.append(file_name)

# Print results
if missing_files:
    print("The following files are missing:")
    for missing_file in missing_files:
        print(missing_file)
else:
    print("All files are present in the folder.")