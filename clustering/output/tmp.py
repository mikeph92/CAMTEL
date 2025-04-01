import pandas as pd
import os


file_path = ['clustering_result_tumor_ocelot.csv',
             'clustering_result_tumor_nucls.csv',
             'clustering_result_tumor_pannuke.csv']

for file in file_path:
    df = pd.read_csv(file, index_col=0)
    df['img_name'] = df['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df.drop(columns=['image_path'], inplace=True)
    df.to_csv(file, index=False)  # Save the changes back to the CSV file
