import pandas as pd

def img_to_path(dataset, img_name):
    if dataset == "ocelot":
        return f"/nfs/datasets/public/ProcessedHistology/Ocelot/inputs/original/{img_name}.tif"
    elif dataset == "lizard":
        return f"/nfs/datasets/public/ProcessedHistology/Lizard/inputs/original/{img_name}.tif"
    elif dataset == "pannuke":
        return f"/nfs/datasets/public/ProcessedHistology/PanNuke/inputs/original/{img_name}.tif"
    elif dataset == "nucls":
        return f"/nfs/datasets/public/ProcessedHistology/NuCLS/inputs/original/{img_name}.tif"
    elif dataset == "cptacCoad":
        return f"/nfs/datasets/public/ProcessedHistology/CPTAC-COAD/inputs/original/{img_name}.tif"
    else:
        return f"/nfs/datasets/public/ProcessedHistology/TCGA-BRCA/inputs/original/{img_name}.tif"

df1 = pd.read_csv("./output/clustering_result_tumor_ocelot.csv")
df1.columns = ["dataset", "img_path", "l_mean", "l_std", "a_mean", "a_std", "b_mean", "b_std", "labelCluster"]

df2 = pd.read_csv("../dataset/full_dataset.csv",names=["dataset", "img_name", "centerX", "centerY", "labelTIL", "labelTumor"])
df2['img_path'] = df2.apply(lambda row: img_to_path(row['dataset'], row['img_name']), axis = 1)


# Merge the two dataframes on 'dataset' and 'img_name'/'Image Path'
df_merged = pd.merge(df1, df2, left_on=['dataset', 'img_path'], right_on=['dataset', 'img_path'])
df_merged_filtered = df_merged[["dataset", "img_name", "centerX", "centerY", "labelTIL", "labelTumor","labelCluster"]]

# print(df_merged.columns)
# print(df_merged_filtered.columns)
# df_merged_filtered.to_csv("../dataset/full_dataset_with_cluster.csv", index=None)


# # Calculate the percentage of labelTIL and labelTumor having value of 1 in each cluster
# cluster_percentage = df_merged.groupby(['dataset','labelCluster']).agg({
#     'labelTIL': lambda x: (x.sum() / len(x)) * 100,
#     'labelTumor': lambda x: (x.sum() / len(x)) * 100
# }).reset_index()

# # Print the final result into a text file
# with open('./analysis_outputs/cluster_percentage.txt', 'w') as file:
#     file.write(cluster_percentage.to_string(index=False))

# print(cluster_percentage)