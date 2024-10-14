from PIL import Image

import pandas as pd


def img_to_path(dataset, img_name):
    if dataset == "ocelot":
        return f"/home/michael/data/ProcessedHistology/Ocelot/inputs/original/{img_name}.tif"
    elif dataset == "lizard":
        return f"/home/michael/data/ProcessedHistology/Lizard/inputs/original/{img_name}.tif"
    elif dataset == "pannuke":
        return f"/home/michael/data/ProcessedHistology/PanNuke/inputs/original/{img_name}.tif"
    elif dataset == "nucls":
        return f"/home/michael/data/ProcessedHistology/NuCLS/inputs/original/{img_name}.tif"
    elif dataset == "cptacCoad":
        return f"/home/michael/data/ProcessedHistology/CPTAC-COAD/inputs/original/{img_name}.tif"
    else:
        return f"/home/michael/data/ProcessedHistology/TCGA-BRCA/inputs/original/{img_name}.tif"
    
IMG_SIZE = 96

# tils = [("nucls","TCGA-A7-A6VW-DX1_0_85475_21184_85788_21498", 238, 141),
#         ("lizard", "glas_56_0_0_775_522",  500,94),
#         ("cptacCoad","05CO014-9bba0a92-9eb5-42ba-8c0f-865141_20094_19621_20832_20146", 502, 293),
#         ("tcgaBrca", "TCGA-E2-A14P-01Z-00-DX1.663B02FF-C64B-41A6-8685-FD61CD76F9C6_0_40393_17721_42197_19047", 888, 1308)]

# tumors = [("ocelot", "380_0_14340_30729_15364_31753", 1019,870),
#           ("pannuke", "3-2402_0_0_0_256_256", 115, 6),
#           ("nucls", "TCGA-S3-AA10-DX1_0_44831_24498_45087_24754", 230,91)]

df = pd.read_csv("dataset/full_dataset.csv")
df_til = df[df.labelTIL == 1].reset_index()
df_tumor = df[df.labelTumor == 1].reset_index()

tils = df_til.sample(n=10)
tumors= df_tumor.sample(n=10)

# for i, img in tils.iterrows():
#     img_path = img_to_path(img['dataset'], img['img_name'])
#     full_img = Image.open(img_path)
#     crop_box = (img['centerX']-IMG_SIZE/2,img['centerY']-IMG_SIZE/2, img['centerX']+IMG_SIZE/2, img['centerY']+IMG_SIZE/2)
#     image = full_img.crop(crop_box)
#     image.save(f"testimg/til/{img['dataset']}_{img['img_name']}.png")
    
for i, img in tumors.iterrows():
    img_path = img_to_path(img['dataset'], img['img_name'])
    full_img = Image.open(img_path)
    crop_box = (img['centerX']-IMG_SIZE/2,img['centerY']-IMG_SIZE/2, img['centerX']+IMG_SIZE/2, img['centerY']+IMG_SIZE/2)
    image = full_img.crop(crop_box)
    image.save(f"testimg/tumor/{img['dataset']}_{img['img_name']}.png")