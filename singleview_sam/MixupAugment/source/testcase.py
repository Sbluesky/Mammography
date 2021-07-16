""" This file use for testing"""
#%%
from dataset import Mixup

#test case
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
#get image in birad 1
birad_df = pd.read_csv("/home/single3/mammo/mammo/data/csv/newsingletrain.csv")
birad_df = birad_df[(birad_df["birad"] == "BI-RADS 1") & (birad_df["position"] == "MLO")][:10]
mask_df = pd.read_csv("/home/single3/mammo/mammo/data/csv/mass-newcrop-train.csv")
#print(mask_df['image_id'].value_counts())
#mask_df = mask_df[(mask_df["image_id"] == "1.3.12.2.1107.5.12.7.5054.30000019071500223231200000424")]
mask_df = mask_df[(mask_df["view_position"] == "MLO")][:10]
sample = Mixup(birad_df,mask_df, 1024, 768)
print(sample[3]["birad"])
plt.imshow(torchvision.utils.make_grid(sample[3]['image']).permute(1, 2, 0))
#%%
"""
index = 5
info_img = birad_df.iloc[index]
imagespath = createpath(info_img["path"])
image = Image.open(imagespath).convert('RGB')
laterality = info_img['laterality']

            #get mask
info_mask = mask_df.iloc[index]
maskpath = createmaskpath(info_mask["study_id"], info_mask["image_id"])           
image_mask = Image.open(maskpath).convert('RGB')
laterality_m = info_mask['laterality']
wim, him = image_mask.size
tfms = get_transform(1024,768)


             #create mask
maskarr = np.zeros((him, wim), dtype = np.uint8) 

if laterality == "R":
    image = transforms.functional.hflip(image)
image = tfms(image)
alpha = np.random.uniform(0.1, 0.4)
image = image * alpha

# transform mask image
if laterality_m == "R":
    image_mask = transforms.functional.hflip(image_mask)
tmfs2 = transforms.ToTensor()
image_mask = tmfs2(image_mask)
x_min, x_max = int(info_mask["x_min"].round()), int(info_mask["x_max"].round())
y_min, y_max = int(info_mask["y_min"].round()), int(info_mask["y_max"].round())
#get bouding box
maskarr[y_min:y_max, x_min:x_max] = 255
maskarr = tmfs2(maskarr)
maskarr = maskarr.expand(3, -1, -1) #expand 3 channels
image_mask = image_mask * maskarr
tfms3 = transforms.Resize((1024,768))
image_mask = tfms3(image_mask)

index = torch.nonzero(image_mask)
for i in index:
    image[i[0], i[1], i[2]] = image_mask[i[0], i[1], i[2]]


plt.imshow(torchvision.utils.make_grid(image).permute(1, 2, 0))
#plt.imshow(torchvision.utils.make_grid(image_mask).permute(1, 2, 0))
#plt.imshow(torchvision.utils.make_grid(maskarr).permute(1, 2, 0))
"""
# %%
