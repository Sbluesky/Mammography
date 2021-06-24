"""THIS FILE USES TO EDA THE DATASET OF MAMMOGRAPHY (BOUNDING BOX)"""
#%%
import numpy as np
import pandas as pd
from collections import Counter

traindf = pd.read_csv("/home/sam/Mammography/code/data/box-birads-extratrain.csv")
traindf.info()
traindf.head(5)

#%%
globaldf = traindf[traindf["type"]=="global"]
localdf = traindf[traindf["type"]=="local"] #list img có tổn thương
globaldf.info()
localdf.info()
# %%
#check xem birad 1 có bounding box hay không?
list_im_gl = globaldf["image_id"].unique()
list_im_lc = localdf["image_id"].unique()

img_birad1 = globaldf[globaldf["birads"]=="BI-RADS 1"] #16055
img_birad2 = globaldf[globaldf["birads"]=="BI-RADS 2"] #5541
img_birad3 = globaldf[globaldf["birads"]=="BI-RADS 3"] #1754
img_birad4 = globaldf[globaldf["birads"]=="BI-RADS 4"] #1899
img_birad5 = globaldf[globaldf["birads"]=="BI-RADS 5"] #692

# %%
lession_in_B5 = []
for im in list_im_lc:
    if im in img_birad5["image_id"].values:
        lession_in_B5.append(im)

print("birad 2 co bounding box: ", len(lession_in_B5) )
# %%
#lession_in_B1: 950, lession_in_B2: 1716, lession_in_B3: 1560, lession_in_B4: 1816, lession_in_B5: 671
print(lesion_in_B1)
# %%
traindf[traindf["image_id"]=="1.3.12.2.1107.5.12.7.5054.30000019060500321132800001110"]
# %%
traindf["box_label"].unique()
# %%
#check trong từng birad có những tổn thương nào
les_bi1 = []
for im in lesion_in_B5:
    les_bi1.append(localdf[localdf["image_id"] == im]["box_label"].values[0])
print(Counter(les_bi1))

"""_______________________________________"""

# %%
#FILE VALID
vdf = pd.read_csv("/home/sam/Mammography/code/data/singleview-holdout.csv")
vboxdf = pd.read_csv("/home/sam/Mammography/code/data/box-birads-holdout.csv")

vdf.info()
vboxdf.info()

# %%
#check img missed
miss_im = []
l_imbox = vboxdf["image_id"].unique()
for im in vdf["image_id"].unique() :
    if im not in l_imbox:
        miss_im.append(im)

len(miss_im) #5 IMGS

#%%
#check xem các imgs có bounding box (local) có trong img global hay không
vboxdf_gl = vboxdf[vboxdf["type"] == "global"]
vboxdf_lc = vboxdf[vboxdf["type"] == "local"]
print(vboxdf_gl["image_id"].nunique()) #train: 25941
print(vboxdf_lc["image_id"].nunique()) #train 6747
img_n_bi = []
for im in vboxdf_lc["image_id"].unique():
    if im not in vboxdf_gl["image_id"].unique():
        img_n_bi.append(im)

print(img_n_bi) #9imgs
len(img_n_bi)

#%%
len(vboxdf_lc["image_id"])
# %%
#check các imbirad khác nhau
Err_Bi = []
for im in l_imbox:
    if im not in img_n_bi:
        if vdf[vdf["image_id"] == im]["birad"].values[0] != vboxdf_gl[vboxdf_gl["image_id"]==im]["birads"].values[0]:
            Err_Bi.append(im)

print(Err_Bi)
print(len(Err_Bi))
print(vdf[vdf["image_id"] == Err_Bi[0]])
print(vboxdf[vboxdf["image_id"] == Err_Bi[0]])


# %%
img_birad1 = vboxdf_gl[vboxdf_gl["birads"]=="BI-RADS 1"] #3448
img_birad2 = vboxdf_gl[vboxdf_gl["birads"]=="BI-RADS 2"] #1132
img_birad3 = vboxdf_gl[vboxdf_gl["birads"]=="BI-RADS 3"] #324
img_birad4 = vboxdf_gl[vboxdf_gl["birads"]=="BI-RADS 4"] #138
img_birad5 = vboxdf_gl[vboxdf_gl["birads"]=="BI-RADS 5"] #26

# %%
def getnumbox(list_im_lc, im_birad):
    lesion_in_B = []
    for im in list_im_lc:
        if im in im_birad["image_id"].values:
            lesion_in_B.append(im)

    print("birad co bounding box: ", len(lesion_in_B) )
    return lesion_in_B

list_im_lc = vboxdf[vboxdf["type"]=="local"]["image_id"].unique() #values
lesion_in_B1 = getnumbox(list_im_lc, img_birad1) #train: 1152
lesion_in_B2 = getnumbox(list_im_lc, img_birad2) #train: 3670
lesion_in_B3 = getnumbox(list_im_lc, img_birad3) #train: 2585
lesion_in_B4 = getnumbox(list_im_lc, img_birad4) #train: 3953
lesion_in_B5 = getnumbox(list_im_lc, img_birad5) #train: 2052

# %%
#check lesion types in each birad
def checkNles(lesion_in_B, localdf):
    les_bi = []
    for im in lesion_in_B:
        for lab in localdf[localdf["image_id"] == im]["box_label"].values:
            les_bi.append(lab)
    return les_bi

les_bi1 = checkNles(lesion_in_B1, vboxdf_lc)
les_bi2 = checkNles(lesion_in_B2, vboxdf_lc)
les_bi3 = checkNles(lesion_in_B3, vboxdf_lc)
les_bi4 = checkNles(lesion_in_B4, vboxdf_lc)
les_bi5 = checkNles(lesion_in_B5, vboxdf_lc)
print(len(les_bi1))
print(len(les_bi2))
print(len(les_bi3))
print(len(les_bi4))
print(len(les_bi5))

print(Counter(les_bi1))
print(Counter(les_bi2))
print(Counter(les_bi3))
print(Counter(les_bi4))
print(Counter(les_bi5))

# %%
#add number of lesion types into EDA table
edadf = pd.DataFrame(columns=vboxdf["box_label"].unique())
def addEDAtable(edadf, les_bi):
    rows = []
    for tp in edadf.columns:
        if tp in Counter(les_bi).keys():
            print(tp)
            rows.append(Counter(les_bi)[tp])
        else:
            rows.append(0)
    print(rows)
    edadf.loc[len(edadf.index)] = rows #add rows into df
    return edadf

edadf = addEDAtable(edadf,les_bi1)
edadf = addEDAtable(edadf,les_bi2)
edadf = addEDAtable(edadf,les_bi3)
edadf = addEDAtable(edadf,les_bi4)
edadf = addEDAtable(edadf,les_bi5)
edadf["Birads"] = ["BIRADS 1","BIRADS 2","BIRADS 3","BIRADS 4","BIRADS 5"]
edadf["Total boxs"] = [len(les_bi1),len(les_bi2),len(les_bi3),len(les_bi4),len(les_bi5)]
edadf["Img boxs"] = [len(lesion_in_B1), len(lesion_in_B2), len(lesion_in_B3), len(lesion_in_B4), len(lesion_in_B5)]
edadf["Total imgs"] = [img_birad1.shape[0], img_birad2.shape[0], img_birad3.shape[0], img_birad4.shape[0], img_birad5.shape[0]]

# %%
edadf["%"] = [np.round((264/3428)*100,2),np.round((709/1134)*100,2),np.round((508/356)*100,2),np.round((181/136)*100,2), np.round((68/30)*100,2)]
edadf = edadf.set_index('Birads').T

edadf
# %%
edadf
#check lesion types in each birad...
# %%
