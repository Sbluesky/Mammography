#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
import albumentations as A
import matplotlib.pyplot as plt  #For representation#For model building
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models, utils
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from torch.nn import functional as F
from skimage import io, transform
from torch.optim import lr_scheduler
from skimage.transform import AffineTransform, warp
from tqdm import tqdm
from models import StackingModel
from dataset import MultiClassMammo
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")
#%%
#%%
holdoutdf = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_holdout.csv")
# %%
#transform images size from (3518, 2800, 3) to (1759,1400,3)
tfms1 = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor()])

# %%
#fold: ['train', 'valid', 'holdout']
holdout_dl = MultiClassMammo(holdoutdf[(holdoutdf["L_birad_max"]!=0) & (holdoutdf["R_birad_max"]!=0) ], transform1 = tfms1,transform2 = None)
# %%
holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = True, batch_size =  8, num_workers = 3)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
#Setting model and moving to device
model_mammo = StackingModel(True).to('cuda')
model_mammo.eval()

#%%
#Create dataset
Ldf = pd.DataFrame(columns=["Bi_CC", "Bi_MLO", "Den_CC", "Den_MLO", "Birad", "Density"])
Rdf = pd.DataFrame(columns=["Bi_CC", "Bi_MLO", "Den_CC", "Den_MLO", "Birad", "Density"])

# %%
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

for batch_idx, sample_batched in enumerate(tqdm(holdout_dataloader)):
    L_CC_img, R_CC_img, L_MLO_img, R_MLO_img = \
                             sample_batched['L_CC_img'].to(device),\
                             sample_batched['R_CC_img'].to(device),\
                             sample_batched['L_MLO_img'].to(device),\
                             sample_batched['R_MLO_img'].to(device)

    L_birad, R_birad, L_density, R_density = \
                             sample_batched['L_birad'].to(device),\
                             sample_batched['R_birad'].to(device),\
                             sample_batched['L_density'].to(device),\
                             sample_batched['R_density'].to(device)

    #L_birad, R_birad, L_density, R_density = L_birad.type(torch.float32), R_birad.type(torch.float32), L_density.type(torch.float32), R_density.type(torch.float32)
            #model predict 

    output  = model_mammo(L_CC_img, R_CC_img, L_MLO_img, R_MLO_img)
            #max label
            
    #Left
    LCCbi = torch.reshape(output["LCCbi"].to(device),(-1,))
    LCCden = torch.reshape(output["LCCden"].to(device),(-1,))
    LMLObi = torch.reshape(output["LMLObi"].to(device),(-1,))
    LMLOden = torch.reshape(output["LMLOden"].to(device),(-1,))
    #Right
    RCCbi = torch.reshape(output["RCCbi"].to(device),(-1,))
    RCCden = torch.reshape(output["RCCden"].to(device),(-1,))
    RMLObi = torch.reshape(output["RMLObi"].to(device),(-1,))
    RMLOden = torch.reshape(output["RMLOden"].to(device),(-1,))
    
    temp = pd.DataFrame(columns=["Bi_CC", "Bi_MLO", "Den_CC", "Den_MLO", "Birad", "Density"])
    temp["Bi_CC"] = LCCbi.tolist()
    temp["Bi_MLO"] = LMLObi.tolist()
    temp["Den_CC"] = LCCden.tolist()
    temp["Den_MLO"] = LMLOden.tolist()
    temp["Birad"] = L_birad.tolist()
    temp["Density"] = L_density.tolist()
    Ldf = pd.concat((Ldf,temp), axis = 0)
    temp = pd.DataFrame(columns=["Bi_CC", "Bi_MLO", "Den_CC", "Den_MLO", "Birad", "Density"])
    temp["Bi_CC"] = RCCbi.tolist()
    temp["Bi_MLO"] = RMLObi.tolist()
    temp["Den_CC"] = RCCden.tolist()
    temp["Den_MLO"] = RMLOden.tolist()
    temp["Birad"] = R_birad.tolist()
    temp["Density"] = R_density.tolist()
    Rdf = pd.concat((Rdf,temp), axis = 0)

Ldf.info()
Rdf.info()
Ldf.to_csv("/home/single4/mammo/mammo/data/updatedata/csv/Left_LGBMdataset_holdout.csv", index = False)
Rdf.to_csv("/home/single4/mammo/mammo/data/updatedata/csv/Right_LGBMdataset_holdout.csv", index = False)


