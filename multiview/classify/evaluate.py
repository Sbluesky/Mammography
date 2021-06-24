#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
import cv2          #For transforming image
import albumentations as A
import matplotlib.pyplot as plt  #For representation#For model building
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models, utils
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from skimage import io, transform
from torch.optim import lr_scheduler
from skimage.transform import AffineTransform, warp
from tqdm import tqdm
from models import ViewWiseModel, BreastWiseModel, ViewWiseModelv1, ImageWiseModel, ViewWiseModelv4
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

tfms2 = A.Compose([
    A.RandomResizedCrop(width=512, height=512),
    A.ShiftScaleRotate(shift_limit=0.0625),
    A.GridDistortion(num_steps = 5),
])
# %%
#fold: ['train', 'valid', 'holdout']
holdout_dl = MultiClassMammo(holdoutdf[(holdoutdf["L_birad_max"]!=0) & (holdoutdf["R_birad_max"]!=0) ], transform1 = tfms1,transform2 = None)
# %%
holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = False, batch_size =  1, num_workers = 1)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
#Setting model and moving to device
model_mammo = ViewWiseModel(True).to('cuda')
model_mammo.load_state_dict(torch.load('/home/single4/mammo/mammo/sam/multiview/classify/models/viewwise-top.pt'))


# %%
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


#%%
L_bi_list = []
R_bi_list = []
L_den_list = []
R_den_list = []
L_bi_pred_list = []
R_bi_pred_list = []
L_den_pred_list = []
R_den_pred_list = []

bi_correct = 0
den_correct = 0


print('Evaluate for multiview: ')
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

            #model predict 
    output = model_mammo(L_CC_img, R_CC_img, L_MLO_img, R_MLO_img)
            #max label        
    print("L_bi: ",output["L_bi"])
    print("L_bi_ TV : ",L_birad)
    L_bi_hat = torch.max(output["L_bi"],1)[1].to(device) + 1 #lúc training đã chuẩn hóa [1,2,3,4,5] thành [0,1,2,3,4]
    L_den_hat = torch.max(output["L_den"],1)[1].to(device) 
    R_bi_hat = torch.max(output["R_bi"],1)[1].to(device) + 1
    R_den_hat = torch.max(output["R_den"],1)[1].to(device) 
    

#L_birad, R_birad, L_density, R_density
    L_bi_list.append(L_birad)
    R_bi_list.append(R_birad)
    L_den_list.append(L_density)
    R_den_list.append(R_density)
    L_bi_pred_list.append(L_bi_hat)
    R_bi_pred_list.append(R_bi_hat)
    L_den_pred_list.append(L_den_hat)
    R_den_pred_list.append(R_den_hat)

    #Accuracy
    bi_correct += (L_bi_hat == L_birad).sum() + (R_bi_hat == R_birad).sum()
    den_correct += (L_den_hat == L_density).sum() + (R_den_hat == R_density).sum()
  


#%%
L_bi_list,R_bi_list = torch.cat(L_bi_list, 0),torch.cat(R_bi_list, 0)
L_den_list,R_den_list = torch.cat(L_den_list, 0), torch.cat(R_den_list, 0)
L_bi_pred_list,R_bi_pred_list = torch.cat(L_bi_pred_list, 0),torch.cat(R_bi_pred_list, 0)
L_den_pred_list, R_den_pred_list = torch.cat(L_den_pred_list, 0),torch.cat(R_den_pred_list, 0)

#%%
print("Accuracy of birad:", bi_correct/(len(holdout_dl)*2))
print("Accuracy of density:", den_correct/(len(holdout_dl)*2))
print("Classification report of each label:")

print("LEFT BIRAD:")
print(classification_report(L_bi_list.cpu(), L_bi_pred_list.cpu()))
print(confusion_matrix(L_bi_list.cpu(), L_bi_pred_list.cpu()))
print("F1 micro:", f1_score(L_bi_list.cpu(), L_bi_pred_list.cpu(), average = 'micro'))
print("F1 macro:", f1_score(L_bi_list.cpu(), L_bi_pred_list.cpu(), average = 'macro'))
print("Quadratic-weighted-kappa of Birad: ",cohen_kappa_score(L_bi_list.cpu(), L_bi_pred_list.cpu(), weights="quadratic"))

print("RIGHT BIRAD:")
print(classification_report(R_bi_list.cpu(), R_bi_pred_list.cpu()))
print(confusion_matrix(R_bi_list.cpu(), R_bi_pred_list.cpu()))
print("F1 micro:", f1_score(R_bi_list.cpu(), R_bi_pred_list.cpu(), average = 'micro'))
print("F1 macro:", f1_score(R_bi_list.cpu(), R_bi_pred_list.cpu(), average = 'macro'))
print("Quadratic-weighted-kappa of Birad: ",cohen_kappa_score(R_bi_list.cpu(), R_bi_pred_list.cpu(), weights="quadratic"))

print("LEFT DENSITY:")
print(classification_report(L_den_list.cpu(), L_den_pred_list.cpu()))
print(confusion_matrix(L_den_list.cpu(), L_den_pred_list.cpu()))
print("F1 micro:", f1_score(L_den_list.cpu(), L_den_pred_list.cpu(), average = 'micro'))
print("F1 macro:", f1_score(L_den_list.cpu(), L_den_pred_list.cpu(), average = 'macro'))
print("Quadratic-weighted-kappa of Density: ",cohen_kappa_score(L_den_list.cpu(), L_den_pred_list.cpu(), weights="quadratic"))

print("RIGHT DENSITY:")
print(classification_report(R_den_list.cpu(), R_den_pred_list.cpu()))
print(confusion_matrix(R_den_list.cpu(), R_den_pred_list.cpu()))
print("F1 micro:", f1_score(R_den_list.cpu(), R_den_pred_list.cpu(), average = 'micro'))
print("F1 macro:", f1_score(R_den_list.cpu(), R_den_pred_list.cpu(), average = 'macro'))
print("Quadratic-weighted-kappa of Density: ",cohen_kappa_score(R_den_list.cpu(), R_den_pred_list.cpu(), weights="quadratic"))

#%%'