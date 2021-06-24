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
from models import ViewWiseModel
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
holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = True, batch_size =  4, num_workers = 3)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
#Setting model and moving to device
model_mammo = ViewWiseModel(True).to('cuda')
model_mammo.load_state_dict(torch.load('/home/single4/mammo/mammo/sam/multiview/modeltruelabel/viewwise-undersample-last.pt'))
criterion_multioutput = nn.MSELoss()


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
def convert_classify_bi(bi, x=0):
    result = []
    for i in range(len(bi)):
        if bi[i] <=1.7:
            result.append(1)
        elif (bi[i] > 1.7) and (bi[i] <=2.4):
            result.append(2)
        elif (bi[i] > 2.4) and (bi[i] <=3.15):
            result.append(3)
        elif (bi[i] > 3.15) and (bi[i] <=4.4):
            result.append(4)
        elif (bi[i] > 4.4):
            result.append(5)
    return torch.as_tensor(result)

def convert_classify_bi_R(bi, x=0.1):
    result = []
    for i in range(len(bi)):
        if bi[i] <1.9:
            result.append(1)
        elif (bi[i] >= 1.9) and (bi[i] <=2.4):
            result.append(2)
        elif (bi[i] > 2.4) and (bi[i] <=3.2):
            result.append(3)
        elif (bi[i] > 3.2) and (bi[i] <=4.3):
            result.append(4)
        elif (bi[i] > 4.3):
            result.append(5)
    return torch.as_tensor(result)

def convert_classify_den(den, y=0.7):
    result = []
    for i in range(len(den)):
        if den[i] <= y+0.2:
            result.append(0)
        elif (den[i] > y+0.2) and (den[i] <= 1+y-0.2):
            result.append(1)
        elif (den[i] > 1+y-0.2) and (den[i] <= 2+y-0.3455):
            result.append(2)
        elif (den[i] > 2+y-0.3455):
            result.append(3)
    return torch.as_tensor(result)

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

    L_birad, R_birad, L_density, R_density = L_birad.type(torch.float32), R_birad.type(torch.float32), L_density.type(torch.float32), R_density.type(torch.float32)
            #model predict
    output = model_mammo(L_CC_img, R_CC_img, L_MLO_img, R_MLO_img)
            #max label
    L_bi_hat = torch.reshape(output["L_bi"].type(torch.float32).to(device),(-1,))
    L_den_hat = torch.reshape(output["L_den"].type(torch.float32).to(device),(-1,))
    R_bi_hat = torch.reshape(output["R_bi"].type(torch.float32).to(device),(-1,))
    R_den_hat = torch.reshape(output["R_den"].type(torch.float32).to(device),(-1,))
    
    
    print("L bi true ", L_birad)
    print("L bi pred ", L_bi_hat)
    print("R bi true ", R_birad)
    print("R bi pred ", R_bi_hat)
    
    L_bi_hat = convert_classify_bi(L_bi_hat).to(device)
    L_den_hat = convert_classify_den(L_den_hat).to(device)
    R_bi_hat = convert_classify_bi_R(R_bi_hat).to(device)
    R_den_hat = convert_classify_den(R_den_hat).to(device)
    print("R bi pred convert", R_bi_hat)

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

#%%