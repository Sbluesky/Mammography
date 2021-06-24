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
from models import MammoModel
from dataset import MultiClassMammo
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")
#%%
#%%
holdoutdf = pd.read_csv("/media/tungthanhlee/DATA/mammo/sam/csv/singleview-holdout.csv")
#path of image
holdoutdf["path"] = "/media/tungthanhlee/DATA/mammo/sam/singleview_sam/dataset/crop-images/" + holdoutdf["study_id"] + "/" + holdoutdf["image_id"] + ".png"
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
holdout_dl = MultiClassMammo(holdoutdf[(holdoutdf["label_birad"] == 2) | (holdoutdf["label_birad"] == 3)], transform1 = tfms1,transform2 = tfms2)
# %%
holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = True, batch_size =  4, num_workers = 3)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
#Setting model and moving to device
model_mammo = MammoModel(True).to('cuda')
model_mammo.load_state_dict(torch.load('/media/tungthanhlee/DATA/mammo/sam/singleview_sam/Birad23/models/class23-top-fz.pt'))
criterion_multioutput = nn.MSELoss()


# %%
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


#%%
bi_list = []
den_list = []
bi_pred_list = []
den_pred_list = []

bi_correct = 0
den_correct = 0
total_correct  = 0
bi_TP, den_TP, bi_TN, den_TN, bi_FP, den_FP, bi_FN, den_FN = 0,0,0,0,0,0,0,0
TP, TN, FP, FN = 0,0,0,0
x = 2.33
def convert_classify_bi(bi):
    result = []
    for i in range(len(bi)):
        if (bi[i] <=x):
            result.append(2)
        else:
            result.append(3)
    return torch.as_tensor(result)
y = 0.5
def convert_classify_den(den):
    result = []
    for i in range(len(den)):
        if den[i] <= y:
            result.append(0)
        elif (den[i] > y) and (den[i] <= 1+y):
            result.append(1)
        elif (den[i] > 1+y) and (den[i] <= 2+y):
            result.append(2)
        elif (den[i] > 2+y):
            result.append(3)
    return torch.as_tensor(result)

print('hi')
for batch_idx, sample_batched in enumerate(tqdm(holdout_dataloader)):
    image, bi, den = sample_batched['image'].to(device), sample_batched['bi'].to(device), sample_batched['den'].to(device)
    bi, den = bi.type(torch.float32), den.type(torch.float32)
    bi_hat, den_hat = model_mammo(image)   
    bi_hat = bi_hat.to(device)
    den_hat = den_hat.to(device)
    bi_hat, den_hat = bi_hat.type(torch.float32), den_hat.type(torch.float32)
    #bi_hat = torch.reshape(bi_hat*6, (-1,)) #di qua activate sigmod tra ve value 0-1, *6 vi co 6 class 
    #den_hat = torch.reshape(den_hat*4, (-1,))
    bi_hat = torch.reshape(bi_hat, (-1,))
    den_hat = torch.reshape(den_hat, (-1,))
    """
    print("bi true ", bi)
    print("bi pred ", bi_hat)
    print("den true ", den)
    print("den pred ", den_hat)
    """
    #print(bi_hat)
    #print(bi)
    bi_hat = convert_classify_bi(bi_hat).to(device)
    den_hat = convert_classify_den(den_hat).to(device)
    bi_list.append(bi)
    den_list.append(den)
    bi_pred_list.append(bi_hat)
    den_pred_list.append(den_hat)

    #Accuracy
    bi_correct += (bi_hat == bi).sum() 
    den_correct += (den_hat == den).sum()
    tmp = ((bi_hat==bi)*(den_hat==den)).sum()
    total_correct += tmp   


#%%
bi_list = torch.cat(bi_list, 0)
den_list = torch.cat(den_list, 0)
bi_pred_list = torch.cat(bi_pred_list, 0)
den_pred_list = torch.cat(den_pred_list, 0)

#%%
print("Accuracy of birad:", bi_correct/len(holdout_dl))
print("Accuracy of density:", den_correct/len(holdout_dl))
print("Total Accuracy", total_correct/len(holdout_dl))
print("Classification report of each label:")

print("BIRAD:")
print(classification_report(bi_list.cpu(), bi_pred_list.cpu()))
print(confusion_matrix(bi_list.cpu(), bi_pred_list.cpu()))
print("F1 micro:", f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'micro'))
print("F1 macro:", f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'macro'))
print("Quadratic-weighted-kappa of Birad: ",cohen_kappa_score(bi_list.cpu(), bi_pred_list.cpu(), weights="quadratic"))

print("DENSITY:")
print(classification_report(den_list.cpu(), den_pred_list.cpu()))
print(confusion_matrix(den_list.cpu(), den_pred_list.cpu()))
print("F1 micro:", f1_score(den_list.cpu(), den_pred_list.cpu(), average = 'micro'))
print("F1 macro:", f1_score(den_list.cpu(), den_pred_list.cpu(), average = 'macro'))
print("Quadratic-weighted-kappa of Density: ",cohen_kappa_score(den_list.cpu(), den_pred_list.cpu(), weights="quadratic"))

#%%