#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
import cv2          #For transforming image
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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from efficientnet_pytorch import EfficientNet
from models import MammoModel
from datasets import MultiClassMammo
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")
#%%

#Crop
df = pd.read_csv("/home/single2/mammo/mammo/data/updatedata/csv/singleview-holdout.csv")
df = df[df["label_birad"] != 0 ]
df.reset_index(drop = True, inplace = True)
#path of image
df["path"] = "/home/single2/mammo/mammo/data/updatedata/crop-images/" + df["study_id"] + "/" + df["image_id"] + ".png"

# %%
#transform images size from (3518, 2800, 3) to (1759,1400,3)
tfms = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
# %%
#fold: ['train', 'valid', 'holdout']
holdout_dl = MultiClassMammo(df, transform = tfms)
# %%

holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = True, batch_size =  2, num_workers = 3)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
#Setting model and moving to device

model_mammo = MammoModel(True).to(device)


# %%
model_mammo.load_state_dict(torch.load('/home/single2/mammo/mammo/sam/singleview_sam/SE-RESNET50/models/top.pt'))


# %%
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


#%%



def convert_classify_bi(bi, x):
    result = []
    for i in range(len(bi)):
        if bi[i] <=1+x:
            result.append(1)
        elif (bi[i] > 1+x) and (bi[i] <=2+x):
            result.append(2)
        elif (bi[i] > 2 +x) and (bi[i] <=3+x):
            result.append(3)
        elif (bi[i] > 3+x) and (bi[i] <=4+x):
            result.append(4)
        elif (bi[i] > 4+x):
            result.append(5)
    return torch.as_tensor(result)
def convert_classify_den(den,y):
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

for i in range (1 , 10, 2):
    bi_list = []
    den_list = []
    bi_pred_list = []
    den_pred_list = []

    bi_correct = 0
    den_correct = 0
    total_correct  = 0
    bi_TP, den_TP, bi_TN, den_TN, bi_FP, den_FP, bi_FN, den_FN = 0,0,0,0,0,0,0,0
    TP, TN, FP, FN = 0,0,0,0
    print("ResNext-50 number " + str(i))
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
        
        #print("bi true ", bi)
        #print("bi pred ", bi_hat)
        """
        print("den true ", den)
        print("den pred ", den_hat)
        """
        x= 0.5 + i/100
        y= 0.5 + i/100
        bi_hat = convert_classify_bi(bi_hat,x).to(device)
        den_hat = convert_classify_den(den_hat,y).to(device)
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
    print("F1 micro:", f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'micro', zero_division=1))
    print("F1 macro:", f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'macro', zero_division=1))
    print("Quadratic-weighted-kappa of Birad: ",cohen_kappa_score(bi_list.cpu(), bi_pred_list.cpu(), weights="quadratic"))

    print("DENSITY:")
    print(classification_report(den_list.cpu(), den_pred_list.cpu()))
    print(confusion_matrix(den_list.cpu(), den_pred_list.cpu()))
    print("F1 micro:", f1_score(den_list.cpu(), den_pred_list.cpu(), average = 'micro', zero_division=1))
    print("F1 macro:", f1_score(den_list.cpu(), den_pred_list.cpu(), average = 'macro', zero_division=1))
    print("Quadratic-weighted-kappa of Density: ",cohen_kappa_score(den_list.cpu(), den_pred_list.cpu(), weights="quadratic"))
#%%