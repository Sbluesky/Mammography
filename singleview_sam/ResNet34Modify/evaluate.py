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
from dataset import MultiLabelMammo
#%%
# Full image
# df = pd.read_csv("/home/dungnb/workspace/2021_projects/mammo_multiview_classification/mammo/huyen/csv_singleview.csv")
# #path of image
# df["path"] = "/home/dungnb/workspace/2021_projects/mammo_multiview_classification/dataset/images/" + df["study_id"] + "/" + df["image_id"] + ".png"

#Crop
df = pd.read_csv("/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/updatedcsv_singleview.csv")
df = df.drop([1146, 1307, 2442, 5710, 7562, 9377, 9382, 15660, 16328, 16348, 18523, 18840])
df.reset_index(drop = True, inplace = True)
#path of image
df["path"] = "/home/single1/BACKUP/SamHUyen/multi_view_mammo_classification/crop-images/crop_images/"  + df["image_id"] + ".png"

# %%
#transform images size from (3518, 2800, 3) to (1759,1400,3)
tfms = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
# %%
#fold: ['train', 'valid', 'holdout']
# train_dl = MultiLabelMammo(df[df["fold"]=="train"], transform = tfms) 
# val_dl = MultiLabelMammo(df[df["fold"]=="valid"], transform = tfms)
holdout_dl = MultiLabelMammo(df[df["fold"]=="holdout"], transform = tfms)
# %%
# train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = 4, num_workers = 3)
# val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = True, batch_size = 4, num_workers = 3)
holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = True, batch_size =  2, num_workers = 3)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
#Setting model and moving to device

model_mammo = MammoModel(True).to(device)


# %%
model_mammo.load_state_dict(torch.load('/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/ResNet34/models/crop-image-model-ADAM.pt'))
criterion_multioutput = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model_CNN.parameters(), lr=0.001, momentum=0.9)
# model_CNN.eval()

# %%
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


#%%
bi_max_list = []
den_max_list = []
bi_pred_list = []
den_pred_list = []
den_pred_list = []

bi_correct = 0
den_correct = 0
total_correct  = 0
bi_TP, den_TP, bi_TN, den_TN, bi_FP, den_FP, bi_FN, den_FN = 0,0,0,0,0,0,0,0
TP, TN, FP, FN = 0,0,0,0


print('hi')
for batch_idx, sample_batched in enumerate(tqdm(holdout_dataloader)):
    image, bi, den = sample_batched['image'].to(device), sample_batched['bi'].to(device), sample_batched['den'].to(device)
    bi_max, den_max = torch.max(bi, 1)[1], torch.max(den, 1)[1] #value of real birad and density
    bi_hat, den_hat = model_mammo(image)
    bi_hat = bi_hat.cuda()
    den_hat = den_hat.cuda()
    bi_pred = torch.max(bi_hat,1)[1]#value of predicted birad
    den_pred = torch.max(den_hat,1)[1]#value of predicted density

    bi_max_list.append(bi_max)
    den_max_list.append(den_max)
    bi_pred_list.append(bi_pred)
    den_pred_list.append(den_pred)

    #Accuracy
    bi_correct += (bi_pred == bi_max).sum() 
    den_correct += (den_pred == den_max).sum()
    tmp = ((bi_pred==bi_max)*(den_pred==den_max)).sum()
    total_correct += tmp   


#%%
bi_max_list = torch.cat(bi_max_list, 0)
den_max_list = torch.cat(den_max_list, 0)
bi_pred_list = torch.cat(bi_pred_list, 0)
den_pred_list = torch.cat(den_pred_list, 0)

#%%
print("Accuracy of birad:", bi_correct/len(holdout_dl))
print("Accuracy of density:", den_correct/len(holdout_dl))
print("Total Accuracy", total_correct/len(holdout_dl))
print("Classification report of each label:")

print("BIRAD:")
print(classification_report(bi_max_list.cpu(), bi_pred_list.cpu()))
print(confusion_matrix(bi_max_list.cpu(), bi_pred_list.cpu()))
print("F1 micro:", f1_score(bi_max_list.cpu(), bi_pred_list.cpu(), average = 'micro', zero_division=1))

print("DENSITY:")
print(classification_report(den_max_list.cpu(), den_pred_list.cpu()))
print(confusion_matrix(den_max_list.cpu(), den_pred_list.cpu()))
print("F1 micro:", f1_score(den_max_list.cpu(), den_pred_list.cpu(), average = 'micro', zero_division=1))
#%%