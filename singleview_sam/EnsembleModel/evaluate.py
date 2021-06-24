#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
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
from EnsembleModel import SingleModel
from datasets import MultiClassMammo
from sklearn.metrics import cohen_kappa_score

#%%
#Crop   
df = pd.read_csv("/home/single2/mammo/mammo/data/updatedata/csv/singleview-holdout.csv")
#path of image
df["path"] = "/home/single2/mammo/mammo/data/updatedata/crop-images/"  + df["study_id"] + "/" + df["image_id"] + ".png"

# %%
#transform images size from (3518, 2800, 3) to (1759,1400,3)
tfms = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
# %%
holdout_dl = MultiClassMammo(df[df["label_birad"] != 0], transform = tfms) 
# %%
holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = True, batch_size =  4, num_workers = 3)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
#Setting model and moving to device
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512,1)

model1 = model.to(device)
model2 = model.to(device)
model3 = model.to(device)
model4 = model.to(device)
model0 = model.to(device)


# %%
model1.load_state_dict(torch.load('/home/single2/mammo/mammo/sam/singleview_sam/EnsembleModel/models/model1-top.pt'))
model2.load_state_dict(torch.load('/home/single2/mammo/mammo/sam/singleview_sam/EnsembleModel/models/model2-top.pt'))
model3.load_state_dict(torch.load('/home/single2/mammo/mammo/sam/singleview_sam/EnsembleModel/models/model3-top.pt'))
model4.load_state_dict(torch.load('/home/single2/mammo/mammo/sam/singleview_sam/EnsembleModel/models/model4-top.pt'))
model0.load_state_dict(torch.load('/home/single2/mammo/mammo/sam/singleview_sam/EnsembleModel/models/model0-top.pt'))

# %%
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

x = 1.2
def convert_classify_bi(bi):
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

#%%
bi_list = []
bi_pred_list = []

bi_correct = 0
bi_TP, bi_TN, bi_FP, bi_FN = 0,0,0,0
TP, TN, FP, FN = 0,0,0,0


print('Mammo Classification:')
for batch_idx, sample_batched in enumerate(tqdm(holdout_dataloader)):
    image, bi = sample_batched['image'].to(device), sample_batched['bi'].to(device)
    bi = bi.type(torch.float32)
    bi_hat = model1(image)
    """
    bi_hat2 = model2(image)
    bi_hat3 = model3(image)
    bi_hat4 = model4(image)
    bi_hat5 = model0(image)
    bi_hat = (bi_hat1 + bi_hat2 + bi_hat3 +bi_hat4 + bi_hat5)/5 
    """
    print("bi_hat: ", bi_hat)
    print("bi: ", bi)
    bi_hat = torch.reshape(bi_hat, (-1,))
    bi_hat = bi_hat.type(torch.float32)

    bi_hat = convert_classify_bi(bi_hat).to(device)
    bi_list.append(bi)
    bi_pred_list.append(bi_hat)

    #Accuracy
    bi_correct += (bi_hat == bi).sum() 


#%%
bi_list = torch.cat(bi_list, 0)
bi_pred_list = torch.cat(bi_pred_list, 0)

#%%
print("Accuracy of birad:", bi_correct/len(holdout_dl))
print("Classification report of each label:")

print("BIRAD:")
print(classification_report(bi_list.cpu(), bi_pred_list.cpu()))
print(confusion_matrix(bi_list.cpu(), bi_pred_list.cpu()))
print("F1 micro:", f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'micro', zero_division=1))

#%%