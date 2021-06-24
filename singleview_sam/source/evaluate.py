#%%
# General libraries
import pandas as pd  #For working with dataframes
import torch
from torch import nn, optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.optim import lr_scheduler
from skimage.transform import AffineTransform, warp
from tqdm import tqdm
from models import BaselineModel
from dataset import MultiClassMammo
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")
#%%
#%%
holdoutdf = pd.read_csv("/home/single1/BACKUP/SamHUyen/mammo/data/updatedata/csv/singleview-holdout.csv")
holdoutdf = holdoutdf[holdoutdf["label_birad"]!=0]
holdoutdf["label_birad"] = holdoutdf["label_birad"] - 1
# %%
#transform images size from (3518, 2800, 3) to (1759,1400,3)
tfms1 = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor()])

# %%
#fold: ['train', 'valid', 'holdout']
holdout_dl = MultiClassMammo(holdoutdf, transform1 = tfms1,transform2 = None)
# %%
holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = False, batch_size =  1, num_workers = 1)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
#Setting model and moving to device
model_mammo = BaselineModel(True).to('cuda')
model_mammo.load_state_dict(torch.load('/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/TrainingBaseline/models/baseline_1024x512_top.pt'))


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


print('Evaluate for multiview: ')
for batch_idx, sample_batched in enumerate(tqdm(holdout_dataloader)):
    img = sample_batched['image'].to(device)

    birad, density = sample_batched['bi'].to(device),\
                             sample_batched['den'].to(device)

            #model predict
    bi_hat, den_hat = model_mammo(img)


            #f1-score
    bi_hat = torch.max(bi_hat,1)[1].to(device) #lúc training đã chuẩn hóa [1,2,3,4,5] thành [0,1,2,3,4]
    den_hat = torch.max(den_hat,1)[1].to(device)
    bi_list.append(birad)
    den_list.append(density)
    bi_pred_list.append(bi_hat)
    den_pred_list.append(den_hat)

    #Accuracy
    bi_correct += (bi_hat == birad).sum() 
    den_correct += (den_hat == density).sum() 
  


#%%
bi_list,den_list = torch.cat(bi_list, 0),torch.cat(den_list, 0)
bi_pred_list,den_pred_list = torch.cat(bi_pred_list, 0),torch.cat(den_pred_list, 0)

#%%
print("Accuracy of birad:", bi_correct/(len(holdout_dl)*1))
print("Accuracy of density:", den_correct/(len(holdout_dl)*1))
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

#%%'