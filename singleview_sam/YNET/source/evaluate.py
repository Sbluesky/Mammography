#%%
# General libraries
import pandas as pd  #For working with dataframes
import torch
from torch import nn
from tqdm import tqdm
import torchvision 
from models import Y_Net
from dataset import MammoLabelMask
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report, confusion_matrix, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
holdoutdf = pd.read_csv("/home/single3/mammo/mammo/data/csv/newsingleholdout.csv")
holdoutdf = holdoutdf[holdoutdf["label_birad"]!=0]
holdoutdf["label_birad"] = holdoutdf["label_birad"] - 1
maskdf = pd.read_csv("/home/single3/mammo/mammo/data/csv/mass-newcrop-holdout.csv")

import matplotlib.pyplot as plt
def plot(image, name):
    plt.imshow(torchvision.utils.make_grid(image).permute(1, 2, 0))
    plt.show()
    plt.savefig("/home/single3/mammo/mammo/sam/singleview_sam/Y-NET/models/" + name)
    

#%%  
###Dice Score###
@torch.no_grad()
def dice_coeff(pred, target, smooth = 1, threshold = 0.01, count = 0):
    #mask processing
    activation_fn = nn.Sigmoid()
    mask_pred = activation_fn(pred) #chuẩn hóa pred thành mask 0, 1
    mask_pred = torch.where(mask_pred >= threshold, torch.tensor(1, dtype = mask_pred.dtype).to(device), torch.tensor(0, dtype = mask_pred.dtype).to(device))
    print("mask_pred: ",torch.count_nonzero(mask_pred))
    print("target: ", torch.count_nonzero(target))
    mask_pred = mask_pred.type(torch.float32)
    plot(mask_pred.cpu(), "pred" + str(count)+ ".png")
    plot(target.cpu(),"groudtruth" + str(count)+ ".png")
    #dice scores computatuion
    num = mask_pred.size(0)
    m1 = mask_pred.view(num, -1).short() # Flatten
    m2 = target.view(num, -1).short() # Flatten
    intersection = (m1 * m2).sum().short()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# %%
#fold: ['train', 'valid', 'holdout']
holdout_dl = MammoLabelMask(holdoutdf[32:], maskdf, h = 1024, w = 768, h_m= 256, w_m= 192)
holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = False, batch_size =  1, num_workers = 2)
# %%
#Setting model and moving to device
model = Y_Net().to(device)
model.load_state_dict(torch.load('/home/single3/mammo/mammo/sam/singleview_sam/Y-NET/models/YNet-1024x768-v1-last.pt'))
model.eval()
#%%
bi_list = []
bi_pred_list = []
bi_correct = 0
dice_scr_full, dice_scr_mask = 0, 0
count_mask = 0 
count = 0
print('Evaluate for multiview: ')
for batch_idx, sample_batched in enumerate(tqdm(holdout_dataloader)):
    with torch.cuda.amp.autocast(enabled=True),torch.no_grad():
        img = sample_batched['image'].to(device)
        mask = sample_batched['mask'].to(device)
        mask_gt = sample_batched['mask_gt'].to(device)
        birad = sample_batched['birad'].to(device)
        birad = birad - 1
        #model predict
        bi_hat, mask_pred = model(img)
        """
        if torch.count_nonzero(mask) > 0. :
            count_mask += 1
            dice_scr_mask += dice_coeff(mask_pred, mask, count= count)
            print('dice score: ', dice_scr_mask)
        """
        #f1-score
        bi_hat = torch.max(bi_hat,1)[1].to(device) #lúc training đã chuẩn hóa [1,2,3,4,5] thành [0,1,2,3,4]
        bi_list.append(birad)
        bi_pred_list.append(bi_hat)

        #Accuracy
        bi_correct += (bi_hat == birad).sum() 
  

#dice_scr_mask = dice_scr_mask/count_mask
bi_list, bi_pred_list = torch.cat(bi_list, 0), torch.cat(bi_pred_list, 0)

#%%
print("Accuracy of birad:", bi_correct/(len(holdout_dl)*1))
print("Classification report of each label:")

print("BIRAD:")
print(classification_report(bi_list.cpu(), bi_pred_list.cpu()))
print(confusion_matrix(bi_list.cpu(), bi_pred_list.cpu()))
print("F1 micro:", f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'micro'))
print("F1 macro:", f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'macro'))
print("Quadratic-weighted-kappa of Birad: ",cohen_kappa_score(bi_list.cpu(), bi_pred_list.cpu(), weights="quadratic"))
print("mask: ", count_mask)
print("Dice scor mask: ", dice_scr_mask)

# %%
