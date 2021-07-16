#%%
# General libraries
import pandas as pd  #For working with dataframes
import torch
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from models import Mixupmodel
from dataset import OnlyImage
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fn_holdoutdf = "/home/single3/mammo/mammo/data/csv/newsingleholdout.csv"
fn_model = "/home/single3/mammo/mammo/sam/singleview_sam/MixAugment/models/Mixup-1024x768-v2-top.pt"
#%%
holdoutdf = pd.read_csv(fn_holdoutdf)
holdoutdf = holdoutdf[holdoutdf["label_birad"]!=0]
holdoutdf["birad"].unique()
# %%
#fold: ['train', 'valid', 'holdout']
holdout_dl = OnlyImage(holdoutdf, h = 1024, w = 768 )
holdout_dataloader = torch.utils.data.DataLoader(holdout_dl, shuffle = False, batch_size =  1, num_workers = 0)
# %%/
# %%
#Setting model and moving to device
model_mammo = Mixupmodel().to(device)
model_mammo.load_state_dict(torch.load(fn_model))
model_mammo.eval()

#%%
bi_list = []
bi_pred_list = []
bi_correct = 0


print('Evaluate for multiview: ')
for batch_idx, sample_batched in enumerate(tqdm(holdout_dataloader)):
    with torch.cuda.amp.autocast(enabled=True),torch.no_grad():
        img = sample_batched['image'].to(device)

        birad = sample_batched['birad'].to(device)
        birad = birad -1
                #model predict
        bi_hat = model_mammo(img)
                #f1-score
        bi_hat = torch.max(bi_hat,1)[1].to(device) #lúc training đã chuẩn hóa [1,2,3,4,5] thành [0,1,2,3,4]
        bi_list.append(birad)
        bi_pred_list.append(bi_hat)

        #Accuracy
        bi_correct += (bi_hat == birad).sum() 
  


#%%
bi_list = torch.cat(bi_list, 0)
bi_pred_list = torch.cat(bi_pred_list, 0)

#%%
print("Accuracy of birad:", bi_correct/(len(holdout_dl)*1))
print("Classification report of each label:")

print("BIRAD:")
print(classification_report(bi_list.cpu(), bi_pred_list.cpu()))
print(confusion_matrix(bi_list.cpu(), bi_pred_list.cpu()))
print("F1 micro:", f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'micro'))
print("F1 macro:", f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'macro'))
print("Quadratic-weighted-kappa of Birad: ",cohen_kappa_score(bi_list.cpu(), bi_pred_list.cpu(), weights="quadratic"))


#%%'