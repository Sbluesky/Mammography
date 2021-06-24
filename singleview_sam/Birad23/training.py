#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
#import cv2          #For transforming image
import albumentations as A #packet for augumentation
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
from sklearn.metrics import f1_score
from termcolor import cprint

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

#%%
traindf = pd.read_csv("/media/tungthanhlee/DATA/mammo/sam/csv/singleview-train.csv")
valdf = pd.read_csv("/media/tungthanhlee/DATA/mammo/sam/csv/singleview-valid.csv")
#path of image
traindf["path"] = "/media/tungthanhlee/DATA/mammo/sam/singleview_sam/dataset/crop-images/" + traindf["study_id"] + "/" + traindf["image_id"] + ".png"
valdf["path"] = "/media/tungthanhlee/DATA/mammo/sam/singleview_sam/dataset/crop-images/" + valdf["study_id"] + "/" + valdf["image_id"] + ".png"

#transform images size from (3518, 2800, 3) to (512,512,3)
tfms1 = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor()])

tfms2 = A.Compose([
    A.RandomResizedCrop(width=512, height=512),
    A.ShiftScaleRotate(shift_limit=0.0625),
    A.GridDistortion(num_steps = 5),
])

traindf = traindf[(traindf["label_birad"] == 2) | (traindf["label_birad"] == 3)] # class2: 5150, class 3: 1654
birad3_df = traindf[traindf['label_birad'] ==3]
traindf = pd.concat((traindf,birad3_df), axis=0)
traindf = traindf.sample(frac=1).reset_index(drop=True) #shuffle dataframe

train_dl = MultiClassMammo(traindf, transform1 = tfms1, transform2 = tfms2) 
val_dl = MultiClassMammo(valdf[(valdf["label_birad"] == 2) | (valdf["label_birad"] == 3)], transform1 = tfms1, transform2 = tfms2)

#%%
#plt.imshow(torchvision.utils.make_grid(train_dl[1]['image']).permute(1, 2, 0))

#%%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

def train_model(model, criterion, optimizer, lr_scheduler, train_dataloader, val_dataloader, n_epochs=25):
    """returns trained model"""
    # initialize tracker for minimum validation loss[]
    valid_loss_min = np.Inf

    global lr
    count = 0
    for epoch in range(0, n_epochs):
        
        losses = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        loss_val = AverageMeter()

        train_loss = 0.0
        valid_loss = 0.0
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(tqdm(train_dataloader)):
            if epoch == 0: 
                optimizer.param_groups[0]['lr'] = lr*(batch_idx+1)/len(train_dataloader)
            # importing data and moving to GPU
            image, bi, den = sample_batched['image'].to(device),\
                             sample_batched['bi'].to(device),\
                             sample_batched['den'].to(device)  
            
            bi, den = bi.type(torch.float32), den.type(torch.float32)

            bi_hat, den_hat =model(image)   
            bi_hat = bi_hat.to(device)
            den_hat = den_hat.to(device)
            bi_hat, den_hat = bi_hat.type(torch.float32), den_hat.type(torch.float32)
            #bi_hat = torch.reshape(bi_hat*6, (-1,)) #di qua activate sigmod tra ve value 0-1, *6 vi co 6 class 
            #den_hat = torch.reshape(den_hat*4, (-1,))
            bi_hat = torch.reshape(bi_hat,(-1,))
            den_hat = torch.reshape(den_hat, (-1,))


            loss1 = criterion(bi_hat, bi)
            loss2 = criterion(den_hat, den)
            loss=loss1+loss2 
            
            losses.update(loss.item(), image.shape[0])
            losses1.update(loss1.item(), image.shape[0])
            losses2.update(loss2.item(), image.shape[0])

            # zero the parameter gradients
            optimizer.zero_grad()
            # back prop
            loss.backward()
            # grad
            optimizer.step()
            if epoch > 0:
                lr_scheduler.step()
            # train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            if batch_idx % 400 == 0 or (batch_idx + 1) == len(train_dataloader):
                print('Epoch %d, Batch %d loss: %.6f (%.6f) lr: %.6f' %
                  (epoch, batch_idx + 1, losses.val, losses.avg, optimizer.param_groups[0]['lr']))
        # validate the model #

        model.eval()
        bi_pred_list = []
        bi_list = []
        f1_score = 0
        for batch_idx, sample_batched in enumerate(tqdm(val_dataloader)):
            image, bi, den = sample_batched['image'].to(device),\
                             sample_batched['bi'].to(device),\
                             sample_batched['den'].to(device)
            bi, den = bi.type(torch.float32), den.type(torch.float32)

            # output = model(image)
            bi_hat, den_hat = model(image)
            bi_hat = bi_hat.to(device)
            den_hat = den_hat.to(device)
            bi_hat, den_hat = bi_hat.type(torch.float32), den_hat.type(torch.float32)
            #bi_hat = torch.reshape(bi_hat*6, (-1,)) #di qua activate sigmod tra ve value 0-1, *6 vi co 6 class 
            #den_hat = torch.reshape(den_hat*4, (-1,))       
            bi_hat = torch.reshape(bi_hat,(-1,))
            den_hat = torch.reshape(den_hat, (-1,))

            # calculate loss
            # loss1=criterion1(bi_hat, bi.squeeze().type(torch.LongTensor).to(device))
            # loss2=criterion2(den_hat, den.squeeze().type(torch.LongTensor).to(device))
                        
            loss_v1 = criterion(bi_hat, bi)
            loss_v2 = criterion(den_hat, den)
            loss=loss_v1+loss_v2
            
            loss_val.update(loss.item(), image.shape[0])
            # valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            #f1-score
            bi_hat = convert_classify_bi(bi_hat).to(device)
            den_hat = convert_classify_den(den_hat).to(device)
            bi_list.append(bi)
            bi_pred_list.append(bi_hat)

        #f1_score = f1_score(bi_list, bi_pred_list, average = 'macro')
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} ({:.6f}) \tf1-score: {:.6f}'.format(
            epoch, losses.avg, loss_val.val, loss_val.avg, f1_score))
        
        ## TODO: save the model if validation loss has decreased
        if loss_val.avg < valid_loss_min:
            torch.save(model.state_dict(), '/media/tungthanhlee/DATA/mammo/sam/singleview_sam/Birad23/models/class23-top-layer4.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            loss_val.avg))
            valid_loss_min = loss_val.avg
        else:
                count +=1
                cprint(f'EarlyStopping counter: {count} out of 10', 'yellow')
                if count == 10:
                    cprint('Early Stop..', 'red')
                exit(-1)
    # return trained model
    return model
# %%
#Setting model and moving to device
model_CNN = MammoModel(True).to(device)
model_CNN.load_state_dict(torch.load('/media/tungthanhlee/DATA/mammo/sam/singleview_sam/Birad23/models/class23-top-layer4.pt'))
#%%
#freeze backbone: list(model_CNN.children())[0]
freeze = False
if freeze == True:
    ct = 0
    for child in model_CNN.children():
        if ct < 1:
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
        ct+=1
    

#For multilabel output: race and age
criterion_multioutput = nn.MSELoss().cuda()
#params
lr = 1e-4
n_epochs = 12
batch_size = 16

train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = batch_size, num_workers = 2)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = True, batch_size = batch_size, num_workers = 2)

optimizer = optim.SGD(model_CNN.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.Adam(model_CNN.parameters(), lr=lr)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(train_dataloader), eta_min=lr/100, last_epoch=-1)
# %%
model_conv=train_model(model_CNN, criterion_multioutput, optimizer, lr_scheduler, train_dataloader, val_dataloader)
torch.save(model_conv.state_dict(), '/media/tungthanhlee/DATA/mammo/sam/singleview_sam/Birad23/models/class23-last-layer4.pt')
# %%
