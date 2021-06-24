#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
import cv2          #For transforming image
import matplotlib.pyplot as plt  #For representation #For model building
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
from dataset import MultiLabelMammo
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from efficientnet_pytorch import EfficientNet


device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
#%%
# Full image
df = pd.read_csv("/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/updatedcsv_singleview.csv")
# #path of image
#df["path"] = "/home/single1/BACKUP/SamHUyen/multi_view_mammo_classification/images/" + df["study_id"] + "/" + df["image_id"] + ".png"
#Crop
#df = pd.read_csv("/home/dungnb/workspace/2021_projects/mammo_multiview_classification/mammo/huyen/csv_singleview.csv")
df = df.drop([1146, 1307, 2442, 5710, 7562, 9377, 9382, 15660, 16328, 16348, 18523, 18840])
df.reset_index(drop = True, inplace = True)
#path of image
df["path"] = "/home/single1/BACKUP/SamHUyen/multi_view_mammo_classification/crop-images-heuristic/"  + df["image_id"] + ".png"

#transform images size from (3518, 2800, 3) to (1759,1400,3)
tfms = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

#fold: ['train', 'valid', 'holdout']
train_dl = MultiLabelMammo(df[df["fold"]=="train"], transform = tfms) 
val_dl = MultiLabelMammo(df[df["fold"]=="valid"], transform = tfms)
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

def train_model(model, criterion, optimizer, lr_scheduler, train_dataloader, val_dataloader, n_epochs=25):
    """returns trained model"""
    # initialize tracker for minimum validation loss[]
    valid_loss_min = np.Inf

    global lr

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
            bi_max, den_max = torch.max(bi, 1)[1], torch.max(den, 1)[1]
            
            bi_hat, den_hat =model(image)
            bi_hat = bi_hat.to(device)
            den_hat = den_hat.to(device)
            
            loss1 = criterion(bi_hat, torch.max(bi,1)[1])
            loss2 = criterion(den_hat, torch.max(den,1)[1])
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

            if batch_idx % 400 == 0 or (batch_idx + 1) == len(train_dataloader):
                print('Epoch %d, Batch %d loss: %.6f (%.6f) lr: %.6f' %
                  (epoch, batch_idx + 1, losses.val, losses.avg, optimizer.param_groups[0]['lr']))
        # validate the model #

        model.eval()
        for batch_idx, sample_batched in enumerate(tqdm(val_dataloader)):
            image, bi, den = sample_batched['image'].to(device),\
                             sample_batched['bi'].to(device),\
                             sample_batched['den'].to(device)
            bi_max, den_max = torch.max(bi, 1)[1], torch.max(den, 1)[1]                 
            bi_hat, den_hat = model(image)
            bi_hat = bi_hat.to(device)
            den_hat = den_hat.to(device)
                    
            loss_v1 = criterion(bi_hat, torch.max(bi,1)[1])
            loss_v2 = criterion(den_hat, torch.max(den,1)[1])
            loss=loss_v1+loss_v2
            
            loss_val.update(loss.item(), image.shape[0])
       
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} ({:.6f}) \tValidation Loss: {:.6f} ({:.6f})'.format(
            epoch, losses.val, losses.avg, loss_val.val, loss_val.avg))
        
        ## TODO: save the model if validation loss has decreased
        if loss_val.avg < valid_loss_min:
            torch.save(model.state_dict(), '/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/ResNet34/models/crop-heuristic-image-model-ADAM.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            loss_val.avg))
            valid_loss_min = loss_val.avg
    # return trained model
    return model
# %%
#Setting model and moving to device
model_mammo = MammoModel(True).to(device)

#model_mammo = EfficientNet.from_pretrained('efficientnet-b4').to(device)
#model_mammo._fc = nn.Sequential()
#model_mammo._fc = MultiOutput().to(device)
#train tiep tu cai nay
#model_mammo.load_state_dict(torch.load('/home/dungnb/workspace/2021_projects/mammo_multiview_classification/mammo/sam/singleview_sam/EfficientNet/crop-heuristic-B4.pt'))


#For multilabel output: race and age
criterion_multioutput = nn.CrossEntropyLoss().cuda()
#params
lr = 1e-4
n_epochs = 25
batch_size = 4

train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = batch_size, num_workers = 3)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = True, batch_size = int(batch_size/2), num_workers = 3)

#optimizer = optim.SGD(model_mammo.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(params = model_mammo.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(train_dataloader), eta_min=lr/100, last_epoch=-1)
# %%
model_conv=train_model(model_mammo, criterion_multioutput, optimizer, lr_scheduler, train_dataloader, val_dataloader)