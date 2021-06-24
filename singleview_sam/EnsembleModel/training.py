#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
#import cv2          #For transforming image
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
from EnsembleModel import SingleModel
from datasets import MultiClassMammo

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
#%%
traindf = pd.read_csv("/home/single2/mammo/mammo/data/updatedata/csv/singleview-train.csv")
valdf = pd.read_csv("/home/single2/mammo/mammo/data/updatedata/csv/singleview-valid.csv")
#path of image
traindf["path"] = "/home/single2/mammo/mammo/data/updatedata/crop-images/" + traindf["study_id"] + "/" + traindf["image_id"] + ".png"
valdf["path"] = "/home/single2/mammo/mammo/data/updatedata/crop-images/" + valdf["study_id"] + "/" + valdf["image_id"] + ".png"
traindf = traindf[traindf["label_birad"]!=0]
valdf = valdf[valdf["label_birad"]!=0]
#function to split data
def create_df(traindf, ind):
    bi1 = traindf[traindf["label_birad"]==1][ind*400:(ind*400+400)]
    bi2 = traindf[traindf["label_birad"]==2][ind*400:(ind*400+400)]
    bi3 = traindf[traindf["label_birad"]==3][ind*200:(ind*200+200)]
    if ind < 4:
        bi4 = traindf[traindf["label_birad"]==4][ind*134:(ind*134+134)]
    else:
        bi4 = traindf[traindf["label_birad"]==4][(649-134):]
    bi5 = traindf[traindf["label_birad"]==5]
    model_df = pd.concat((bi1,bi2,bi3,bi4,bi5), axis=0)
    return model_df



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

def train_model(model, criterion, optimizer, lr_scheduler, train_dataloader, val_dataloader, n_epochs=10, modelpath = ""):
    """returns trained model"""
    # initialize tracker for minimum validation loss[]
    valid_loss_min = np.Inf

    global lr

    for epoch in range(0, n_epochs):
        
        losses = AverageMeter()
        loss_val = AverageMeter()

        train_loss = 0.0
        valid_loss = 0.0
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(tqdm(train_dataloader)):
            if epoch == 0: 
                optimizer.param_groups[0]['lr'] = lr*(batch_idx+1)/len(train_dataloader)
            # importing data and moving to GPU
            image, bi= sample_batched['image'].to(device),\
                             sample_batched['bi'].to(device)
            
            bi = bi.type(torch.float32)

            bi_hat =model(image)   
            bi_hat = bi_hat.to(device)
            bi_hat = bi_hat.type(torch.float32)
            bi_hat = torch.reshape(bi_hat,(-1,))

            loss = criterion(bi_hat, bi)

            losses.update(loss.item(), image.shape[0])

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
            image, bi = sample_batched['image'].to(device),\
                             sample_batched['bi'].to(device)
            bi = bi.type(torch.float32)

            bi_hat= model(image)
            bi_hat = bi_hat.to(device)
            bi_hat= bi_hat.type(torch.float32)
   
            bi_hat = torch.reshape(bi_hat,(-1,))
            
            loss= criterion(bi_hat, bi)
            
            loss_val.update(loss.item(), image.shape[0])

        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} ({:.6f})'.format(
            epoch, losses.avg, loss_val.val, loss_val.avg))
        
        ## TODO: save the model if validation loss has decreased
        if loss_val.avg < valid_loss_min:
            torch.save(model.state_dict(), modelpath)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            loss_val.avg))
            valid_loss_min = loss_val.avg
    # return trained model
    return model
# %%

#transform images size from (3518, 2800, 3) to (1759,1400,3)
lr = 1e-4
n_epochs = 12
batch_size = 4
tfms = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
val_dl = MultiClassMammo(valdf, transform = tfms)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = True, batch_size = batch_size, num_workers = 2)
#for i in range(0,5):
traindf = create_df(traindf, 4)
    #fold: ['train', 'valid', 'holdout']
train_dl = MultiClassMammo(traindf, transform = tfms) 
    #Setting model and moving to device
    #model_CNN = SingleModel(True).to(device)
model_CNN = models.resnet18(pretrained=True)
model_CNN.fc = nn.Linear(512,1)
model_CNN = model_CNN.to(device)
    #model_CNN.load_state_dict(torch.load('/home/tungthanhlee/mammo/sam/singleview_sam/models/regresstion-layer3-last.pt'))
    #For multilabel output: race and age
criterion_multioutput = nn.MSELoss().cuda()
    #params
pathmodel = "/home/single2/mammo/mammo/sam/singleview_sam/EnsembleModel/models/model" + str(4) +"-top.pt"
train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = batch_size, num_workers = 2)

optimizer = optim.SGD(model_CNN.parameters(), lr=lr, momentum=0.9)
    #optimizer = optim.Adam(model_CNN.parameters(), lr=lr)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(train_dataloader), eta_min=lr/100, last_epoch=-1)
    # %%
model_conv=train_model(model_CNN, criterion_multioutput, optimizer, lr_scheduler, train_dataloader, val_dataloader, modelpath =pathmodel )
torch.save(model_conv.state_dict(), '/home/single2/mammo/mammo/sam/singleview_sam/EnsembleModel/models/model'+ str(4) +"-last.pt")
# %%
