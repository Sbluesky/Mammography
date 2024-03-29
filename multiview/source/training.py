#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
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
from models import  ViewWiseModelv1, ViewWiseModelv2
from dataset import MultiClassMammo
from termcolor import cprint
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")


#%%
traindf = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv")
valdf = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_valid.csv")


#%%

#Augmentation
tfms1 = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor()])

tfms2 = A.Compose([
    A.RandomResizedCrop(width=512, height=512),
    A.ShiftScaleRotate(shift_limit=0.0625),
    A.GridDistortion(num_steps = 5),
])

train_dl = MultiClassMammo(traindf, transform1 = tfms1, transform2 = None) 
val_dl = MultiClassMammo(valdf, transform1 = tfms1, transform2 = None)

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
#%%
def Lossplot(trainLoss, validLoss, name = 'loss_fig.png'):
    plt.plot(trainLoss)
    plt.plot(validLoss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('/home/single4/mammo/mammo/sam/multiview/lossfig/'+name, facecolor = "w")
    plt.show()
    

#%%
def train_model(model, criterion, optimizer, lr_scheduler, train_dataloader, val_dataloader, n_epochs=50):
    """returns trained model"""
    # initialize tracker for minimum validation loss[]
    valid_loss_min = np.Inf

    global lr
    count = 0
    trainLoss = [] #for build Loss model fig
    validLoss = []
    for epoch in range(0, n_epochs):
        
        losses = AverageMeter()
        loss_val = AverageMeter()
        losses_L_bi = AverageMeter()
        losses_L_den = AverageMeter()
        losses_R_bi = AverageMeter()
        losses_R_den = AverageMeter()

        train_loss = 0.0
        valid_loss = 0.0
        # train the model #
        model.train()
        """for index, (name, param) in enumerate(model.named_parameters()):
            a ,b =name, param.grad
        print(a, b)"""
        for batch_idx, sample_batched in enumerate(tqdm(train_dataloader)):
            if epoch == 0: 
                optimizer.param_groups[0]['lr'] = lr*(batch_idx+1)/len(train_dataloader)
            # importing data and moving to GPU
            L_CC_img, R_CC_img, L_MLO_img, R_MLO_img = \
                             sample_batched['L_CC_img'].to(device),\
                             sample_batched['R_CC_img'].to(device),\
                             sample_batched['L_MLO_img'].to(device),\
                             sample_batched['R_MLO_img'].to(device)

            L_birad, R_birad, L_density, R_density = \
                             sample_batched['L_birad'].to(device),\
                             sample_batched['R_birad'].to(device),\
                             sample_batched['L_density'].to(device),\
                             sample_batched['R_density'].to(device)

            L_birad, R_birad, L_density, R_density = L_birad.type(torch.float32), R_birad.type(torch.float32), L_density.type(torch.float32), R_density.type(torch.float32)
            #model predict
            L_bi_hat, L_den_hat, R_bi_hat, R_den_hat = model(L_CC_img, R_CC_img, L_MLO_img, R_MLO_img)
            L_bi_hat = torch.reshape(L_bi_hat.type(torch.float32).to(device),(-1,))
            L_den_hat = torch.reshape(L_den_hat.type(torch.float32).to(device),(-1,))
            R_bi_hat = torch.reshape(R_bi_hat.type(torch.float32).to(device),(-1,))
            R_den_hat = torch.reshape(R_den_hat.type(torch.float32).to(device),(-1,))
        

            #print("L_bi_hat, L_den_hat, R_bi_hat, R_den_hat",L_bi_hat, L_den_hat, R_bi_hat, R_den_hat)
            loss_L_bi = criterion(L_bi_hat, L_birad)
            loss_L_den = criterion(L_den_hat, L_density)
            loss_R_bi = criterion(R_bi_hat, R_birad)
            loss_R_den = criterion(R_den_hat, R_density)
            loss=(loss_L_bi+loss_L_den+loss_R_bi+loss_R_den)/4
            losses.update(loss.item(), 8) #L_CC_img.shape[0] = batchsize
            
            
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
        for batch_idx, sample_batched in enumerate(tqdm(val_dataloader)):
            L_CC_img, R_CC_img, L_MLO_img, R_MLO_img = \
                             sample_batched['L_CC_img'].to(device),\
                             sample_batched['R_CC_img'].to(device),\
                             sample_batched['L_MLO_img'].to(device),\
                             sample_batched['R_MLO_img'].to(device)

            L_birad, R_birad, L_density, R_density = \
                             sample_batched['L_birad'].to(device),\
                             sample_batched['R_birad'].to(device),\
                             sample_batched['L_density'].to(device),\
                             sample_batched['R_density'].to(device)
                             
            L_birad, R_birad, L_density, R_density = L_birad.type(torch.float32), R_birad.type(torch.float32), L_density.type(torch.float32), R_density.type(torch.float32)
            
            #model predict
            L_bi_hat, L_den_hat, R_bi_hat, R_den_hat =model(L_CC_img, R_CC_img, L_MLO_img, R_MLO_img)
            L_bi_hat = torch.reshape(L_bi_hat.type(torch.float32).to(device),(-1,))
            L_den_hat = torch.reshape(L_den_hat.type(torch.float32).to(device),(-1,))
            R_bi_hat = torch.reshape(R_bi_hat.type(torch.float32).to(device),(-1,))
            R_den_hat = torch.reshape(R_den_hat.type(torch.float32).to(device),(-1,))


            loss_L_bi_v = criterion(L_bi_hat, L_birad)
            loss_L_den_v = criterion(L_den_hat, L_density)
            loss_R_bi_v = criterion(R_bi_hat, R_birad)
            loss_R_den_v = criterion(R_den_hat, R_density)
            loss=(loss_L_bi_v+loss_L_den_v+loss_R_bi_v+loss_R_den_v)/4
            
            loss_val.update(loss.item(), 8)

        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} ({:.6f})'.format(
            epoch, losses.avg, loss_val.val, loss_val.avg))
        
        trainLoss.append(losses.avg)
        validLoss.append(loss_val.avg)
        ## TODO: save the model if validation loss has decreased
        if loss_val.avg < valid_loss_min:
            torch.save(model.state_dict(), '/home/single4/mammo/mammo/sam/multiview/models_extradata/viewwise1-top.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            loss_val.avg))
            valid_loss_min = loss_val.avg
            count = 0
        else:
            count +=1
            cprint(f'EarlyStopping counter: {count} out of 20', 'yellow')
            if count == 25:
                cprint('Early Stop..', 'red')
                torch.save(model.state_dict(), '/home/single4/mammo/mammo/sam/multiview/models_extradata/viewwise1_last.pt')
                Lossplot(trainLoss,validLoss,'viewwise1_extradata.png')
                exit(-1)
    # return trained model
    return model, trainLoss, validLoss
# %%
#Setting model and moving to device
model_CNN = ViewWiseModelv2(True).to(device) #BreastWiseModel, MammoModel
model_CNN.load_state_dict(torch.load('/home/single4/mammo/mammo/sam/multiview/models_extradata/viewwise1-top.pt'))
#%%
#freeze backbone: list(model_CNN.children()) #len = 8
freeze = True
if freeze == True:
    ct = 0
    for child in model_CNN.children():
        if ct < 5:
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
        ct+=1
criterion_multioutput = nn.MSELoss().cuda()
#params
lr = 1e-4
n_epochs = 12
batch_size = 16

train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = batch_size, num_workers = 2)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = False, batch_size = int(batch_size/2), num_workers = 2)

optimizer = optim.SGD(model_CNN.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.Adam(model_CNN.parameters(), lr=lr)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(train_dataloader), eta_min=lr/100, last_epoch=-1)
# %%
model_conv, trainLoss, validLoss=train_model(model_CNN, criterion_multioutput, optimizer, lr_scheduler, train_dataloader, val_dataloader)
torch.save(model_conv.state_dict(), '/home/single4/mammo/mammo/sam/multiview/models_extradata/viewwise1-last.pt')
Lossplot(trainLoss,validLoss,'viewwise1_extradata.png')
# %%
