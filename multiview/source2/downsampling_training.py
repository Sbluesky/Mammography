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
from models import  BreastWiseModel, ViewWiseModel, ViewWiseModelv4
from dataset import MultiClassMammo
from termcolor import cprint
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")


#%%
traindf = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_train.csv")
valdf = pd.read_csv("/home/single4/mammo/mammo/data/updatedata/csv/multiview_valid.csv")

traindf1 = traindf[(traindf["L_birad_max"] == 1) | (traindf["R_birad_max"] == 1)]
traindf2 = traindf[(traindf["L_birad_max"] == 2) | (traindf["R_birad_max"] == 2)]
traindf3 = traindf[(traindf["L_birad_max"] == 3) | (traindf["R_birad_max"] == 3)]
traindf4 = traindf[(traindf["L_birad_max"] == 4) | (traindf["R_birad_max"] == 4)]
traindf5 = traindf[(traindf["L_birad_max"] == 5) | (traindf["R_birad_max"] == 5)]
"""
traindf1.info()
traindf2.info()
traindf3.info()
traindf4.info()
traindf5.info()
"""
trainlist = []
for i in range(0,5):
    if i < 2:
        tempdf = pd.concat((traindf1[i*312:i*312+312], traindf2[i*312:i*312+312], traindf3[:312], traindf4, traindf5,traindf5,traindf5,traindf5,traindf5 ), axis=0)
    else:
        tempdf = pd.concat((traindf1[i*312:i*312+312], traindf2[i*312:i*312+312], traindf3[312:], traindf4, traindf5,traindf5,traindf5,traindf5,traindf5  ), axis=0)
    tempdf = tempdf.reset_index()
    trainlist.append(tempdf.drop(['index'], axis = 1))

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
def train_model(model, criterion, optimizer, lr_scheduler, trainlist, val_dataloader, n_epochs=50):
    """returns trained model"""
    # initialize tracker for minimum validation loss[]
    valid_loss_min = np.Inf

    global lr
    count = 0
    batch_size = 2
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
        
        #data loader #mỗi epoch sẽ lấy 1 tập dataset khác nhau từ 1-5
        train_dl = MultiClassMammo(trainlist[epoch%5], transform1 = tfms1, transform2 = None) 
        train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = batch_size, num_workers = 2)

        for batch_idx, sample_batched in enumerate(tqdm(train_dataloader)):
            if epoch == 0: 
                optimizer.param_groups[0]['lr'] = lr*(batch_idx+1)/len(train_dataloader)
            # importing data and moving to GPU
            L_CC_img, R_CC_img, L_MLO_img, R_MLO_img = \
                             sample_batched['L_CC_img'].to(device),\
                             sample_batched['R_CC_img'].to(device),\
                             sample_batched['L_MLO_img'].to(device),\
                             sample_batched['R_MLO_img'].to(device)
            #max label
            L_birad, R_birad, L_density, R_density = \
                             sample_batched['L_birad'].to(device),\
                             sample_batched['R_birad'].to(device),\
                             sample_batched['L_density'].to(device),\
                             sample_batched['R_density'].to(device)
            
            #true label
            LCC_bi, RCC_bi, LMLO_bi, RMLO_bi = \
                             sample_batched['LCC_bi'].to(device),\
                             sample_batched['RCC_bi'].to(device),\
                             sample_batched['LMLO_bi'].to(device),\
                             sample_batched['RMLO_bi'].to(device)

            L_birad, R_birad, L_density, R_density = L_birad.type(torch.float32), R_birad.type(torch.float32), L_density.type(torch.float32), R_density.type(torch.float32)
            LCC_bi, RCC_bi, LMLO_bi, RMLO_bi = LCC_bi.type(torch.float32), RCC_bi.type(torch.float32), LMLO_bi.type(torch.float32), RMLO_bi.type(torch.float32)

            #model predict
            output = model(L_CC_img, R_CC_img, L_MLO_img, R_MLO_img)
            #max label
            L_bi_hat = torch.reshape(output["L_bi"].type(torch.float32).to(device),(-1,))
            L_den_hat = torch.reshape(output["L_den"].type(torch.float32).to(device),(-1,))
            R_bi_hat = torch.reshape(output["R_bi"].type(torch.float32).to(device),(-1,))
            R_den_hat = torch.reshape(output["R_den"].type(torch.float32).to(device),(-1,))
            #true label
            LCC_bi_hat = torch.reshape(output["ft_LCC"].type(torch.float32).to(device),(-1,))
            RCC_bi_hat = torch.reshape(output["ft_RCC"].type(torch.float32).to(device),(-1,))
            LMLO_bi_hat = torch.reshape(output["ft_LMLO"].type(torch.float32).to(device),(-1,))
            RMLO_bi_hat = torch.reshape(output["ft_RMLO"].type(torch.float32).to(device),(-1,))
        
            #loss
            loss_L_bi = criterion(L_bi_hat, L_birad)
            loss_L_den = criterion(L_den_hat, L_density)
            loss_R_bi = criterion(R_bi_hat, R_birad)
            loss_R_den = criterion(R_den_hat, R_density)
            loss_LCC = criterion(LCC_bi_hat, LCC_bi)
            loss_LMLO = criterion(LMLO_bi_hat, LMLO_bi)
            loss_RCC = criterion(RCC_bi_hat, RCC_bi)
            loss_RMLO = criterion(RMLO_bi_hat, RMLO_bi)

            loss=(loss_L_bi + loss_L_den + loss_R_bi + loss_R_den + loss_LCC + loss_LMLO + loss_RCC + loss_RMLO)/8

            #loss calculator 
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
            output = model(L_CC_img, R_CC_img, L_MLO_img, R_MLO_img)
            #max label
            L_bi_hat = torch.reshape(output["L_bi"].type(torch.float32).to(device),(-1,))
            L_den_hat = torch.reshape(output["L_den"].type(torch.float32).to(device),(-1,))
            R_bi_hat = torch.reshape(output["R_bi"].type(torch.float32).to(device),(-1,))
            R_den_hat = torch.reshape(output["R_den"].type(torch.float32).to(device),(-1,))
            #true label
            LCC_bi_hat = torch.reshape(output["ft_LCC"].type(torch.float32).to(device),(-1,))
            RCC_bi_hat = torch.reshape(output["ft_RCC"].type(torch.float32).to(device),(-1,))
            LMLO_bi_hat = torch.reshape(output["ft_LMLO"].type(torch.float32).to(device),(-1,))
            RMLO_bi_hat = torch.reshape(output["ft_RMLO"].type(torch.float32).to(device),(-1,))

            #loss
            loss_L_bi_v = criterion(L_bi_hat, L_birad)
            loss_L_den_v = criterion(L_den_hat, L_density)
            loss_R_bi_v = criterion(R_bi_hat, R_birad)
            loss_R_den_v = criterion(R_den_hat, R_density)
            loss_LCC_v = criterion(LCC_bi_hat, LCC_bi)
            loss_LMLO_v = criterion(LMLO_bi_hat, LMLO_bi)
            loss_RCC_v = criterion(RCC_bi_hat, RCC_bi)
            loss_RMLO_v = criterion(RMLO_bi_hat, RMLO_bi)
            loss=(loss_L_bi_v+loss_L_den_v+loss_R_bi_v+loss_R_den_v+loss_LCC_v+loss_LMLO_v+loss_RCC_v+loss_RMLO_v)/8
            
            loss_val.update(loss.item(), 8)

        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} ({:.6f})'.format(
            epoch, losses.avg, loss_val.val, loss_val.avg))
        
        trainLoss.append(losses.avg)
        validLoss.append(loss_val.avg)
        ## TODO: save the model if validation loss has decreased
        if loss_val.avg < valid_loss_min:
            torch.save(model.state_dict(), '/home/single4/mammo/mammo/sam/multiview/modeltruelabel/viewwise-undersample-top.pt')
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
                torch.save(model.state_dict(), '/home/single4/mammo/mammo/sam/multiview/modeltruelabel/viewwise-undersample-last.pt')
                Lossplot(trainLoss,validLoss,'viewwisev_truelabel-undersample.png')
                exit(-1)
    # return trained model
    return model, trainLoss, validLoss
# %%
#Setting model and moving to device
model_CNN = ViewWiseModel(True).to(device) #BreastWiseModel, MammoModel
#model_CNN.load_state_dict(torch.load('/home/single4/mammo/mammo/sam/multiview/models/viewwisev5-top.pt'))
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
criterion_multioutput = nn.MSELoss().cuda()
#params
lr = 1e-4
n_epochs = 12
batch_size = 2

val_dl = MultiClassMammo(valdf, transform1 = tfms1, transform2 = None)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = False, batch_size = batch_size, num_workers = 2)

optimizer = optim.SGD(model_CNN.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.Adam(model_CNN.parameters(), lr=lr)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*1314, eta_min=lr/100, last_epoch=-1)
# %%
model_conv, trainLoss, validLoss=train_model(model_CNN, criterion_multioutput, optimizer, lr_scheduler, trainlist, val_dataloader)
torch.save(model_conv.state_dict(), '/home/single4/mammo/mammo/sam/multiview/modeltruelabel/viewwise-undersample-last.pt')
Lossplot(trainLoss,validLoss,'viewwise_truelabel-undersample.png')
# %%
