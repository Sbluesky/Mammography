#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
import matplotlib.pyplot as plt  #For representation#For model building
import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from models import  XNetModel
from dataset import MammoLabelMass
from termcolor import cprint
from sklearn.metrics import  f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() 
                                  else "cpu")


#%%
traindf = pd.read_csv("/home/single3/mammo/mammo/data/updatedata/csv/singleview-extratrain.csv")
valdf = pd.read_csv("/home/single3/mammo/mammo/data/updatedata/csv/singleview-valid.csv")
maskdf = pd.read_csv("/home/single3/mammo/mammo/data/updatedata/csv/mass-extratrain.csv")
maskValiddf = pd.read_csv("/home/single3/mammo/mammo/data/updatedata/csv/mass-valid.csv")
valdf = valdf[valdf["label_birad"] !=0 ]

#%%
train_dl = MammoLabelMass(traindf, maskdf, h = 1024, w = 768 ) #h = 1536, w = 1024 => shape features map: (bs, 1, 48, 32)
val_dl = MammoLabelMass(valdf, maskValiddf, h = 1024, w = 768 )
#%%
#plt.imshow(torchvision.utils.make_grid(val_dl[4]['mask']).permute(1, 2, 0))

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
def Lossplot(trainLoss, validLoss, name = 'loss_fig.png', folder = "/home/single3/mammo/mammo/sam/singleview_sam/AnatomyXnet/models/"):
    plt.plot(trainLoss)
    plt.plot(validLoss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(folder+name, facecolor = "w")
    plt.show()

#%%
def train_model(model, criterion_bi, criterion_mask , optimizer, lr_scheduler, train_dataloader, val_dataloader, n_epochs=50, batchsize = 8, use_amp = True):
    """returns trained model"""
    # initialize tracker for minimum validation loss[]
    valid_loss_min = np.Inf
    valid_f1_max = 0.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    global lr
    trainLoss = [] #for build Loss model fig
    validLoss = []
    for epoch in range(0, n_epochs):
        
        losses = AverageMeter()
        loss_val = AverageMeter()
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(tqdm(train_dataloader)):
            with torch.cuda.amp.autocast(enabled=use_amp):
                if epoch == 0: 
                    optimizer.param_groups[0]['lr'] = lr*(batch_idx+1)/len(train_dataloader)
                # importing data and moving to GPU
                img = sample_batched['image'].to(device)
                mask = sample_batched['mask'].to(device) #shape: bs,1,48,32
                birad, density = sample_batched['birad'].to(device),\
                                sample_batched['density'].to(device)
                birad = birad - 1
                #model predict
                bi_hat, mask_pred = model(img)

                loss_bi = criterion_bi(bi_hat, birad)
                loss_mask = criterion_mask(mask_pred, mask)
                loss=(loss_bi+loss_mask)/2
                losses.update(loss.item(), batchsize) 
            
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if epoch > 0:
                lr_scheduler.step()

            if batch_idx % 400 == 0 or (batch_idx + 1) == len(train_dataloader):
                print('Epoch %d, Batch %d loss: %.6f (%.6f) lr: %.6f' %
                  (epoch, batch_idx + 1, losses.val, losses.avg, optimizer.param_groups[0]['lr']))

        # validate the model #
        torch.save(model.state_dict(), '/home/single3/mammo/mammo/sam/singleview_sam/AnatomyXnet/models/efb2_full_1024x768_' + str(epoch) + '.pt')

        bi_list = []
        #den_list = []
        bi_pred_list = []
        #den_pred_list = []
        #dice_scr = 0
        model.eval()
        for batch_idx, sample_batched in enumerate(tqdm(val_dataloader)):
            with torch.cuda.amp.autocast(enabled=use_amp), torch.no_grad():
                img = sample_batched['image'].to(device)
                mask = sample_batched['mask'].to(device)
                birad = sample_batched['birad'].to(device)
                birad = birad - 1
                #model predict
                bi_hat, mask_pred = model(img)

                loss_bi_v = criterion_bi(bi_hat, birad)
                loss_mask_v = criterion_mask(mask_pred, mask)

                loss=(loss_bi_v+loss_mask_v)/2
                loss_val.update(loss.item(), batchsize)

                #f1-score
                bi_hat = torch.max(bi_hat,1)[1].to(device) #lúc training đã chuẩn hóa [1,2,3,4,5] thành [0,1,2,3,4]
                #den_hat = torch.max(den_hat,1)[1].to(device)
                bi_list.append(birad)
                #den_list.append(density)
                bi_pred_list.append(bi_hat)
                #den_pred_list.append(den_hat)

                #Dice Score
                #dice_scr += dice_coeff(mask_pred, mask)
        
        bi_list, bi_pred_list = torch.cat(bi_list, 0), torch.cat(bi_pred_list, 0)
        #den_list, den_pred_list = torch.cat(den_list, 0), torch.cat(den_pred_list, 0)
        f1_scr = f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'macro')
        #dice_scr = dice_scr/(len(bi_list))
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} ({:.6f}) \t f1-score {:.6f} \t dice-scr {:.6f}'.format(
            epoch, losses.avg, loss_val.val, loss_val.avg, f1_scr, 0 ))
        
        trainLoss.append(losses.avg)
        validLoss.append(loss_val.avg)
        ## TODO: save the model if validation loss has decreased
        # if loss_val.avg < valid_loss_min:
        if f1_scr > valid_f1_max: 
            torch.save(model.state_dict(), '/home/single3/mammo/mammo/sam/singleview_sam/AnatomyXnet/models/efb2_full_1024x768_topv1.pt')
            print('Validation f1 increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_f1_max,
            f1_scr))
            valid_f1_max = f1_scr
            # valid_loss_min = loss_val.avg
            count = 0
        else:
            count +=1
            cprint(f'EarlyStopping counter: {count} out of 5', 'yellow')
            if count == 5 or epoch == n_epochs - 1:
                cprint('Early Stop..', 'red')
                torch.save(model.state_dict(), '/home/single3/mammo/mammo/sam/singleview_sam/AnatomyXnet/models/efb2_full_1024x768_last.pt')
                Lossplot(trainLoss,validLoss,'efb2_full_1024x768.png')
                exit(-1)
        
    return model, trainLoss, validLoss
# %%
#Setting model and moving to device
model_CNN = XNetModel(True).to(device)
model_CNN.load_state_dict(torch.load('/home/single3/mammo/mammo/sam/singleview_sam/AnatomyXnet/models/efb2_full_1024x768_last.pt'))
#%%
#freeze backbone: list(model_CNN.children()) #len = 8
freeze = False
if freeze == True:
    ct = 0
    for child in model_CNN.children():
        if ct < 5:
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
        ct+=1

#params
criterion_bi = nn.CrossEntropyLoss().cuda()
criterion_mask = nn.BCEWithLogitsLoss().cuda()
lr = 1e-3
n_epochs = 13
batchsize = 4

train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = batchsize, num_workers = 2)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = True, batch_size = 1, num_workers = 2)

optimizer = optim.SGD(model_CNN.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.Adam(model_CNN.parameters(), lr=lr)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(train_dataloader), eta_min=lr/20, last_epoch=-1)
# %%
model_conv, trainLoss, validLoss=train_model(model_CNN, criterion_bi, criterion_mask, optimizer, lr_scheduler, train_dataloader, val_dataloader, batchsize = batchsize, n_epochs=n_epochs)
torch.save(model_conv.state_dict(), '/home/single3/mammo/mammo/sam/singleview_sam/AnatomyXnet/models/efb2_full_1024x768_last.pt')
Lossplot(trainLoss,validLoss,'efb2_full_1024x768.png')

# %%
