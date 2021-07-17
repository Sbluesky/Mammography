#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
import matplotlib.pyplot as plt  #For representation#For model building
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from models import  XNetModel
from dataset import MammoLabelMass
from termcolor import cprint
from sklearn.metrics import  f1_score
device = torch.device("cuda:0" if torch.cuda.is_available() 
                                  else "cpu")

#filename
fn_traindf = "/home/single3/mammo/mammo/data/updatedata/csv/singleview-extratrain.csv"
fn_valdf = "/home/single3/mammo/mammo/data/updatedata/csv/singleview-valid.csv"
fn_maskdf = "/home/single3/mammo/mammo/data/updatedata/csv/mass-extratrain.csv"
fn_maskValdf = "/home/single3/mammo/mammo/data/updatedata/csv/mass-valid.csv"
fn_fig = "/home/single3/mammo/mammo/sam/singleview_sam/AnatomyXnet/models/efb2_crop_1024x768.png"
fn_log = "/home/single3/mammo/mammo/sam/singleview_sam/AnatomyXnet/models/"
fn_model = "/home/single3/mammo/mammo/sam/singleview_sam/AnatomyXnet/models/efb2_full_1024x768_top-"
fn_premodel = None

#params
tfms_height = 1024
tfms_width = 768
criterion_bi = nn.CrossEntropyLoss().cuda()
criterion_mask = nn.BCEWithLogitsLoss().cuda()
lr = 1e-3
n_epochs = 13
batchsize = 4
#%%
traindf = pd.read_csv(fn_traindf)
valdf = pd.read_csv(fn_valdf)
maskdf = pd.read_csv(fn_maskdf)
maskValiddf = pd.read_csv(fn_maskValdf)
valdf = valdf[valdf["label_birad"] !=0 ]

#%%
train_dl = MammoLabelMass(traindf, maskdf, h = tfms_height, w = tfms_width ) #h = 1536, w = 1024 => shape features map: (bs, 1, 48, 32)
val_dl = MammoLabelMass(valdf, maskValiddf, h = tfms_height, w = tfms_width )
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
def Lossplot(trainLoss, validLoss):
    plt.plot(trainLoss)
    plt.plot(validLoss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(fn_fig, facecolor = "w")
    plt.show()


def savelog(epoch,losstrain, lossval,f1scr, lossbi, lossmask, name = fn_log):
    log = '\nEpoch: '+ str(epoch) + ' \tTraining Loss: '+ str(losstrain) + ' \tValidation Loss: ' + str(lossval) + ' \t f1-score: ' +str(f1scr) + '\tLoss Birad: ' + str(lossbi) + '\tLoss Mask: ' + str(lossmask)
    f = open(name, "a")
    f.write(log)
    f.close()
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
        loss_mask_val = AverageMeter()
        loss_bi_val = AverageMeter()

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
        torch.save(model.state_dict(), fn_model + str(epoch) + '.pt')

        bi_list = []
        bi_pred_list = []
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

                if torch.count_nonzero(mask) > 0:
                    loss_mask_v = criterion_mask(mask_pred, mask) #BCEloss
                    loss_mask_val.update(loss_mask_v.item())
                else:
                    loss_mask_v = 0
                    loss_mask_val.update(loss_mask_v)

                loss=(loss_bi_v+loss_mask_v)/2
                loss_val.update(loss.item())
                loss_bi_val.update(loss_bi_v.item())
                #f1-score
                bi_hat = torch.max(bi_hat,1)[1].to(device) #lúc training đã chuẩn hóa [1,2,3,4,5] thành [0,1,2,3,4]
                bi_list.append(birad)
                bi_pred_list.append(bi_hat)

                #Dice Score
                #dice_scr += dice_coeff(mask_pred, mask)
        
        bi_list, bi_pred_list = torch.cat(bi_list, 0), torch.cat(bi_pred_list, 0)
        f1_scr = f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'macro')
        #dice_scr = dice_scr/(len(bi_list))
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} ({:.6f}) \t f1-score {:.6f} \t dice-scr {:.6f}'.format(
            epoch, losses.avg, loss_val.val, loss_val.avg, f1_scr, 0 ))
        
        savelog(epoch,losses.avg, loss_val.avg,f1_scr, loss_bi_val.avg, loss_mask_val.avg, name = fn_log)
        trainLoss.append(losses.avg)
        validLoss.append(loss_val.avg)
        ## TODO: save the model if validation loss has decreased
        # if loss_val.avg < valid_loss_min:
        if f1_scr > valid_f1_max: 
            torch.save(model.state_dict(), fn_model + 'top.pt')
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
                torch.save(model.state_dict(), fn_model + 'last.pt')
                Lossplot(trainLoss,validLoss)
                exit(-1)

        if loss_val.avg < valid_loss_min:
            valid_loss_min = loss_val.avg
            torch.save(model.state_dict(), fn_model + 'loss.pt')
        
    return model, trainLoss, validLoss
# %%
#Setting model and moving to device
model_CNN = XNetModel(True).to(device)
if fn_premodel != None:
    model_CNN.load_state_dict(torch.load(fn_premodel))
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


train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = batchsize, num_workers = 2)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = False, batch_size = 1, num_workers = 2)

optimizer = optim.SGD(model_CNN.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.Adam(model_CNN.parameters(), lr=lr)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(train_dataloader), eta_min=lr/20, last_epoch=-1)
# %%
model_conv, trainLoss, validLoss=train_model(model_CNN, criterion_bi, criterion_mask, optimizer, lr_scheduler, train_dataloader, val_dataloader, batchsize = batchsize, n_epochs=n_epochs)
