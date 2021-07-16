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
from models import  Y_Net
from dataset import MammoLabelMask
from compute import AverageMeter
from termcolor import cprint
from sklearn.metrics import  f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() 
                                  else "cpu")

#parameters:
fn_traindf = "/home/single3/mammo/mammo/data/csv/newsingletrain.csv"
fn_valdf = "/home/single3/mammo/mammo/data/csv/newsinglevalid.csv"
fn_maskdf = "/home/single3/mammo/mammo/data/csv/mass-newcrop-train.csv"
fn_maskvldf = "/home/single3/mammo/mammo/data/csv/mass-newcrop-valid.csv"
fn_fig = "/home/single3/mammo/mammo/sam/singleview_sam/Y-NET/models/"
fn_log = "/home/single3/mammo/mammo/sam/singleview_sam/Y-NET/models/YNet-1024x768v1.txt"
fn_model = "/home/single3/mammo/mammo/sam/singleview_sam/Y-NET/models/YNet-1024x768-v1-"
fn_premodel = "/home/single3/mammo/mammo/sam/singleview_sam/Y-NET/models/YNet-1024x768-v1-last.pt"
criterion_bi = nn.CrossEntropyLoss().to(device)
criterion_mask = nn.BCEWithLogitsLoss().to(device)
lr = 1e-3
n_epochs = 15
batchsize = 8
num_workers = 2
freeze = False #param to freeze backbone
optimize = 'SGD'

#%%
traindf = pd.read_csv(fn_traindf)
valdf = pd.read_csv(fn_valdf)
maskdf = pd.read_csv(fn_maskdf)
maskValiddf = pd.read_csv(fn_maskvldf)
valdf = valdf[valdf["label_birad"] !=0 ]

#%%
train_dl = MammoLabelMask(traindf, maskdf, h = 1024, w = 768, h_m = 256, w_m = 192 ) 
val_dl = MammoLabelMask(valdf, maskValiddf, h = 1024, w = 768,  h_m = 256, w_m = 192)

# %%
#Setting model and moving to device
model_CNN = Y_Net().to(device)
if fn_premodel != None:
    model_CNN.load_state_dict(torch.load(fn_premodel))
#%%
#freeze backbone: list(model_CNN.children()) #len = 8
if freeze == True:
    ct = 0
    for child in model_CNN.children():
        if ct < 5:
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
        ct+=1


train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = batchsize, num_workers = num_workers)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = False, batch_size = 1, num_workers = num_workers)

if optimize == 'SGD':
    optimizer = optim.SGD(model_CNN.parameters(), lr=lr, momentum=0.9)
else:
    optimizer = optim.Adam(model_CNN.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(train_dataloader), eta_min=lr/20, last_epoch=-1)

#%%
#plt.imshow(torchvision.utils.make_grid(val_dl[4]['mask']).permute(1, 2, 0))
#%%
def Lossplot(trainLoss, validLoss, name = 'loss_fig.png', folder = fn_fig):
    plt.plot(trainLoss)
    plt.plot(validLoss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(folder+name, facecolor = "w")
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
    count = 0

    for epoch in range(0, n_epochs):
        
        losses = AverageMeter()
        loss_val = AverageMeter()
        loss_bi_val = AverageMeter()
        loss_mask_val = AverageMeter()
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(tqdm(train_dataloader)):
            with torch.cuda.amp.autocast(enabled=use_amp):
                if epoch == 0: 
                    optimizer.param_groups[0]['lr'] = lr*(batch_idx+1)/len(train_dataloader)
                # importing data and moving to GPU
                img = sample_batched['image'].to(device)
                mask = sample_batched['mask'].to(device)
                birad = sample_batched['birad'].to(device)
                birad = birad - 1

                #model predict
                bi_hat, mask_pred = model(img)
                loss_bi = criterion_bi(bi_hat, birad)
                if torch.count_nonzero(mask) > 0:
                    loss_mask = criterion_mask(mask_pred, mask)
                else:
                    loss_mask = 0
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
                    loss_mask_v = criterion_mask(mask_pred, mask)
                    loss_mask_val.update(loss_mask_v.item(), batchsize)
                else:
                    loss_mask_v = 0
                    loss_mask_val.update(loss_mask_v, batchsize)

                loss=(loss_bi_v+loss_mask_v)/2
                loss_val.update(loss.item(), batchsize)
                loss_bi_val.update(loss_bi_v.item(), batchsize)
                

                #f1-score
                bi_hat = torch.max(bi_hat,1)[1].to(device) 
                bi_list.append(birad)
                bi_pred_list.append(bi_hat)

        
        bi_list, bi_pred_list = torch.cat(bi_list, 0), torch.cat(bi_pred_list, 0)
        f1_scr = f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'macro')

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} ({:.6f}) \t f1-score {:.6f} \tLoss Birad {:.6f} \tLoss Mask: {:.6f}'.format(
            epoch, losses.avg, loss_val.val, loss_val.avg, f1_scr, loss_bi_val.avg, loss_mask_val.avg ))
        
        #save log
        savelog(epoch,losses.avg, loss_val.avg,f1_scr, loss_bi_val.avg, loss_mask_val.avg, name = fn_log)
        trainLoss.append(losses.avg)
        validLoss.append(loss_val.avg)
        ## TODO: save the model if validation loss has decreased
        if f1_scr > valid_f1_max: 
            torch.save(model.state_dict(), fn_model + 'top.pt')
            print('Validation f1 increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_f1_max,
            f1_scr))
            valid_f1_max = f1_scr
            count = 0
        else:
            count +=1
            cprint(f'EarlyStopping counter: {count} out of 5', 'yellow')
            if count == 5 or epoch == n_epochs - 1:
                cprint('Early Stop..', 'red')
                torch.save(model.state_dict(), fn_model + 'last.pt')
                Lossplot(trainLoss,validLoss,'efb2_full_1024x768.png')
                exit(-1)
        
        if loss_val.avg < valid_loss_min:
            valid_loss_min = loss_val.avg
            torch.save(model.state_dict(), fn_model + 'loss.pt')
    return model

model_conv = train_model(model_CNN, criterion_bi, criterion_mask, optimizer, lr_scheduler, train_dataloader, val_dataloader, batchsize = batchsize, n_epochs=n_epochs)
