#%%
# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
import matplotlib.pyplot as plt  #For representation#For model building
import torch
from torchvision import transforms
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from models import  BaselineModel
from dataset import MultiClassMammo
from termcolor import cprint
from sklearn.metrics import  f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() 
                                  else "cpu")


#%%
traindf = pd.read_csv("/home/single1/BACKUP/SamHUyen/mammo/data/updatedata/csv/singleview-extratrain.csv")
valdf = pd.read_csv("/home/single1/BACKUP/SamHUyen/mammo/data/updatedata/csv/singleview-valid.csv")
valdf = valdf[valdf["label_birad"] !=0 ]
traindf["label_birad"] = traindf["label_birad"] - 1
valdf["label_birad"] = valdf["label_birad"] - 1
#%%
#Augmentation
tfms1 = transforms.Compose([
    transforms.Resize((1024, 512)), 
    transforms.ToTensor()])


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
    plt.savefig('/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/TrainingBaseline/models'+name, facecolor = "w")
    plt.show()
    

#%%
def train_model(model, criterion, optimizer, lr_scheduler, train_dataloader, val_dataloader, n_epochs=50, batchsize = 8):
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
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(tqdm(train_dataloader)):
            if epoch == 0: 
                optimizer.param_groups[0]['lr'] = lr*(batch_idx+1)/len(train_dataloader)
            # importing data and moving to GPU
            img = sample_batched['image'].to(device)

            birad, density = sample_batched['bi'].to(device),\
                             sample_batched['den'].to(device)

            #model predict
            bi_hat, den_hat = model(img)

            loss_bi = criterion(bi_hat, birad)
            loss_den = criterion(den_hat, density)
            loss=(loss_bi+loss_den)/2
            losses.update(loss.item(), batchsize) 
            
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

        bi_list = []
        den_list = []
        bi_pred_list = []
        den_pred_list = []
        model.eval()
        for batch_idx, sample_batched in enumerate(tqdm(val_dataloader)):
            img = sample_batched['image'].to(device)

            birad, density = sample_batched['bi'].to(device),\
                             sample_batched['den'].to(device)

            #model predict
            bi_hat, den_hat = model(img)

            loss_bi_v = criterion(bi_hat, birad)
            loss_den_v = criterion(den_hat, density)

            loss=(loss_bi_v+loss_den_v)/2
            loss_val.update(loss.item(), batchsize)

            #f1-score
            bi_hat = torch.max(bi_hat,1)[1].to(device) #lúc training đã chuẩn hóa [1,2,3,4,5] thành [0,1,2,3,4]
            den_hat = torch.max(den_hat,1)[1].to(device)
            bi_list.append(birad)
            den_list.append(density)
            bi_pred_list.append(bi_hat)
            den_pred_list.append(den_hat)
        
        bi_list,den_list = torch.cat(bi_list, 0),torch.cat(den_list, 0)
        bi_pred_list,den_pred_list = torch.cat(bi_pred_list, 0),torch.cat(den_pred_list, 0)
        f1_scr = f1_score(bi_list.cpu(), bi_pred_list.cpu(), average = 'macro')
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} ({:.6f}) \t f1-score {:.6f} '.format(
            epoch, losses.avg, loss_val.val, loss_val.avg, f1_scr ))
        
        trainLoss.append(losses.avg)
        validLoss.append(loss_val.avg)
        ## TODO: save the model if validation loss has decreased
        if loss_val.avg < valid_loss_min:
            torch.save(model.state_dict(), '/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/TrainingBaseline/models/baseline_1024x512_top.pt')
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
                torch.save(model.state_dict(), '/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/TrainingBaseline/models/baseline_1024x512_last.pt')
                Lossplot(trainLoss,validLoss,'baseline_1024x512.png')
                exit(-1)
    # return trained model
    return model, trainLoss, validLoss
# %%
#Setting model and moving to device
model_CNN = BaselineModel(True).to(device)
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
criterion_multioutput = nn.CrossEntropyLoss().cuda()
#params
lr = 1e-3
n_epochs = 50
batchsize = 8

train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, batch_size = batchsize, num_workers = 2)
val_dataloader = torch.utils.data.DataLoader(val_dl, shuffle = False, batch_size = int(batchsize/2), num_workers = 2)

optimizer = optim.SGD(model_CNN.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.Adam(model_CNN.parameters(), lr=lr)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(train_dataloader), eta_min=lr/100, last_epoch=-1)
# %%
model_conv, trainLoss, validLoss=train_model(model_CNN, criterion_multioutput, optimizer, lr_scheduler, train_dataloader, val_dataloader, batchsize = batchsize, n_epochs=n_epochs)
torch.save(model_conv.state_dict(), '/home/single1/BACKUP/SamHUyen/mammo/sam/singleview_sam/TrainingBaseline/models/baseline_1024x512_last.pt')
Lossplot(trainLoss,validLoss,'baseline_1024x512.png')
# %%
