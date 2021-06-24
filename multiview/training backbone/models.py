#%%
import torch
import torch.nn as nn
from torchvision import models
import pretrainedmodels
import torch.nn.functional as F

class ClsModel(nn.Module):
    def __init__(self, pretrained = True):
        super(ClsModel, self).__init__()
        if pretrained is True:
            # self.model = EfficientNet.from_pretrained('efficientnet-b2')
            self.backbone = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-1]))
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 5)  #For birad class
        self.fc2 = nn.Linear(512, 4)   #For density class
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        # x = self.model.extract_features(x)
        # x = self.avgpool(x)
        # x = self.model(x)
        x = self.backbone(x)
        x = x.reshape(bs,-1)
        bi = self.fc1(x)
        den= self.fc2(x)
        return bi, den

class StackingModel(nn.Module):
    def __init__(self, pretrained = True):
        super(StackingModel, self).__init__()
        self.backbone = ClsModel()
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch35res34pretrainedce_extraLCC_512_focal.pt"))
        self.LCC_backbone = self.backbone
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch21res34pretrainedce_extraRCC_512_focal.pt"))
        self.RCC_backbone = self.backbone
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch46res34pretrainedce_extraLMLO_512_focal.pt"))
        self.LMLO_backbone = self.backbone
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch21res34pretrainedce_extraRMLO_512_focal.pt"))
        self.RMLO_backbone = self.backbone
 
    def forward(self, left_CC, right_CC, left_MLO, right_MLO):

        #Left-CC-view
        left_CC_bi, left_CC_den = self.LCC_backbone(left_CC) #(batchsize, 512,1,1)
        left_CC_bi = torch.max(left_CC_bi,1)[1] + 1
        left_CC_den = torch.max(left_CC_den,1)[1] 

        #left-MLO-view
        left_MLO_bi, left_MLO_den = self.LMLO_backbone(left_MLO)
        left_MLO_bi = torch.max(left_MLO_bi,1)[1] + 1
        left_MLO_den = torch.max(left_MLO_den,1)[1]

        #Right-CC-view
        right_CC_bi, right_CC_den = self.RCC_backbone(right_CC)
        right_CC_bi = torch.max(right_CC_bi,1)[1] + 1
        right_CC_den = torch.max(right_CC_den,1)[1]
        
        #Right-MLO-view
        right_MLO_bi, right_MLO_den = self.RMLO_backbone(right_MLO)
        right_MLO_bi = torch.max(right_MLO_bi,1)[1] + 1
        right_MLO_den = torch.max(right_MLO_den,1)[1]
        
        L_bi = BiradVotingStrategies(left_CC_bi,left_MLO_bi)
        L_den = DensityVotingStrategies(left_CC_den, left_MLO_den)
        R_bi = BiradVotingStrategies(right_CC_bi,right_MLO_bi)
        R_den = DensityVotingStrategies(right_CC_den, right_MLO_den)
        """
        output = {
            "LCCbi" : left_CC_bi, "LCCden" : left_CC_den,\
            "LMLObi" : left_MLO_bi, "LMLOden" : left_MLO_den,\
            "RCCbi" : right_CC_bi, "RCCden" : right_CC_den,\
            "RMLObi" : right_MLO_bi, "RMLOden": right_MLO_den
        }
        """
        output = {
            "L_bi" : L_bi, "L_den" : L_den,\
            "R_bi" : R_bi, "R_den" : R_den,\
            
        }
        return output
"""
def BiradVotingStrategies(CC_bi, MLO_bi):
    return torch.max(CC_bi, MLO_bi)
"""
def BiradVotingStrategies(CC_bi, MLO_bi):
    bi = []
    for ind in range(CC_bi.shape[0]):
        if (CC_bi[ind] == 4) | (MLO_bi[ind] == 4):
            bi.append(torch.tensor(4))
        elif (CC_bi[ind] == 3) | (MLO_bi[ind] == 3):
            bi.append(torch.tensor(3))
        elif (CC_bi[ind] == 5) | (MLO_bi[ind] == 5):
            bi.append(torch.tensor(5))
        elif (CC_bi[ind] == 2) & (MLO_bi[ind] == 2):
            bi.append(torch.tensor(2))
        else:
            bi.append(torch.tensor(1))
    return torch.stack(bi)


def DensityVotingStrategies(CC_den, MLO_den):
    den = []
    for ind in range(CC_den.shape[0]):
        if (CC_den[ind] == 1) | (MLO_den[ind] == 1):
            den.append(torch.tensor(1))
        elif (CC_den[ind] == 0) | (MLO_den[ind] == 0):
            den.append(torch.tensor(0))
        elif (CC_den[ind] == 3) & (MLO_den[ind] == 3):
            den.append(torch.tensor(3))
        else:
            den.append(torch.tensor(2))
    return torch.stack(den)

