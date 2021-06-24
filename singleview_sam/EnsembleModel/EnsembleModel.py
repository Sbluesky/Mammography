import torch
import torch.nn as nn
from torchvision import models
import pretrainedmodels
import torch.nn.functional as F



class SingleModel(nn.Module):
    def __init__(self, pretrained=True):
        super(SingleModel, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.layer1 = nn.Linear(512,1) #for birad
 
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.backbone(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        #birad
        label1 = self.layer1(x)

        #density
        #label2 = self.layer2(x)
        return label1#, label2
"""
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3, model4, model5):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model1 = model1
        self.model1 = model1
        self.model1 = model1
        self.model1 = model1
        self.layer1 = nn.Linear(5,1) #for birad
        self.layer2 = nn.Linear(5,1) 
        self.leakyrelu = nn.LeakyReLU(0.1)
 
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.backbone(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        #birad
        label1 = self.layer1(x)

        #density
        label2 = self.layer2(x)
        return label1, label2
"""