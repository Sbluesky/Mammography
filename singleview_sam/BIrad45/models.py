import torch
import torch.nn as nn
from torchvision import models
import pretrainedmodels
import torch.nn.functional as F


class MammoModel(nn.Module):
    def __init__(self, pretrained=True):
        super(MammoModel, self).__init__()
        if pretrained is True:
            self.backbone = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #self.layer3 = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[-4]))
        #self.layer4 = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[-3]))
        self.layer1 = nn.Linear(512,1) #for birad
        self.layer2 = nn.Linear(512,1) 
        #self.layer3 = nn.Linear(256,4) #for density
        #self.leakyrelu = nn.LeakyReLU(0.1)
 
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.backbone(x)
        #x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        #birad
        #print(label1.shape)
        x = x.reshape(bs,-1)
        label1 = self.layer1(x)
        #label1 = self.leakyrelu(label1)

        #density
        label2 = self.layer2(x)
        #label2 = self.leakyrelu(label2)
        return label1, label2
