import torch
import torch.nn as nn
from torchvision import models
import pretrainedmodels
import torch.nn.functional as F

"""
class MammoModel1(nn.Module):
    def __init__(self, pretrained=True):
        super(MammoModel1, self).__init__()
        #self.backbone = models.resnet34(pretrained=pretrained).cuda()
        if pretrained is True:
            self.backbone = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-2])) #xóa layer fully connected cuối

        self.conv1 = nn.Conv2d(512, 512, (3,3))
        self.AvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.layer1 = nn.Linear(512,256)
        self.layer2 = nn.Linear(512,6) #for birad
        self.layer3 = nn.Linear(256,4) #for density
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bs,_,_,_ = x.shape
        x = self.backbone(x) #shape = (4,512,1,1)
        x1 = nn.functional.adaptive_avg_pool2d(x, 1).reshape(bs, -1) #shape = (4,512)
        #print(x.shape)
        label1 = self.conv1(x)
        #print(label1.shape)
        label1 = self.AvgPool(label1)
        label1 = label1.reshape(bs,-1)
        #print(label1.shape)
        label1 = self.layer2(label1)
        #label1 = self.softmax(label1)
        label2 = self.AvgPool(x)
        label2 = label2.reshape(bs,-1)
        label2 = self.layer1(label2)
        label2 = self.layer3(label2)
        #label2 = self.softmax(label2)
        return label1, label2
"""
"""
class MammoModel(nn.Module):
    def __init__(self, pretrained=True):
        super(MammoModel, self).__init__()
        if pretrained is True:
            self.backbone = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-1]))
        self.layer1 = nn.Linear(512,6) #for birad
        self.layer2 = nn.Linear(512,256) 
        self.layer3 = nn.Linear(256,4) #for density
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.backbone(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.layer1(x)
        label1 = self.softmax(label1)
        label2 = self.layer2(x)
        label2 = self.layer3(label2)
        label2 = self.softmax(label2) 
        return label1, label2
"""

class MammoModel(nn.Module):
    def __init__(self, pretrained):
        super(MammoModel, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        self.fc1 = nn.Linear(512, 6)  #For birad class
        self.fc2 = nn.Linear(512, 4)   #For density class

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        # x = self.model(x)
        # print(x.shape)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        bi = self.fc1(x)
        den= self.fc2(x)
        return bi, den

