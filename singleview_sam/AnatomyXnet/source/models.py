#%%
import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

class XNetModel(nn.Module):
    def __init__(self, pretrained = True):
        super(XNetModel, self).__init__()
        if pretrained is True:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b2')
            #self.backbone = torch.nn.Sequential(*(list(models.resnet50(pretrained=pretrained).children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size = 1)
        ###Classification###
        self.fc1 = nn.Linear(1408, 5)  #for birad
        self.fc2 = nn.Linear(1408, 4)  #for density

        ### for Segmentation ### 
        self.conv1 = nn.Conv2d(in_channels = 1408, out_channels = 1024, kernel_size = (3,3), padding = (1,1))
        self.conv2 = nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = (3,3), padding = (1,1))
        self.conv3 = nn.Conv2d(in_channels =512, out_channels =1, kernel_size = (1,1))

        
    def forward(self, x):
        bs = x.shape[0]
        x = self.backbone.extract_features(x) #bs, 1408, 48, 32

        ###Classification###
        bi = self.avgpool(x) #bs, 1408, 1, 1
        bi = bi.reshape(bs, -1) #bs, 1408
        bi = self.fc1(bi) 

        ###Segmentation###
        mask = F.relu(self.conv1(x)) #bs,1024,48,32
        mask = F.relu(self.conv2(mask)) #bs, 512, 48,32
        mask = self.conv3(mask) #bs, 1,48, 32

        return bi, mask




# %%
48*32
# %%
