#%%
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class Mixupmodel(nn.Module):
    def __init__(self, pretrained = True):
        super(Mixupmodel, self).__init__()
        if pretrained is True:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b2')
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc1 = nn.Linear(1408, 5)  
        
    def forward(self, x):
        bs = x.shape[0]
        x = self.backbone.extract_features(x) #bs, 1408, 48,32
        x = self.avgpool(x) #bs, 1408, 1, 1
        x = x.reshape(bs, -1)
        bi = self.fc1(x)
        return bi

#%%
