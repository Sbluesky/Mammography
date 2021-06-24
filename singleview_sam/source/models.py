#%%
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class BaselineModel(nn.Module):
    def __init__(self, pretrained = True):
        super(BaselineModel, self).__init__()
        if pretrained is True:
            # self.model = EfficientNet.from_pretrained('efficientnet-b2')
            self.backbone = torch.nn.Sequential(*(list(models.resnet50(pretrained=pretrained).children())[:-1]))
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2048, 5)  
        self.fc2 = nn.Linear(2048, 4)

        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.backbone(x)
        x = x.reshape(bs,-1)
        bi = self.fc1(x)
        den = self.fc1(x)
        return bi, den

# %%
