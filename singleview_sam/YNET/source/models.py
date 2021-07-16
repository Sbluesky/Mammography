#%%
from typing_extensions import Concatenate
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
device = torch.device("cuda:1" if torch.cuda.is_available() 
                                  else "cpu")

#%%
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """[autocrop function]
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging between levels/blocks is possible.
    """
    if encoder_layer.shape[2:] != decoder_layer[2:]: #shape: (batchsize, channel, height, weight)
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4: #2D    
            encoder_layer = encoder_layer[
                             :,
                             :,
                             ((ds[0] - es[0]) // 2) : ((ds[0] + es[0]) // 2), 
                             ((ds[1] - es[1]) // 2) : ((ds[1] + es[1]) // 2)
                            ]
    return encoder_layer
class concatenate(nn.Module):
    def __init__(self):
        super(concatenate, self).__init__()
    
    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2),1)

        return x


def get_up_layer(in_channels, 
                 out_channels,
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 padding: int =1,
                 output_padding: int = 1):
    return nn.ConvTranspose2d(in_channels, out_channels,kernel_size = kernel_size, stride = stride, dilation = dilation, output_padding = output_padding, padding = padding)

def get_conv_layer(in_channels, 
                 out_channels,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)

def get_activation(activation: str):
    if activation =='relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope = 0.1)

def get_normalization(normalization: str, num_channels: int):
    if normalization == "batch":
        return nn.BatchNorm2d(num_channels)
    elif normalization == "instance":
        return nn.InstanceNorm2d(num_channels)
    
class UpBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 up_mode: str = 'transposed'):
        super(UpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.activation = activation
        self.up_mode = up_mode
        self.padding = 1

        #upConv/Upsamle layer
        self.upconv = get_up_layer(self.in_channels, self.out_channels)

        #conv layers
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size = 1, stride=1, padding=0)
        self.conv1 = get_conv_layer(2*self.out_channels, self.out_channels, kernel_size = 3, stride = 1, padding = self.padding) #for merged_layer
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size = 3, stride =1 , padding = self.padding)

        #batchnorm layers
        self.norm = get_normalization(normalization = self.normalization, num_channels = self.out_channels)

        #activation layers
        self.act = get_activation(self.activation)

        #concatenate layer
        self.concat = concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """[Forward pass]

        Args:
            encoder_layer ([tensor]): [connection from the encoder pathway]
            decoder_layer ([tensor]): [Tensor from the decoder pathway (to be up sampling)]
        """
        #print('decoder_layer: ', decoder_layer.shape)
        #print('encoder_layer: ', encoder_layer.shape)
        up_layer = self.upconv(decoder_layer) #up sampling
        #print('after up sampling decoder_layer: ', up_layer.shape)
        cropped_encoder_layer = autocrop(encoder_layer, up_layer)
        #print('after crop encoder: ', cropped_encoder_layer.shape)
        if self.up_mode != 'transposed':
            #need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer)
        up_layer = self.act(up_layer)
        up_layer = self.norm(up_layer)

        merged_layer = self.concat(up_layer, cropped_encoder_layer)
        #print("after merged layer: ", merged_layer.shape)
        y = self.conv1(merged_layer)
        y = self.act(y)
        y = self.norm(y)

        y = self.conv2(y)
        y = self.act(y)
        y = self.norm(y)
        return y

class Y_Net(nn.Module):
    def __init__(self):
        super(Y_Net, self).__init__()

        #Dow-sampling
        self.backbone = EfficientNet.from_pretrained('efficientnet-b2')
        self.block01 = torch.nn.Sequential(*(list(self.backbone.children())[:2])) #Conv2dStaticPadding + BatchNorm shape(bs,32,512,384)
        #Module list #Down Sampling
        self.block2 = torch.nn.Sequential(*(list(self.backbone.children())[2])) #Module list, out shape: (bs,352,32,24) 
        self.block2_0_2 = torch.nn.Sequential(*(list(self.block2.children())[:3]))
        self.block2_3_5 = torch.nn.Sequential(*(list(self.block2.children())[3:6]))
        self.block2_6_8 = torch.nn.Sequential(*(list(self.block2.children())[6:9]))
        self.block2_9_16 = torch.nn.Sequential(*(list(self.block2.children())[9:17]))
        self.block2_17_22 = torch.nn.Sequential(*(list(self.block2.children())[17:]))

        self.block3456 = torch.nn.Sequential(*(list(self.backbone.children())[3:7])) #COnv2dStaticPadding (bs, 1408,32,24) + BatchNorm, AvgPool, DropOut: (bs, 1408, 1, 1)

        #Task classification
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc1 = nn.Linear(1408, 5)  #2048 resnet50 #for birad

        #Task segmentation
        self.up_samp1 = UpBlock(in_channels = 352, out_channels=88)
        self.up_samp2 = UpBlock(in_channels = 88, out_channels=48)
        self.up_samp3 = UpBlock(in_channels = 48, out_channels= 24)
        self.convseg = nn.Conv2d(in_channels =24, out_channels =1, kernel_size = (1,1))

        
    def forward(self, x):
        bs = x.shape[0]
        #print("x shape: ",x.shape) #bs,3,1024,768
        #Classification #Down Sampling
        down1 = self.block01(x) 
        #print("(DOWN) After block01: ",down1.shape) #bs,32,512,384
        down2 = self.block2_0_2(down1)
        #print("(DOWN) After block2_0_2: ",down2.shape) #bs,24,256, 192
        down3 = self.block2_3_5(down2)
        #print("(DOWN) After block2_3_5: ",down3.shape) #bs,48,128,96
        down4 = self.block2_6_8(down3)
        #print("(DOWN) After block2_6_8: ",down4.shape) #bs, 88, 64,48
        down5 = self.block2_9_16(down4)
        #print("(DOWN) After block2_9_16: ",down5.shape) #bs, 208, 32,24
        down6 = self.block2_17_22(down5)
        #print("(DOWN) After block2_17_22: ",down6.shape) #bs, 352, 32,24

        down7 = self.block3456(down6)
        #print("After block3456: ",down7.shape) #bs, 1408,1,1
        bi = down7.reshape(bs,-1)
        bi = self.fc1(bi)

        
        #Segmentation #Upsampling
        up1 = self.up_samp1(encoder_layer = down4, decoder_layer = down6)
        #print("(UP) 1: ", up1.shape) #[1, 88, 64, 48]
        up2 = self.up_samp2(encoder_layer = down3, decoder_layer = up1)
        #print("(UP) 2: ", up2.shape) #[1, 48, 128, 96]
        up3 = self.up_samp3(encoder_layer = down2, decoder_layer = up2)
        #print("(UP) 3: ", up3.shape) #[1, 24, 256, 192]
    
        mask = self.convseg(up3)
        return bi, mask








# %%
"""
from PIL import Image
from torchvision import transforms
from torchsummary import summary

filename = '/home/single3/mammo/mammo/data/updatedata/crop-images/1.2.410.200010.1063715.6294.840813.852270.852270/1.3.12.2.1107.5.12.7.5054.30000019102200431821800000007.png'
tfms1 = transforms.Compose([
    transforms.Resize((1024, 768)), 
    transforms.ToTensor()])
img = Image.open(filename).convert('RGB')
img = tfms1(img)
img = img.reshape(1,3,1024,768)
model = Y_Net()
bi, mask = model(img)
print("BIRAD: ",bi)
print("MASK: ", mask)
#summary = summary(model, (1,3, 1024, 768))
"""
