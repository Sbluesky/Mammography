#%%
import torch
import torch.nn as nn
from torchvision import models
import pretrainedmodels
import torch.nn.functional as F


#%%
class ImageWiseModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageWiseModel, self).__init__()
        if pretrained is True:
            self.backbone = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(512,256) 
        self.fc2 = nn.Linear(256,1) 
 
    def forward(self, left_CC, right_CC, left_MLO, right_MLO):
        bs_left_CC, _, _, _ = left_CC.shape #get left cc batchsize
        bs_right_CC, _, _, _ = right_CC.shape 
        bs_left_MLO,_,_,_ = left_MLO.shape #get mlo batchsize
        bs_right_MLO,_,_,_ = right_MLO.shape

        #Left-CC-view
        left_CC = self.backbone(left_CC)
        left_CC = left_CC.reshape(bs_left_CC,-1)
        left_CC = self.fc1(left_CC)
        Denleft_CC = self.fc2(left_CC) #density left cc view
        Bileft_CC = self.fc2(left_CC) #Birad left cc view

        #left-MLO-view
        left_MLO = self.backbone(left_MLO)
        left_MLO = left_MLO.reshape(bs_left_MLO,-1)
        left_MLO = self.fc1(left_MLO)
        Denleft_MLO = self.fc2(left_MLO) #density left mlo view
        Bileft_MLO = self.fc2(left_MLO) #Birad left mlo view

        #Right-CC-view
        right_CC = self.backbone(right_CC)
        right_CC = right_CC.reshape(bs_right_CC,-1)
        right_CC = self.fc1(right_CC)
        Denright_CC = self.fc2(right_CC) #density mlo view
        Biright_CC = self.fc2(right_CC) #birad mlo view

        #Right-MLO-view
        right_MLO = self.backbone(right_MLO)
        right_MLO = right_MLO.reshape(bs_right_MLO,-1)
        right_MLO = self.fc1(right_MLO)
        Denright_MLO = self.fc2(right_MLO) #density mlo view
        Biright_MLO = self.fc2(right_MLO) #birad mlo view

        #Average 
        L_bi = getAVG(Bileft_CC,Bileft_MLO)
        L_den = getAVG(Denleft_CC,Denleft_MLO)
        R_bi = getAVG(Biright_CC,Biright_MLO)
        R_den = getAVG(Denright_CC,Denright_MLO)
        return L_bi, L_den, R_bi,R_den

class BreastWiseModel(nn.Module):
    def __init__(self, pretrained=True):
        super(BreastWiseModel, self).__init__()
        if pretrained is True:
            self.backbone = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(512*2,512) 
        self.fc2 = nn.Linear(512,1) 
 
    def forward(self, left_CC, right_CC, left_MLO, right_MLO):
        bs_left_CC, _, _, _ = left_CC.shape #get left cc batchsize
        bs_right_CC, _, _, _ = right_CC.shape 
        bs_left_MLO,_,_,_ = left_MLO.shape #get mlo batchsize
        bs_right_MLO,_,_,_ = right_MLO.shape

        #Left-CC-view
        left_CC = self.backbone(left_CC) #(batchsize, 512,1,1)
        left_CC = left_CC.reshape(bs_left_CC,-1) #(batchsize, 512)
        

        #left-MLO-view
        left_MLO = self.backbone(left_MLO)
        left_MLO = left_MLO.reshape(bs_left_MLO,-1)
        
        #Right-CC-view
        right_CC = self.backbone(right_CC)
        right_CC = right_CC.reshape(bs_right_CC,-1)
        

        #Right-MLO-view
        right_MLO = self.backbone(right_MLO)
        right_MLO = right_MLO.reshape(bs_right_MLO,-1)
        

        #concat 
        Left = torch.cat((left_CC,left_MLO), dim =1) #(2,1024)
        Right = torch.cat((right_CC,right_MLO), dim =1) #(2,1024)
        
        #2 fully connected
        Left = self.fc1(Left)
        Left = F.relu(Left)
        L_bi = self.fc2(Left)
        L_den = self.fc2(Left)

        Right = self.fc1(Right)
        Right = F.relu(Right)
        R_bi = self.fc2(Right)
        R_den = self.fc2(Right)
        return L_bi, L_den, R_bi,R_den

class ViewWiseModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ViewWiseModel, self).__init__()
        if pretrained is True:
            self.backbone = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(512*2,512) 
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,1) 
        self.fc4 = nn.Linear(512,1)
 
    def forward(self, left_CC, right_CC, left_MLO, right_MLO):
        bs_left_CC, _, _, _ = left_CC.shape #get left cc batchsize
        bs_right_CC, _, _, _ = right_CC.shape 
        bs_left_MLO,_,_,_ = left_MLO.shape #get mlo batchsize
        bs_right_MLO,_,_,_ = right_MLO.shape

        #Left-CC-view
        left_CC = self.backbone(left_CC) #(batchsize, 512,1,1)
        left_CC = left_CC.reshape(bs_left_CC,-1) #(batchsize, 512)
        

        #left-MLO-view
        left_MLO = self.backbone(left_MLO)
        left_MLO = left_MLO.reshape(bs_left_MLO,-1)
        
        #Right-CC-view
        right_CC = self.backbone(right_CC)
        right_CC = right_CC.reshape(bs_right_CC,-1)
        

        #Right-MLO-view
        right_MLO = self.backbone(right_MLO)
        right_MLO = right_MLO.reshape(bs_right_MLO,-1)
        

        #concat 
        CC = torch.cat((left_CC,right_CC), dim =1) #(2,1024)
        MLO= torch.cat((left_MLO,right_MLO), dim =1) #(2,1024)
        
        #2 fully connected
        CC = self.fc1(CC)
        CC = F.relu(CC)
        L_bi_CC = self.fc2(CC)
        L_bi_CC = self.fc3(L_bi_CC)
        L_den_CC = self.fc4(CC)
        R_bi_CC = self.fc4(CC)
        R_den_CC = self.fc4(CC)

        MLO = self.fc1(MLO)
        MLO = F.relu(MLO)
        L_bi_MLO = self.fc2(MLO)
        L_bi_MLO = self.fc3(L_bi_MLO)
        L_den_MLO = self.fc4(MLO)
        R_bi_MLO = self.fc4(MLO)
        R_den_MLO = self.fc4(MLO)
        #print("R_bi_CC: ", R_bi_CC)
        #print("L_bi_CC",L_bi_CC)
        #Avg
        L_bi = getAVG(L_bi_CC, L_bi_MLO)
        L_den = getAVG(L_den_CC, L_den_MLO)
        R_bi = getAVG(R_bi_CC, R_bi_MLO)
        R_den = getAVG(R_den_CC, R_den_MLO)
        output = {
            "L_bi": L_bi, "L_den": L_den, "R_bi" : R_bi,"R_den" : R_den
        }
        return output

class ViewWiseModelv2(nn.Module):
    def __init__(self, pretrained=True):
        super(ViewWiseModelv2, self).__init__()
        self.backbone = ClsModel()
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch35res34pretrainedce_extraLCC_512_focal.pt"))
        self.LCC_backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch21res34pretrainedce_extraRCC_512_focal.pt"))
        self.RCC_backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch46res34pretrainedce_extraLMLO_512_focal.pt"))
        self.LMLO_backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch21res34pretrainedce_extraRMLO_512_focal.pt"))
        self.RMLO_backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(512*2,512) 
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,1) 
        self.fc4 = nn.Linear(512,1)
        
 
    def forward(self, left_CC, right_CC, left_MLO, right_MLO):
        bs_left_CC, _, _, _ = left_CC.shape #get left cc batchsize
        bs_right_CC, _, _, _ = right_CC.shape 
        bs_left_MLO,_,_,_ = left_MLO.shape #get mlo batchsize
        bs_right_MLO,_,_,_ = right_MLO.shape

        #Left-CC-view
        left_CC = self.LCC_backbone(left_CC) #(batchsize, 512,1,1)
        left_CC = left_CC.reshape(bs_left_CC,-1) #(batchsize, 512)

        #left-MLO-view
        left_MLO = self.LMLO_backbone(left_MLO)
        left_MLO = left_MLO.reshape(bs_left_MLO,-1)
        
        #Right-CC-view
        right_CC = self.RCC_backbone(right_CC)
        right_CC = right_CC.reshape(bs_right_CC,-1)
        

        #Right-MLO-view
        right_MLO = self.RMLO_backbone(right_MLO)
        right_MLO = right_MLO.reshape(bs_right_MLO,-1)
        

        #concat 
        CC = torch.cat((left_CC,right_CC), dim =1) #(2,1024)
        MLO= torch.cat((left_MLO,right_MLO), dim =1) #(2,1024)
        
        #2 fully connected
        CC = self.fc1(CC)
        CC = F.relu(CC)
        #Left CC
        L_bi_CC = F.relu(self.fc2(CC))
        L_bi_CC = self.fc3(L_bi_CC)
        L_den_CC = self.fc4(CC)
        #Right CC
        R_bi_CC = F.relu(self.fc2(CC))
        R_bi_CC = self.fc3(R_bi_CC)
        R_den_CC = self.fc4(CC)

        MLO = self.fc1(MLO)
        MLO = F.relu(MLO)
        #Left MLO
        L_bi_MLO = F.relu(self.fc2(MLO))
        L_bi_MLO = self.fc3(L_bi_MLO)
        L_den_MLO = self.fc4(MLO)
        #Right MLO       
        R_bi_MLO = F.relu(self.fc2(MLO))
        R_bi_MLO = self.fc3(R_bi_MLO)
        R_den_MLO = self.fc4(MLO)
        
        #Avg
        L_bi = getAVG(L_bi_CC, L_bi_MLO)
        L_den = getAVG(L_den_CC, L_den_MLO)
        R_bi = getAVG(R_bi_CC, R_bi_MLO)
        R_den = getAVG(R_den_CC, R_den_MLO)
        return L_bi, L_den, R_bi,R_den

class BackBone(nn.Module):
    def __init__(self, pretrained=True):
        super(BackBone, self).__init__()
        if pretrained is True:
            self.backbone = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(512,1) 
        self.fc2 = nn.Linear(512,1) 
 
    def forward(self, img):
        bs, _, _, _ = img.shape #get batchsize

        #Left-CC-view
        img = self.backbone(img)
        img = img.reshape(bs,-1)
        Bi = self.fc1(img)
        Den = self.fc2(img) #density left cc view 
        return Bi, Den

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

class ViewWiseModelv1(nn.Module):
    def __init__(self, pretrained = True):
        super(ViewWiseModelv1, self).__init__()
        self.backbone = ClsModel()
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch35res34pretrainedce_extraLCC_512_focal.pt"))
        self.LCC_backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch21res34pretrainedce_extraRCC_512_focal.pt"))
        self.RCC_backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch46res34pretrainedce_extraLMLO_512_focal.pt"))
        self.LMLO_backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.backbone.load_state_dict(torch.load("/home/single4/mammo/mammo/huyen/cls_single/bestmodel/bestepoch21res34pretrainedce_extraRMLO_512_focal.pt"))
        self.RMLO_backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(512*2,512) 
        self.fc2 = nn.Linear(512,1)
 
    def forward(self, left_CC, right_CC, left_MLO, right_MLO):
        bs_left_CC, _, _, _ = left_CC.shape #get left cc batchsize
        bs_right_CC, _, _, _ = right_CC.shape 
        bs_left_MLO,_,_,_ = left_MLO.shape #get mlo batchsize
        bs_right_MLO,_,_,_ = right_MLO.shape

        #Left-CC-view
        left_CC = self.LCC_backbone(left_CC) #(batchsize, 512,1,1)
        left_CC = left_CC.reshape(bs_left_CC,-1) #(batchsize, 512)

        #left-MLO-view
        left_MLO = self.LMLO_backbone(left_MLO)
        left_MLO = left_MLO.reshape(bs_left_MLO,-1)
        
        #Right-CC-view
        right_CC = self.RCC_backbone(right_CC)
        right_CC = right_CC.reshape(bs_right_CC,-1)
        

        #Right-MLO-view
        right_MLO = self.RMLO_backbone(right_MLO)
        right_MLO = right_MLO.reshape(bs_right_MLO,-1)
        

        #concat 
        CC = torch.cat((left_CC,right_CC), dim =1) #(2,1024)
        MLO= torch.cat((left_MLO,right_MLO), dim =1) #(2,1024)
        
        #2 fully connected
        CC = self.fc1(CC)
        CC = F.relu(CC)
        L_bi_CC = self.fc2(CC)
        L_den_CC = self.fc2(CC)
        R_bi_CC = self.fc2(CC)
        R_den_CC = self.fc2(CC)

        MLO = self.fc1(MLO)
        MLO = F.relu(MLO)
        L_bi_MLO = self.fc2(MLO)
        L_den_MLO = self.fc2(MLO)
        R_bi_MLO = self.fc2(MLO)
        R_den_MLO = self.fc2(MLO)

        #Avg
        L_bi = getAVG(L_bi_CC, L_bi_MLO)
        L_den = getAVG(L_den_CC, L_den_MLO)
        R_bi = getAVG(R_bi_CC, R_bi_MLO)
        R_den = getAVG(R_den_CC, R_den_MLO)
        return L_bi, L_den, R_bi,R_den



class JointModel(nn.Module):
    def __init__(self, pretrained=True):
        super(JointModel, self).__init__()
        if pretrained is True:
            self.backbone = torch.nn.Sequential(*(list(models.resnet34(pretrained=pretrained).children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(512*4,512) 
        self.fc2 = nn.Linear(512,1) 
 
    def forward(self, left_CC, right_CC, left_MLO, right_MLO):
        bs_left_CC, _, _, _ = left_CC.shape #get left cc batchsize
        bs_right_CC, _, _, _ = right_CC.shape 
        bs_left_MLO,_,_,_ = left_MLO.shape #get mlo batchsize
        bs_right_MLO,_,_,_ = right_MLO.shape

        #Left-CC-view
        left_CC = self.backbone(left_CC) #(batchsize, 512,1,1)
        left_CC = left_CC.reshape(bs_left_CC,-1) #(batchsize, 512)
        

        #left-MLO-view
        left_MLO = self.backbone(left_MLO)
        left_MLO = left_MLO.reshape(bs_left_MLO,-1)
        
        #Right-CC-view
        right_CC = self.backbone(right_CC)
        right_CC = right_CC.reshape(bs_right_CC,-1)
        

        #Right-MLO-view
        right_MLO = self.backbone(right_MLO)
        right_MLO = right_MLO.reshape(bs_right_MLO,-1)
        

        #concat 
        Joint = torch.cat((left_CC,left_MLO,right_CC,right_MLO), dim =1) #(2,512*4)
        
        #2 fully connected
        Joint = self.fc1(Joint)
        Joint = F.relu(Joint)
        L_bi = self.fc2(Joint)
        L_den = self.fc2(Joint)
        R_bi = self.fc2(Joint)
        R_den = self.fc2(Joint)
        return L_bi, L_den, R_bi,R_den

def getMAX(ten1, ten2):
    result = []
    for ind in range(ten1.shape[0]):
        result.append(torch.max(ten1[ind],ten2[ind]))
    return torch.stack(result)

def getAVG(ten1, ten2):
    result = []
    for ind in range(ten1.shape[0]):
        result.append((ten1[ind] + ten2[ind])/2)
    return torch.stack(result)

# model = ViewWiseModelv1(True) 
# from torchviz import make_dot
# x1 = torch.randn(1, 3, 224, 224, dtype=torch.float, requires_grad=False) 
# x2 = torch.randn(1, 3, 224, 224, dtype=torch.float, requires_grad=False) 
# x3 = torch.randn(1, 3, 224, 224, dtype=torch.float, requires_grad=False) 
# x4 = torch.randn(1, 3, 224, 224, dtype=torch.float, requires_grad=False) 
# # print(x[0], x[1], x[2], x[3])
# out = model(x1, x2, x3, x4)
# make_dot(out, params=dict(model.named_parameters())).render("rnn_torchviz1", format="png")