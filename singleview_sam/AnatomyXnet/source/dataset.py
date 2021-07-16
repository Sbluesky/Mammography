
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch

#Dataloader
#Dataloader
def createpath(path):
    tmppath = path.replace("single4", "single3")
    tmppath = tmppath.replace("updatedata/crop-images", "images")
    tmppath = tmppath.replace("cropimages_vietrad", "images")
    return tmppath
class MammoLabelMass(Dataset):
    

    def __init__(self, dataframe, datamask, h, w):
        self.dataframe = dataframe
        self.datamask = datamask # /home/single3/mammo/mammo/data/updatedata/csv/mass-extratrain.csv
        self.h = h
        self.w = w
        self._birads_to_idxs = {
            "BI-RADS 0": 0,
            "BI-RADS 1": 1,
            "BI-RADS 2": 2,
            "BI-RADS 3": 3,
            "BI-RADS 4": 4,
            "BI-RADS 5": 5,
        }
        self._densities_to_idxs = {
            "DENSITY-A": 0,
            "DENSITY-B": 1,
            "DENSITY-C": 2,
            "DENSITY-D": 3,
        }
        self.tfms = transforms.Compose([transforms.Resize((self.h, self.w)), transforms.ToTensor()])

    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        #return image, birad, density
        info = self.dataframe.iloc[index]
        image_id = info['image_id']
        imagespath = createpath(info["path"])
        image = Image.open(imagespath).convert('RGB')
        wim, him = image.size
        laterality = info['laterality']
        birad = self._birads_to_idxs[info["birad"]]
        density = self._densities_to_idxs[info["density"]]
        maskarr = np.zeros((him, wim), dtype = np.uint8) #create black image h, w
        mask = Image.fromarray(np.uint8(maskarr)) #convert mask to PIL image
        mask_gt = Image.fromarray(np.uint8(maskarr))
        if laterality == "R":
            image = transforms.functional.hflip(image)
        image = self.tfms(image)
        # transform mask image
        mask = transforms.Compose([transforms.Resize((32,24)), transforms.ToTensor()])(mask) #48, 32
        mask_gt = transforms.Compose([transforms.ToTensor()])(mask_gt)
        #create mask for bounding boxes in image
        if image_id in self.datamask['image_id'].unique():
            imgdf = self.datamask[self.datamask['image_id'] == image_id]
            imgboxdf = imgdf[imgdf['box_label'].isin(['Discrete mass', 'Spiculated mass', 'Stellate mass'])] # take box dataframe image with 3 masses
            for i in imgboxdf.index: # assign 1 for mass coordinates
                maskarr[int(imgboxdf['y_min'][i].round()) : int(imgboxdf['y_max'][i].round()), \
                    int(imgboxdf['x_min'][i].round()) : int(imgboxdf['x_max'][i].round())] =  1
            mask = Image.fromarray(np.uint8(maskarr)) #convert mask to PIL image
            if laterality == "R": #flip R mask image
                mask = transforms.functional.hflip(mask) 
            # transform mask image
            mask_gt = transforms.Compose([transforms.Resize((self.h, self.w)),transforms.ToTensor()])(mask)
            mask = transforms.Compose([transforms.Resize((32,24)), transforms.ToTensor()])(mask)
            for ind in torch.nonzero(mask):
                mask[ind[0],ind[1], ind[2]] = 1
            for ind in torch.nonzero(mask_gt):
                mask_gt[ind[0],ind[1], ind[2]] = 1
        sample = {"image": image, "birad": birad, "density": density, "mask":mask, 'mask_gt': mask_gt} # 'mask_gt': mask_gt
        return sample

#%%
