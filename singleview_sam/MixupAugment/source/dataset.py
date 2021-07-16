
#%%
from matplotlib.pyplot import get
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch

def createpath(path):
    tmppath = path.replace("/home/single4/mammo/mammo/data/updatedata/crop-images", "/media/single3/data1/samhuyendata/new-crop-image")
    tmppath = tmppath.replace("/home/single4/mammo/mammo/data/vietrad/cropimages_vietrad", "/media/single3/data1/samhuyendata/new-crop-image")
    return tmppath

def createmaskpath(mask_std,mask_id):
    return "/media/single3/data1/samhuyendata/new-crop-image/" + mask_std +"/" + mask_id + ".png"

def get_transform(type = '', height: int = 1024, width: int = 768):
    if type == 'tensor':
        return transforms.ToTensor()
    elif type == "resize":
        return transforms.Resize((height, width))
    else:
        return transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])

def get_birad():
    return {
            "BI-RADS 0": 0,
            "BI-RADS 1": 1,
            "BI-RADS 2": 2,
            "BI-RADS 3": 3,
            "BI-RADS 4": 4,
            "BI-RADS 5": 5,
        }

def get_dens():
    return {
            "DENSITY-A": 0,
            "DENSITY-B": 1,
            "DENSITY-C": 2,
            "DENSITY-D": 3,
        }

#%%
class OnlyMass(Dataset):
    def __init__(self,datamask , h, w, upper, lower):
        """
        [This class returns images of birad 3,4,5 with bounding boxes pixels multiplication with 1 and backgrounds pixels multiplication with random alpha]

        Args:
            datamask ([dataframe]]): [dataframe consists of bouding box & images information]
            h ([int]]): [height of transform image]
            w ([int]]): [width of transform image]]
            upper ([float]): [upder threshold of alpha]
            lower ([float]): [lower threshold of alpha]
        """
        self.datamask = datamask
        self.imgidmask = datamask[datamask.type == 'global'].reset_index(drop = True)
        self.h = h
        self.w = w
        self.upper = upper
        self.lower = lower
        self._birads_to_idxs = get_birad()
        self._densities_to_idxs = get_dens()
        self.tfms = get_transform(height = self.h, width =self.w)

    def __len__(self):
        return len(self.imgidmask)
    
    def __getitem__(self, index):
        #return image, birad
        info = self.imgidmask.iloc[index]
        image_id = info['image_id']
        imagespath = createmaskpath(info['study_id'], info['image_id'])
        image = Image.open(imagespath)
        wim, him = image.size
        laterality = info['laterality']
        birad = self._birads_to_idxs[info["birads"]]

        alpha = np.random.uniform(self.lower, self.upper)
        maskarr = np.full((him, wim), alpha) #create black image h, w
        
        imgdf = self.datamask[self.datamask['image_id'] == image_id]
        imgboxdf = imgdf[imgdf['box_label'].isin(['Discrete mass', 'Spiculated mass', 'Stellate mass'])] # take box dataframe image with 3 masses
        for i in imgboxdf.index: # assign 1 for mass coordinates
            maskarr[int(imgboxdf['y_min'][i].round()) : int(imgboxdf['y_max'][i].round()), \
                int(imgboxdf['x_min'][i].round()) : int(imgboxdf['x_max'][i].round())] =  1 
        
        image = image * maskarr
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        if laterality == "R": #flip R mask image
            image = transforms.functional.hflip(image) 
        image = self.tfms(image)
        
        sample = {"image": image, "birad": birad} 
        return sample

class OnlyImage(Dataset):
    

    def __init__(self, dataframe, h, w):
        self.dataframe = dataframe
        self.h = h
        self.w = w
        self._birads_to_idxs = get_birad()
        self.tfms = get_transform(height = self.h, width =self.w)

    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        #return image, birad, density
        info = self.dataframe.iloc[index]
        imagespath = createpath(info["path"])
        image = Image.open(imagespath).convert('RGB')
        laterality = info['laterality']
        birad = self._birads_to_idxs[info["birad"]]
        if laterality == "R":
            image = transforms.functional.hflip(image)
        image = self.tfms(image)
        sample = {"image": image, "birad": birad} 
        return sample
#%%
class Mixup(Dataset):
    def __init__(self, birad_df, mask_df, h, w):
        """[this class to mix bounding boxes from images of birad 3,4,5 into images of birad 1,2]

        Args:
            birad_df ([dataframe]]): [dataframe consists of images of birad 1,2]
            mask_df ([dataframe]]): [dataframe consists of bouding box & images information]
            h ([int]]): [height]
            w ([int]]): [width]]

        Flow:
            Input: image of birad 1 (image), image has bouding box (image_mask).
            Step 1: create 1 mask has pixel values in bouding box = 255, pixel values background = 0
            Step 2: Get bouding box: multiplicate mask with image_mask
            Step 3: Resize bounding box to same the shape with image of birad 1
            Step 4: multiplicate image with random alpha. And mix bouding box with image
        """
        self.birad_df = birad_df
        self.mask_df = mask_df[mask_df.type == 'local'].reset_index(drop = True)
        self.imgidmask = mask_df[mask_df.type == 'global'].reset_index(drop = True)
        self.h = h
        self.w = w
        self._birads_to_idxs = get_birad()
        self._densities_to_idxs = get_dens()

        #transform image
        self.tfms = get_transform(height = self.h, width = self.w)
        self.totensor = get_transform(type = "tensor")
        self.resize = get_transform(type = "resize",height = self.h, width = self.w)

    def __len__(self):
        return len(self.imgidmask)    
    
    def __getitem__(self, index):
        #get image in birad 1
        info_img = self.birad_df.iloc[index]
        imagespath = createpath(info_img["path"])
        image = Image.open(imagespath).convert('RGB')
        laterality = info_img['laterality']

        #transform image
        if laterality == "R":
            image = transforms.functional.hflip(image)
        image = self.tfms(image)
        alpha = np.random.uniform(0.0, 0.8)
        image = image * alpha #multiplicate background with random alpha

        #get mask
        info_mask = self.imgidmask.iloc[index]
        birad = self._birads_to_idxs[info_mask["birads"]]
        maskpath = createmaskpath(info_mask["study_id"], info_mask["image_id"])           
        image_mask = Image.open(maskpath).convert('RGB')
        laterality_m = info_mask['laterality']
        wim, him = image_mask.size

        #create mask
        maskarr = np.zeros((him, wim), dtype = np.uint8) 

        # transform mask image
        if laterality_m == "R":
            image_mask = transforms.functional.hflip(image_mask)
        #get bouding box
        imgdf = self.mask_df[self.mask_df['image_id'] == info_mask["image_id"]]
        for ind in imgdf.index:
            x_min, x_max = int(imgdf["x_min"][ind].round()), int(imgdf["x_max"][ind].round())
            y_min, y_max = int(imgdf["y_min"][ind].round()), int(imgdf["y_max"][ind].round())
            maskarr[y_min:y_max, x_min:x_max] = 255 #create mask have background = 0 and bouding box = 255
        image_mask = self.totensor(image_mask)
        maskarr = self.totensor(maskarr)
        maskarr = maskarr.expand(3, -1, -1) #expand from 1 channel to 3 channels

        image_mask = image_mask * maskarr #get pixel values in bounding box, background pixel values = 0
        image_mask = self.resize(image_mask) #resize image mask to have same shape with birad 1 image

        #mix bouding box with image of birad 1
        index = torch.nonzero(image_mask)
        for i in index:
            image[i[0], i[1], i[2]] = image_mask[i[0], i[1], i[2]]

        sample = {'image': image, 'birad': birad}

        return sample


# %%
