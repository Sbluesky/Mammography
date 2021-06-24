from torchvision import transforms
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

class MultiClassMammo(Dataset):
    
    def __init__(self, dataframe, transform1 = None, transform2=None):
        
        self.dataframe = dataframe
        self.study_id = dataframe.study_id.values
        self.L_CC = dataframe.L_CC_imid.values
        self.R_CC = dataframe.R_CC_imid.values
        self.L_MLO = dataframe.L_MLO_imid.values
        self.R_MLO = dataframe.L_MLO_imid.values

        self.L_bi = dataframe.L_birad_max.values
        self.R_bi = dataframe.R_birad_max.values
        self.L_den = dataframe.L_den_max.values
        self.R_den = dataframe.R_den_max.values
        #augmentation  
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        
        L_CC_img = Image.open(createpath(self.study_id[index], self.L_CC[index])).convert('RGB')
        R_CC_img = Image.open(createpath(self.study_id[index], self.R_CC[index])).convert('RGB')
        L_MLO_img = Image.open(createpath(self.study_id[index], self.L_MLO[index])).convert('RGB')
        R_MLO_img = Image.open(createpath(self.study_id[index], self.R_MLO[index])).convert('RGB')

        L_birad = self.L_bi[index]
        R_birad = self.R_bi[index]
        L_density = self.L_den[index]
        R_density = self.R_den[index]
        sample = {'L_CC_img': L_CC_img, 'R_CC_img': R_CC_img, 'L_MLO_img': L_MLO_img, 'R_MLO_img' : R_MLO_img, 'L_birad': L_birad, 'R_birad':R_birad, 'L_density':L_density, 'R_density':R_density  }
        if self.transform2:
            print("Transform 2: ***")
            L_CC_img = self.transform2(image = np.array(sample['L_CC_img'])) #convert from PIL to cv2
            R_CC_img = self.transform2(image = np.array(sample['R_CC_img'])) 
            L_MLO_img = self.transform2(image = np.array(sample['L_MLO_img'])) 
            R_MLO_img = self.transform2(image = np.array(sample['R_MLO_img'])) 

            #PIL image has range from 0-1, cv2 img has img from 0 to 255
            L_CC_image_array = np.asarray((L_CC_img["image"] * 255).astype(np.uint8))
            R_CC_image_array = np.asarray((R_CC_img["image"] * 255).astype(np.uint8))
            L_MLO_image_array = np.asarray((L_MLO_img["image"] * 255).astype(np.uint8))
            R_MLO_image_array = np.asarray((R_MLO_img["image"] * 255).astype(np.uint8))

            #convert from cv2 to PIL
            L_CC_img = self.transform1(Image.fromarray(L_CC_image_array)) 
            R_CC_img = self.transform1(Image.fromarray(R_CC_image_array)) 
            L_MLO_img = self.transform1(Image.fromarray(L_MLO_image_array)) 
            R_MLO_img = self.transform1(Image.fromarray(R_MLO_image_array)) 
            
        else:
            L_CC_img = self.transform1(sample['L_CC_img'])
            R_CC_img = self.transform1(sample['R_CC_img'])
            L_MLO_img = self.transform1(sample['L_MLO_img'])
            R_MLO_img = self.transform1(sample['R_MLO_img'])

        sample = {'L_CC_img': L_CC_img, 'R_CC_img': R_CC_img, 'L_MLO_img': L_MLO_img, 'R_MLO_img' : R_MLO_img, 'L_birad': L_birad, 'R_birad':R_birad, 'L_density':L_density, 'R_density':R_density  }
        return sample

def createpath(study_id, img_id):
    return "/home/single4/mammo/mammo/data/updatedata/crop-images/"+study_id+"/"+img_id + ".png"