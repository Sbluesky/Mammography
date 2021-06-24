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
        #max left, max right
        self.L_bi = dataframe.L_birad_max.values
        self.R_bi = dataframe.R_birad_max.values
        self.L_den = dataframe.L_den_max.values
        self.R_den = dataframe.R_den_max.values

        #L_CC_birad,L_CC_den,L_MLO_bi,L_MLO_den,R_CC_birad,R_CC_den,R_MLO_bi,R_MLO_den
        #true label
        self.LCC_bi = dataframe.L_CC_birad.values
        self.RCC_bi = dataframe.R_CC_birad.values
        self.LMLO_bi = dataframe.L_MLO_bi.values
        self.RMLO_bi = dataframe.R_MLO_bi.values


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

        #max label for left, right
        L_birad = self.L_bi[index]
        R_birad = self.R_bi[index]
        L_density = self.L_den[index]
        R_density = self.R_den[index]

        #true label
        LCC_bi = self.LCC_bi[index]
        RCC_bi = self.RCC_bi[index]
        LMLO_bi = self.LMLO_bi[index]
        RMLO_bi = self.RMLO_bi[index]

        if self.transform1:
            L_CC_img = self.transform1(L_CC_img)
            R_CC_img = self.transform1(R_CC_img)
            L_MLO_img = self.transform1(L_MLO_img)
            R_MLO_img = self.transform1(R_MLO_img)
        
        sample = {'L_CC_img': L_CC_img, 'R_CC_img': R_CC_img, 'L_MLO_img': L_MLO_img, 'R_MLO_img' : R_MLO_img, 
            'L_birad': L_birad, 'R_birad':R_birad, 'L_density':L_density, 'R_density':R_density,
            'LCC_bi': LCC_bi, 'RCC_bi': RCC_bi, 'LMLO_bi' : LMLO_bi, 'RMLO_bi': RMLO_bi }
        
        return sample

def createpath(study_id, img_id):
    return "/home/single4/mammo/mammo/data/updatedata/crop-images/"+study_id+"/"+img_id + ".png"