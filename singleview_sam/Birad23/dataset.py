from torchvision import transforms
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np 

class MultiClassMammo(Dataset):
    
    def __init__(self, dataframe, transform1 = None, transform2=None):
        
        self.dataframe = dataframe
        self.imagespath = dataframe.path.values
        self.laterality = dataframe.laterality.values
        self.transform1 = transform1
        self.transform2 = transform2
        self.label_birad = dataframe.label_birad.values
        self.label_density = dataframe.label_density.values
        
        
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        
        image = Image.open(self.imagespath[index]).convert('RGB')
        if self.laterality[index] == "L":
            image = transforms.functional.hflip(image)
        lb_birad = self.label_birad[index]
        lb_density = self.label_density[index]
        sample = {'image': image, 'label_birad': lb_birad, 'label_density' : lb_density }
        if self.transform1:
            image = self.transform2(image = np.array(sample["image"])) #convert from PIL to cv2
            #PIL image has range from 0-1, cv2 img has img from 0 to 255
            image_array = np.asarray((image["image"] * 255).astype(np.uint8))
            image = self.transform1(Image.fromarray(image_array)) #convert from cv2 to PIL
            #self.transform(sample['image'])
            sample = {'image': image, 'bi': lb_birad, 'den' : lb_density }
        
        return sample