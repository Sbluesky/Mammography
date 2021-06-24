from torchvision import transforms
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

def createpath(path):
    return path.replace("single4/mammo", "single1/BACKUP/SamHUyen")

class MultiClassMammo(Dataset):
    
    def __init__(self, dataframe, transform1 = None, transform2=None):
        
        self.dataframe = dataframe
        self.studyid = dataframe.study_id.values
        self.imageid = dataframe.image_id.values
        self.path = dataframe.path.values
        self.laterality = dataframe.laterality.values
        self.transform1 = transform1
        self.transform2 = transform2
        self.label_birad = dataframe.label_birad.values
        self.label_density = dataframe.label_density.values
        
        
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        
        image = Image.open(createpath(self.path[index])).convert('RGB')
        if self.laterality[index] == "R":
            image = transforms.functional.hflip(image)
        lb_birad = self.label_birad[index]
        lb_density = self.label_density[index]
        if self.transform2:
            print("Transform 2: ")
            image = self.transform2(image = np.array(image)) #convert from PIL to cv2
            #PIL image has range from 0-1, cv2 img has img from 0 to 255
            image_array = np.asarray((image['image'] * 255).astype(np.uint8))
            image = self.transform1(Image.fromarray(image_array)) #convert from cv2 to PIL
            #self.transform(sample['image'])
        else:
            image = self.transform1(image)
        sample = {'image': image, 'bi': lb_birad, 'den' : lb_density }
        return sample

