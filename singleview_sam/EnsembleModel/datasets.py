from torchvision import transforms
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultiClassMammo(Dataset):
    
    def __init__(self, dataframe, transform = None):
        
        self.dataframe = dataframe
        self.imagespath = dataframe.path.values
        self.laterality = dataframe.laterality.values
        self.transform = transform
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
        if self.transform:
            image = self.transform(sample['image'])
            #sample = {'image': image, 'bi': lb_birad, 'den' : lb_density }
            sample = {'image': image, 'bi': lb_birad}
        
        return sample