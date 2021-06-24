import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MultiLabelMammo(Dataset):
    
    def __init__(self, dataframe, transform = None):
        
        self.dataframe = dataframe
        self.imagespath = dataframe.path.values
        self.transform = transform
        self.labels = dataframe.multilabel.values.tolist()
        
        
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        
        image = Image.open(self.imagespath[index]).convert('RGB')
        label = self.labels[index]
        label = label.strip('][').split(', ')
        label = np.array(label).astype('float')
        bi = label[:6]
        den = label[6:]
        sample = {'image': image, 'bi': bi, 'den': den}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image, 'bi': bi, 'den': den}
        
        return sample