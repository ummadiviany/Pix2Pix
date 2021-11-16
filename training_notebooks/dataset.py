# Importing all necessary libraries
from torch.utils.data import DataLoader, Dataset
import albumentations as A #
from albumentations.pytorch import ToTensorV2 #
import os #
import numpy as np #
from PIL import Image # 

class MapDataset(Dataset):
    def __init__(self,root_dir,input_size,direction):
        '''
        Init class for drawing inputs from the instance
        '''
        self.root_dir = root_dir
        self.input_size = input_size
        self.direction = direction
        self.list_files = os.listdir(self.root_dir) # getting list of all images

    def __len__(self):
        '''
        Length return method 
        '''
        return len(self.list_files) #returning len of dataset

    def __getitem__(self,index):
        '''
        GetItem method for returning one input image and one target image with indexing.
        '''
        img_file = self.list_files[index]   #getting a img file
        img_path = os.path.join(self.root_dir, img_file)    #joining path directory
        image = np.array(Image.open(img_path))  # reading image with Image
        
        input_image = image[:,:self.input_size,:]   #splitting it into input and target images
        target_image = image[:,self.input_size:,:]
        
        # A composition of transformations
        # 1)resize, Normalize, Tensor
        both_transform= A.Compose(
            [   
                A.Resize(width=256,height=256),
                A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],max_pixel_value=255.0),
                ToTensorV2()
            ]
        )
        input_image =  both_transform(image = input_image)['image']     # accessing input image
        target_image = both_transform(image = target_image)['image']    # accessing target image

        if self.direction:
            return target_image, input_image    #returning input and target images
        return input_image, target_image