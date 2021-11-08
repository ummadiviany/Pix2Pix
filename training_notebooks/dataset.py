
class MapDataset(Dataset):
    def __init__(self,root_dir,input_size,direction):
        self.root_dir = root_dir
        self.input_size = input_size
        self.direction = direction
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self,index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        
        input_image = image[:,:self.input_size,:]
        target_image = image[:,self.input_size:,:]
        
        both_transform= A.Compose(
            [   
                A.Resize(width=256,height=256),
                A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],max_pixel_value=255.0),
                ToTensorV2()
            ]
        )
        input_image =  both_transform(image = input_image)['image']
        target_image = both_transform(image = target_image)['image']

        if self.direction:
            return target_image, input_image
        return input_image, target_image