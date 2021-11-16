# Importing all necessary libraries
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,down=True,act='relu',use_dropout=False):
        '''
        This is an init class of block module, getting all inputs from instances
        '''
        super().__init__()
        # creating a sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,2,1,bias=False,padding_mode='reflect')
            if down 
            else nn.ConvTranspose2d(in_channels,out_channels,4,2,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2),

        )
        # if dropout , creaating a dropout  with 0.5 probability
        self.use_dropout = use_dropout
        self.drop = nn.Dropout(0.5)

    def forward(self,x):
        '''
        Rewrite of the forward function
        '''
        x=self.conv(x)  # Passing through sequential set created
        return self.drop(x) if self.use_dropout else x  # using dropout if dropout=True


class Generator(nn.Module):
    def __init__(self,in_channels=3,features=64):
        '''
        Init module of Generator class.
        Getting all the inputs from instances
        '''
        super().__init__()

        # Creating a sequential form of Conv, LeakyReLU
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels,features,4,2,1,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        # This is encoder part of the unet architecture
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU
        self.down1 = Block(features,features*2,down=True,act='leaky',use_dropout=False)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU
        self.down2 = Block(features*2,features*4,down=True,act='leaky',use_dropout=False)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU
        self.down3 = Block(features*4,features*8,down=True,act='leaky',use_dropout=False)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU
        self.down4 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU
        self.down5 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU
        self.down6 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)

        # Adding last Conv and ReLU
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1),nn.ReLU()
        )
        # This is the decoder part of the u-net architecture
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU, Dropout
        self.up1 = Block(features*8,features*8, down=False,act='relu', use_dropout=True)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU, Dropout
        self.up2 = Block(features*8*2,features*8, down=False,act='relu', use_dropout=True)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU, Dropout
        self.up3 = Block(features*8*2,features*8, down=False,act='relu', use_dropout=True)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU, Dropout
        self.up4 = Block(features*8*2,features*8, down=False,act='relu', use_dropout=True)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU, Dropout
        self.up5 = Block(features*8*2,features*4, down=False,act='relu', use_dropout=True)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU, Dropout
        self.up6 = Block(features*4*2,features*2, down=False,act='relu', use_dropout=True)
        # One block of sequential of Conv/Transposed Conv, BatchNorm, ReLU/LeakyReLU, Dropout
        self.up7 = Block(features*2*2,features, down=False,act='relu', use_dropout=True)

        # One last conv layer to map input_image channels with Tanh
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )


    def forward(self,x):
        '''
        This is a rewrite of the  forward function .
        Encoder-Decoder of U-Net
        '''
        # Passing through Downsampling blocks
        d1 = self.initial_down(x)
        # Passing through Downsampling blocks
        d2 = self.down1(d1)
        # Passing through Downsampling blocks
        d3 = self.down2(d2)
        # Passing through Downsampling blocks
        d4 = self.down3(d3)
        # Passing through Downsampling blocks
        d5 = self.down4(d4)
        # Passing through Downsampling blocks
        d6 = self.down5(d5)
        # Passing through Downsampling blocks
        d7 = self.down6(d6)
        # Passing through bottlenect block
        bottleneck = self.bottleneck(d7)

        # Passing through upsampling block
        up1 = self.up1(bottleneck)
        # Passing through upsampling block
        up2 = self.up2(torch.cat([up1,d7],1))
        # Passing through upsampling block + concatinating feature maps from downsampling block as well
        up3 = self.up3(torch.cat([up2,d6],1))
        # Passing through upsampling block + concatinating feature maps from downsampling block as well
        up4 = self.up4(torch.cat([up3,d5],1))
        # Passing through upsampling block + concatinating feature maps from downsampling block as well
        up5 = self.up5(torch.cat([up4,d4],1))
        # Passing through upsampling block + concatinating feature maps from downsampling block as well
        up6 = self.up6(torch.cat([up5,d3],1))
        # Passing through upsampling block + concatinating feature maps from downsampling block as well
        up7 = self.up7(torch.cat([up6,d2],1))
        # Passing through upsampling block + concatinating feature maps from downsampling block as well
        return self.final_up(torch.cat([up7,d1],1))



