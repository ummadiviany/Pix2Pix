# importing all necessary libraries
import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2):
        '''
        Getting inputs from instances
        '''
        super().__init__()
        # A sequentatial block with Conv,BatchNorm,LeakyReLU
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,bias=False,padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self,x):
        '''
        rewriting forward function
        '''
        return self.conv(x) #return output

    
class Discriminator(nn.Module):
    def __init__(self,in_channels=3,features=[64,128,256,512]):
        ''''
        Getting inputs from instances
        '''
        super().__init__()
        #   Initial sequential block with only Conv and LeakyReLU
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2,features[0],kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        # Creating list of layers
        layers = [] 
        in_channels = features[0]
        # Iterating through all features and creating sequential CNN Blocks and appending to the layers list
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels,feature,stride= 1 if feature == features[-1] else 2)
            )
            in_channels = feature

        # one last Conv layer 
        layers.append(
            nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode='reflect')
        )

        # Unpacking all layers and creating a sequential form
        self.model = nn.Sequential(*layers)

    def forward(self,x,y):
        '''
        Rewrite of forward function
        '''
        x = torch.cat([x,y],dim=1) # Concatinating input and tartget image
        x = self.initial(x)         # First through initial layer
        return self.model(x)        # then through entire sequential model