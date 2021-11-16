# importing all necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt



def test_on_val_data( epoch, destination_dir, gen_model, validation_loader, device):
    '''
    This function saves one batch of translated examples to the destination_dir
    '''
    inputs,outputs = next(iter(validation_loader)) # fetching single batch of examples
    inputs,outputs = inputs.to( device), outputs.to( device)    # sending input images and target images to device(GPU)

    gen_model.eval()    # setting model in evaluation mode
    with torch.no_grad():   #using torch with no gradients
        
        outputs_fake = gen_model(inputs)    # genrating translated images from inputs
        res = torch.cat([inputs * 0.5 + 0.5,outputs * 0.5 + 0.5,outputs_fake*0.5 + 0.5],dim=2) # creating input images, output images, translated images into a single tensor
        save_image(res, destination_dir + f"input_label_gen_{epoch+1}.png") # saving image using save_image function

    gen_model.train()   # setting model in training mode
