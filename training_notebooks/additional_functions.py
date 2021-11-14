import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision.utils import save_image



def test_on_val_data( epoch, destination_dir, gen_model, validation_loader, device):
    '''
    This function saves one batch of translated examples to the destination_dir
    '''
    inputs,outputs = next(iter(validation_loader))
    inputs,outputs = inputs.to( device), outputs.to( device)

    gen_model.eval()
    with torch.no_grad():
        
        outputs_fake = gen_model(inputs)
        res = torch.cat([inputs * 0.5 + 0.5,outputs * 0.5 + 0.5,outputs_fake*0.5 + 0.5],dim=2)
        save_image(res, destination_dir + f"/input_label_gen_{epoch+688}.png")

    gen_model.train()

def save_checkpoint(model, optimizer, scheduler, checkpoint_filename):
    '''
    To save checkpoint with model_state, optimizer_state, scheduler_state
    '''
    print("------Saving Checkpoint-------")
    checkpoint = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict()
    }
    torch.save(checkpoint,checkpoint_filename)

def load_checkpoint(checkpoint_file, model, optimizer,scheduler,lr,device):
    '''
    This function is used to load state dicts from the checkpoint file. Load state_dicts for model, optimizer, scheduler.
    '''
    print("------Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file,map_location= device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr