

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory


data_transforms ={
    'train': transforms.compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],)
    ]),
}
data_dir ='/home/ibab/Documents/Deep_learning_chiranjeevi_sai_deva/lab12/data/hymenoptera_data/'
image_datasets ={
    x: datasets.ImageFolder(os.path.join(data_dir, x),
                data_transforms[x]
                )
                 for x in ['train','val']
}
dataloaders = {
    x:torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=4,
        shuffle=True,
        num_workrs=4
    )
    for x in ['train','val']
}
dataset_sizes ={
    x: len(image_datasets[x]
           )
    for x in ['train','val']
}
class_names = image_datasets['train'].classes

device = torch.accelerator.current_accelerator().type
         if torch,accelerator.is_available()
