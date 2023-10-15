#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:38:02 2023

"""

#%%
import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image

#%%
def perception_data_loader(root, transform=None, target_transform=None, 
                           download=False, batch_size=128, 
                           classes=[0,1,2,3,4,5,6,7,8,9], num_points=10000):
    # Load dataset
    train = PERCEPTION(root=root, train=True, transform=transform, download=download,
                       target_transform=target_transform, classes=classes, num_points=num_points)
    
    test = PERCEPTION(root=root, train=False, transform=transform, download=download, 
                      target_transform=target_transform, classes=classes, num_points=num_points)
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    return trainloader, testloader

#%%
class PERCEPTION(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 download=False, classes=[0,1,2,3,4,5,6,7,8,9], num_points = 20000):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        # Load file
        if self.train:
            file = np.load('unsuper/data/PERCEPTION/training.npz')
        else:
            file = np.load('unsuper/data/PERCEPTION/testing.npz')
        self.data = torch.tensor(file['data'])
        self.targets = torch.tensor(file['labels'])
        
        # Cut of data
        self.data = self.data[:num_points]
        self.targets = self.targets[:num_points]
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    
#%%
if __name__ == '__main__':
    dataset = PERCEPTION(' ')
    trainloader, testloader = perception_data_loader(root = ' ')
