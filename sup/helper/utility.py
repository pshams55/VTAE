#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:21:57 2023

"""
#%%
import os
import torch
from torch import nn

#%% 
def memconsumption():
    import gc
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())

#%%
def get_dir(file):
    
    return os.path.dirname(os.path.realpath(file))

#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#%%
def model_summary(model):
    print(40*"=" + " Model Summary " + 40*"=")
    print(model)
    print('Number of parameters:', count_parameters(model))
    print(95*"=")

#%%
class CenterCrop(nn.Module):
    def __init__(self, h, w):
        super(CenterCrop, self).__init__()
        self.h = h
        self.w = w
        
    def forward(self, x):
        h, w = x.shape[2:]
        x1 = int(round((h - self.h) / 2.))
        y1 = int(round((w - self.w) / 2.))
        out = x[:,:,x1:x1+self.h,y1:y1+self.w]
        return out

#%%
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#%%
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input):
        return input.view(input.size(0), -1)   

#%%
class BatchReshape(nn.Module):
    def __init__(self, shape):
        super(BatchReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)

#%%
def affine_decompose(A):
    sx = (A[:,0,0].pow(2) + A[:,1,0].pow(2)).sqrt()
    sy = (A[:,1,1] * A[:,0,0] - A[:,0,1] * A[:,1,0]) / sx
    m = (A[:,0,1] * A[:,0,0] + A[:,1,0] * A[:,1,1]) / (A[:,1,1] * A[:,0,0] - A[:,0,1] * A[:,1,0])
    theta = torch.atan2(A[:,1,0] / sx, A[:,0,0] / sx)
    tx = A[:, 0, 2]
    ty = A[:, 1, 2]
    return sx, sy, m, theta, tx, ty

#%%
def construct_affine(params):
    device = params.device
    n = params.shape[0]
    sx, sy, angle = params[:,0], params[:,1], params[:,2]
    m, tx, ty = params[:,3], params[:,4], params[:,5]
    zeros = torch.zeros(n, 2, 2, device=device)
    rot = torch.stack((torch.stack((angle.cos(), -angle.sin()), dim=1), 
                       torch.stack((angle.sin(), angle.cos()), dim=1)), dim=1)
    shear = zeros.clone()
    shear[:,0,0] = 1; shear[:,1,1] = 1; shear[:,0,1] = m
    scale = zeros.clone()
    scale[:,0,0] = sx; scale[:,1,1] = sy
    A = torch.matmul(torch.matmul(rot, shear), scale)
    b = torch.stack((tx, ty), dim=1)
    theta = torch.cat((A,b[:,:,None]), dim=2)
    return theta.reshape(n, 6)
    
#%%
if __name__ == '__main__':
    pass
