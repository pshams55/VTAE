#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 05:33:17 2023

"""
#%%
import torch
from torch import nn
from torch.nn import functional as F
from .expm import torch_expm
from .utility import construct_affine

#%%
def expm(theta): 
    n_theta = theta.shape[0] 
    zero_row = torch.zeros(n_theta, 1, 3, dtype=theta.dtype, device=theta.device) 
    theta = torch.cat([theta, zero_row], dim=1) 
    theta = torch_expm(theta) 
    theta = theta[:,:2,:] 
    return theta 

#%%
class ST_Affine(nn.Module):
    def __init__(self, input_shape):
        super(ST_Affine, self).__init__()
        self.input_shape = input_shape
        
    def forward(self, x, theta, inverse=False):
        if inverse:
            A = theta[:,:4]
            b = theta[:,4:]
            A = torch.inverse(A.view(-1, 2, 2)).reshape(-1, 4)
            b = -b
            theta = torch.cat((A,b), dim=1)
            
        theta = theta.view(-1, 2, 3)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x
    
    def trans_theta(self, theta):
        return theta
    
    def dim(self):
        return 6

#%%
class ST_AffineDecomp(nn.Module):
    def __init__(self, input_shape):
        super(ST_AffineDecomp, self).__init__()
        self.input_shape = input_shape
        
    def forward(self, x, theta, inverse=False):
        # theta = [sx, sy, angle, shear, tx, ty]
        if inverse:
            theta[:,:2] = 1/theta[:,:2]
            theta[:,2:] = -theta[:,2:]
            
        theta = construct_affine(theta)
        theta = theta.view(-1, 2, 3)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x
            
    def trans_theta(self, theta):
        return theta
    
    def dim(self):
        return 6

#%%
class ST_AffineDiff(nn.Module):
    def __init__(self, input_shape):
        super(ST_AffineDiff, self).__init__()
        self.input_shape = input_shape
        
    def forward(self, x, theta, inverse=False):
        if inverse:
            theta = -theta
        theta = theta.view(-1, 2, 3)
        theta = expm(theta)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x
    
    def trans_theta(self, theta):
        return expm(theta)
    
    def dim(self):
        return 6

#%%
try:
    from libcpab import cpab

    class ST_CPAB(nn.Module):
        def __init__(self, input_shape):
            super(ST_CPAB, self).__init__()
            self.input_shape = input_shape
            self.cpab = cpab([2,4], backend='pytorch', device='gpu',
                             zero_boundary=True, 
                             volume_perservation=False)
        
        def forward(self, x, theta, inverse=False):
            if inverse:
                theta = -theta
            out = self.cpab.transform_data(data = x, 
                                           theta = theta,    
                                           outsize = self.input_shape[1:])
            return out
        
        def trans_theta(self, theta):
            return theta
        
        def dim(self):
            return self.cpab.get_theta_dim()
except Exception as e:
    print('Could not import libcpab, error was')
    print(e)
    class ST_CPAB(nn.Module):
        def __init__(self, input_shape):
            super(ST_CPAB, self).__init__()
            self.input_shape = input_shape
            
        def forward(self, x, theta, inverse=False):
            raise ValueError('''libcpab was not correctly initialized, so you 
                             cannot run with --stn_type cpab''')
    
#%%
def get_transformer(name):
    transformers = {'affine': ST_Affine,
                    'affinediff': ST_AffineDiff,
                    'affinedecomp': ST_AffineDecomp,
                    'cpab': ST_CPAB
                    }
    assert (name in transformers), 'Transformer not found, choose between: ' \
            + ', '.join([k for k in transformers.keys()])
    return transformers[name]


#%%
if __name__ == '__main__':
    pass