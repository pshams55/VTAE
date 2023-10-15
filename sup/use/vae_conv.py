#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 11:52:36 2023

"""

#%%
import torch
from torch import nn
import numpy as np
from torchvision.utils import make_grid

from ..helper.utility import CenterCrop
from ..helper.losses import ELBO

#%%
class VAE_Conv(nn.Module):
    def __init__(self, input_shape, latent_dim, **kwargs):
        super(VAE_Conv, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.latent_dim = [latent_dim]
        
        # Define encoder and decoder
        c,h,w = input_shape
        self.z_dim = int(np.ceil(h/2**2)) # receptive field downsampled 2 times
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.z_mean = nn.Linear(64 * self.z_dim**2, latent_dim)
        self.z_var = nn.Linear(64 * self.z_dim**2, latent_dim)
        self.z_develop = nn.Linear(latent_dim, 64 * self.z_dim**2)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
            CenterCrop(h,w),
            nn.Sigmoid()
        )        
    
    #%%
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.z_mean(x)
        logvar = self.z_var(x)
        return mu, logvar
    
    #%%
    def decode(self, z):
        out = self.z_develop(z)
        out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
        out = self.decoder(out)
        return out
    
    #%%
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu
    
    #%%
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, [mu], [logvar]
    
    #%%
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim[0], device=device)
            out = self.decode(z)
            return out
    
    #%%
    def latent_representation(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return [z]
    
    #%%
    def loss_f(self, data, recon_data, mus, logvars, epoch, warmup):
        return ELBO(data, recon_data, mus, logvars, epoch, warmup)
    
    #%%
    def __len__(self):
        return 1
    
    #%%
    def callback(self, writer, loader, epoch):
        # If 2d latent space we can make a fine meshgrid of sampled points
        if self.latent_dim[0] == 2:
            device = next(self.parameters()).device
            x = np.linspace(-3, 3, 20)
            y = np.linspace(-3, 3, 20)
            z = np.stack([array.flatten() for array in np.meshgrid(x,y)], axis=1)
            z = torch.tensor(z, dtype=torch.float32)
            out = self.decode(z.to(device))
            writer.add_image('samples/meshgrid', make_grid(out.cpu(), nrow=20),
                             global_step=epoch)
    
#%%
if __name__ == '__main__':
    model = VAE_Conv((1, 28, 28), 32)
