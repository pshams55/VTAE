#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:22:15 2023

"""

#%%
import torch
from torch import nn
import numpy as np
from torchvision.utils import make_grid

#%%
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim, encoder, decoder, outputdensity, **kwargs):
        super(VAE, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.flat_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.latent_spaces = 1
        self.outputdensity = outputdensity
        
        # Define outputdensities
        if outputdensity == 'bernoulli':
            outputnonlin = nn.Sigmoid()
        elif outputdensity == 'gaussian':
            outputnonlin = nn.ReLU()
        else:
            ValueError('Unknown output density')
        
        # Define encoder and decoder
        self.encoder = encoder(input_shape, latent_dim)
        self.decoder = decoder(input_shape, latent_dim, outputnonlin)
    
    #%%
    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        batch_size, latent_dim = mu.shape
        eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
        return (mu[:,None,None,:] + var[:,None,None,:].sqrt() * eps).reshape(-1, latent_dim)
    
    #%%
    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0):
        z_mu, z_var = self.encoder(x)
        z = self.reparameterize(z_mu, z_var, eq_samples, iw_samples)
        x_mu, x_var = self.decoder(z)
        x_var = switch*x_var + (1-switch)*(1**2)
        return x_mu, x_var, [z], [z_mu], [z_var]
    
    #%%
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            x_mu, x_var = self.decoder(z)
            return x_mu
        
    #%%
    def special_sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            x_mu, x_var = self.decoder(z)
            return x_mu, [z]
    
    #%%
    def semantics(self, x, eq_samples=1, iw_samples=1, switch=1.0):
        z_mu, z_var = self.encoder(x)
        z = self.reparameterize(z_mu, z_var, eq_samples, iw_samples)
        x_mu, x_var = self.decoder(z)
        x_var = switch*x_var + (1-switch)*(1**2)
        return x_mu, x_var, [z], [z_mu], [z_var]
    
    #%%
    def latent_representation(self, x):
        z_mu, z_var = self.encoder(x)
        return [z_mu]
    
    #%%
    def callback(self, writer, loader, epoch):
        # If 2d latent space we can make a fine meshgrid of sampled points
        if self.latent_dim == 2:
            device = next(self.parameters()).device
            x = np.linspace(-3, 3, 20)
            y = np.linspace(-3, 3, 20)
            z = np.stack([array.flatten() for array in np.meshgrid(x,y)], axis=1)
            z = torch.tensor(z, dtype=torch.float32)
            out_mu, out_var = self.decoder(z.to(device))
            writer.add_image('samples/meshgrid', make_grid(out_mu.cpu(), nrow=20),
                             global_step=epoch)
    
#%%
if __name__ == '__main__':
    pass