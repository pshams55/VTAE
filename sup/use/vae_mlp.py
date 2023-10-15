# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 22:09:34 2023

"""

#%%
import torch
from torch import nn
import numpy as np
from torchvision.utils import make_grid

from ..helper.losses import ELBO

#%%
class VAE_Mlp(nn.Module):
    def __init__(self, input_shape, latent_dim, **kwargs):
        super(VAE_Mlp, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.flat_dim = np.prod(input_shape)
        self.latent_dim = [latent_dim]
        
        # Define encoder and decoder
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.z_mean = nn.Linear(256, latent_dim)
        self.z_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.flat_dim),
            nn.Sigmoid()
        )
    
    #%%
    def encode(self, x):
        x = self.encoder(x.view(x.shape[0], -1))
        mu = self.z_mean(x)
        logvar = self.z_var(x)
        return mu, logvar
    
    #%%
    def decode(self, z):
        out = self.decoder(z)
        return out.view(-1, *self.input_shape)
    
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
    model = VAE_Mlp((1, 28, 28), 32)


