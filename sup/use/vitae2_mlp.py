#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 18:36:41 2023

"""

#%%
import torch
from torch import nn
from torchvision.utils import make_grid
import numpy as np

from ..helper.utility import affine_decompose
from ..helper.spatial_transformer import STN_AffineDiff, expm
from ..helper.losses import ELBO

#%%
class VITAE2_Mlp(nn.Module):
    def __init__(self, input_shape, latent_dim, mcmc_samples=1, **kwargs):
        super(VITAE2_Mlp, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.flat_dim = np.prod(input_shape)
        self.latent_dim = [latent_dim, latent_dim]
        self.mcmc_samples = mcmc_samples
        
        # Define encoder and decoder
        self.encoder1 = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.z_mean1 = nn.Linear(256, latent_dim)
        self.z_var1 = nn.Linear(256, latent_dim)
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.flat_dim),
            nn.Sigmoid()
        )
        self.encoder2 = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.z_mean2 = nn.Linear(256, latent_dim)
        self.z_var2 = nn.Linear(256, latent_dim)
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 6)
        )
        # Initialize close to the identity transformation
        for i, layer in enumerate(self.decoder2):
            if i == 2:
                nn.init.normal_(layer.weight, std=1e-2)
                nn.init.zeros_(layer.bias)
        
        self.stn = STN_AffineDiff(input_shape=input_shape)
    
    #%%
    def encode1(self, x):
        x = self.encoder1(x.view(x.shape[0], -1))
        mu = self.z_mean1(x)
        logvar = self.z_var1(x)
        return mu, logvar
    
    #%%
    def encode2(self, x):
        x = self.encoder2(x.view(x.shape[0], -1))
        mu = self.z_mean2(x)
        logvar = self.z_var2(x)
        return mu, logvar
    
    #%%
    def decode1(self, z):
        out = self.decoder1(z)
        return out.view(-1, *self.input_shape)
    
    #%%
    def decode2(self, z):
        out = self.decoder2(z)
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
    def reparameterize2(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn(self.mcmc_samples, *std.shape)
            return eps.mul(std).add(mu).reshape(-1, std.shape[1])
        else:
            return mu
    
    #%%
    def forward(self, x): 
        mu1, logvar1 = self.encode1(x) 
        mu2, logvar2 = self.encode2(x) 
        z1 = self.reparameterize(mu1, logvar1) 
        z2 = self.reparameterize(mu2, logvar2) 
        dec = self.decode1(z1) 
        theta = self.decode2(z2) 
        out = self.stn(dec, theta) 
        return out, [mu1, mu2], [logvar1, logvar2] 

    #%%
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.latent_dim[0], device=device)
            z2 = torch.randn(n, self.latent_dim[1], device=device)
            dec = self.decode1(z1)
            theta = self.decode2(z2)
            out = self.stn(dec, theta)
            return out
    
    #%%
    def sample_only_images(self, n, trans):
        device = next(self.parameters()).device
        with torch.no_grad():
            trans = trans[None, :].repeat(n, 1).to(device)
            z1 = torch.randn(n, self.latent_dim[0], device=device)
            dec = self.decode1(z1)
            out = self.stn(dec, trans)
            return out
     
    #%%
    def sample_only_trans(self, n, img):
        device = next(self.parameters()).device
        with torch.no_grad():
            img = img.repeat(n, 1, 1, 1).to(device)
            z2 = torch.randn(n, self.latent_dim[1], device=device)
            theta = self.decode2(z2)
            out = self.stn(img, theta)
            return out
    
    #%%
    def latent_representation(self, x):
        mu1, logvar1 = self.encode1(x)
        mu2, logvar2 = self.encode2(x)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        return [z1, z2]
    
    #%%
    def sample_transformation(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z2 = torch.randn(n, self.latent_dim[1], device=device)
            theta = self.decode2(z2)
            theta = expm(theta.reshape(-1, 2, 3))
            return theta.reshape(-1, 6)
    
    #%%
    def loss_f(self, data, recon_data, mus, logvars, epoch, warmup):
        return ELBO(data, recon_data, mus, logvars, epoch, warmup)
    
    #%%
    def __len__(self):
        return 2

    #%%
    def callback(self, writer, loader, epoch):
        n = 10      
        trans = torch.tensor([0,0,0,0,0,0], dtype=torch.float32)
        samples = self.sample_only_images(n*n, trans)
        writer.add_image('samples/fixed_trans', make_grid(samples.cpu(), nrow=n),
                         global_step=epoch)
        
        img = next(iter(loader))[0][0]
        samples = self.sample_only_trans(n*n, img)
        writer.add_image('samples/fixed_img', make_grid(samples.cpu(), nrow=n),
                          global_step=epoch)
    
        # Lets log a histogram of the transformation
        theta = self.sample_transformation(1000)
        for i in range(6):
            writer.add_histogram('transformation/a' + str(i), theta[:,i], 
                                 global_step=epoch, bins='auto')
            writer.add_scalar('transformation/mean_a' + str(i), theta[:,i].mean(),
                              global_step=epoch)
        # Also to a decomposition of the matrix and log these values
        values = affine_decompose(theta.view(-1, 2, 3))
        tags = ['sx', 'sy', 'm', 'theta', 'tx', 'ty']
        for i in range(6):
            writer.add_histogram('transformation/' + tags[i], values[i],
                                 global_step=epoch, bins='auto')
            writer.add_scalar('transformation/mean_' + tags[i], values[i].mean(),
                              global_step=epoch)
            
        # If 2d latent space we can make a fine meshgrid of sampled points
        if self.latent_dim[0] == 2:
            device = next(self.parameters()).device
            x = np.linspace(-3, 3, 20)
            y = np.linspace(-3, 3, 20)
            z = np.stack([array.flatten() for array in np.meshgrid(x,y)], axis=1)
            z = torch.tensor(z, dtype=torch.float32)
            trans = torch.tensor([0,0,0,0,0,0], dtype=torch.float32).repeat(20*20, 1, 1)
            dec = self.decode1(z.to(device))
            out = self.stn(dec, trans.to(device))
            writer.add_image('samples/meshgrid', make_grid(out.cpu(), nrow=20),
                             global_step=epoch)
    
#%% 
if __name__ == '__main__':
    model = VITAE2_Mlp((1, 28, 28), 32)
