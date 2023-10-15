# -*- coding: utf-8 -*-
"""
Created on Tue Thu Sep 14 18:21:47 2023

"""

#%%
import torch
from torch import nn
import numpy as np
from torchvision.utils import make_grid
from ..helper.utility import affine_decompose, Identity
from ..helper.spatial_transformer import get_transformer

#%%
class VITAE_UI(nn.Module):
    def __init__(self, input_shape, latent_dim, encoder, decoder, outputdensity, ST_type, **kwargs):
        super(VITAE_UI, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_spaces = 2
        self.outputdensity = outputdensity
        
        # Spatial transformer
        self.stn = get_transformer(ST_type)(input_shape)
        self.ST_type = ST_type
        
        # Define outputdensities
        if outputdensity == 'bernoulli':
            outputnonlin = nn.Sigmoid()
        elif outputdensity == 'gaussian':
            outputnonlin = Identity()
        else:
            ValueError('Unknown output density')
        
        # Define encoder and decoder
        self.encoder1 = encoder(input_shape, latent_dim)
        self.decoder1 = decoder((self.stn.dim(),), latent_dim, Identity())
        
        self.encoder2 = encoder(input_shape, latent_dim)
        self.decoder2 = decoder(input_shape, latent_dim, outputnonlin)

    #%%
    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        batch_size, latent_dim = mu.shape
        eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
        return (mu[:,None,None,:] + var[:,None,None,:].sqrt() * eps).reshape(-1, latent_dim)
    
    #%%
    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0):
        # Encode/decode transformer space
        mu1, var1 = self.encoder1(x)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1)
        
        # Encode/decode semantic space
        mu2, var2 = self.encoder2(x)
        z2 = self.reparameterize(mu2, var2, eq_samples, iw_samples)
        x_mean, x_var = self.decoder2(z2)
        
        # Transform output
        x_mean = self.stn(x_mean, theta_mean, inverse=False)
        x_var = self.stn(x_var, theta_mean, inverse=False)
        x_var = switch*x_var + (1-switch)*0.02**2
        
        return x_mean, x_var, [z1, z2], [mu1, mu2], [var1, var2]

    #%%
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.latent_dim, device=device)
            z2 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            x_mean, _ = self.decoder2(z2)
            out_mean = self.stn(x_mean, theta_mean)
            return out_mean
        
    #%%
    def special_sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.latent_dim, device=device)
            z2 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            x_mean, _ = self.decoder2(z2)
            out_mean = self.stn(x_mean, theta_mean)
            return out_mean, [z1, z2]

    #%%
    def sample_only_trans(self, n, img):
        device = next(self.parameters()).device
        with torch.no_grad():
            img = img.repeat(n, 1, 1, 1).to(device)
            z1 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            out = self.stn(img, theta_mean)
            return out

    #%%
    def sample_only_images(self, n, trans):
        device = next(self.parameters()).device
        with torch.no_grad():
            trans = trans[None, :].repeat(n, 1).to(device)
            z2 = torch.randn(n, self.latent_dim, device=device)
            x_mean, _ = self.decoder2(z2)
            out = self.stn(x_mean, trans)
            return out
    
    #%%
    def sample_transformation(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            theta = self.stn.trans_theta(theta_mean.reshape(-1, 2, 3))
            return theta.reshape(-1, 6)
    
    #%%
    def semantics(self, x, eq_samples=1, iw_samples=1, switch=1.0):
        mu1, var1 = self.encoder1(x)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1)
        mu2, var2 = self.encoder2(x)
        z2 = self.reparameterize(mu2, var2, eq_samples, iw_samples)
        x_mean, x_var = self.decoder2(z2)
        return x_mean, x_var, [z1, z2], [mu1, mu2], [var1, var2]
    
    #%%
    def latent_representation(self, x):
        z_mu1, _ = self.encoder1(x)
        z_mu2, _ = self.encoder2(x)
        return [z_mu1, z_mu2]

    #%%
    def callback(self, writer, loader, epoch):
        n = 10      
        trans = torch.tensor(np.zeros(self.stn.dim()), dtype=torch.float32)
        samples = self.sample_only_images(n*n, trans)
        writer.add_image('samples/fixed_trans', make_grid(samples.cpu(), nrow=n),
                         global_step=epoch)
        del samples
        
        img = (next(iter(loader))[0][0]).to(torch.float32)
        samples = self.sample_only_trans(n*n, img)
        writer.add_image('samples/fixed_img', make_grid(samples.cpu(), nrow=n),
                          global_step=epoch)
        del samples
        
        # Lets log a histogram of the transformation
        theta = self.sample_transformation(1000)
        for i in range(theta.shape[1]):
            writer.add_histogram('transformation/a' + str(i), theta[:,i], 
                                 global_step=epoch, bins='auto')
            writer.add_scalar('transformation/mean_a' + str(i), theta[:,i].mean(),
                              global_step=epoch)
        
        # Also to a decomposition of the matrix and log these values
        if self.stn.dim() == 6:
            values = affine_decompose(theta.view(-1, 2, 3))
            tags = ['sx', 'sy', 'm', 'theta', 'tx', 'ty']
            for i in range(6):
                writer.add_histogram('transformation/' + tags[i], values[i],
                                     global_step=epoch, bins='auto')
                writer.add_scalar('transformation/mean_' + tags[i], values[i].mean(),
                                  global_step=epoch)
            del values
        del theta
            
        # If 2d latent space we can make a fine meshgrid of sampled points
        if self.latent_dim == 2:
            device = next(self.parameters()).device
            x = np.linspace(-3, 3, 20)
            y = np.linspace(-3, 3, 20)
            z = np.stack([array.flatten() for array in np.meshgrid(x,y)], axis=1)
            z = torch.tensor(z, dtype=torch.float32)
            trans = torch.tensor(np.zeros(self.stn.dim()), dtype=torch.float32).repeat(20*20, 1)
            x_mean, x_var = self.decoder2(z.to(device))
            out = self.stn(x_mean, trans.to(device))
            writer.add_image('samples/meshgrid', make_grid(out.cpu(), nrow=20),
                             global_step=epoch)
            del out