# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:49:44 2019

@author: nsde
"""
#%%
from sup.helper.utility import count_parameters
from sup.models import get_model
from torch import nn
import numpy as np

#%%
class mlp_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(mlp_encoder, self).__init__()
        self.flat_dim = np.prod(input_shape)
        self.encoder_mu = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_dim)
        )
        self.encoder_var = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, latent_dim),
            nn.Softplus(),
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z_mu = self.encoder_mu(x)
        z_var = self.encoder_var(x)
        return z_mu, z_var
    
#%%
class mlp_decoder(nn.Module):
    def __init__(self, output_shape, latent_dim, outputnonlin):
        super(mlp_decoder, self).__init__()
        self.flat_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.flat_dim),
            outputnonlin
        )
        self.decoder_var = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, self.flat_dim),
            nn.Softplus()
        )
        
    def forward(self, z):
        x_mu = self.decoder_mu(z).reshape(-1, *self.output_shape)
        x_var = 0.1#self.decoder_var(z).reshape(-1, *self.output_shape)
        return x_mu, x_var
    
#%%
model_class = get_model('vitae_ci')
model = model_class(input_shape = (1,28,28),
                    latent_dim = 2, 
                    encoder = mlp_encoder, 
                    decoder = mlp_decoder, 
                    outputdensity = 'bernoulli',
                    ST_type = 'affine')
print(count_parameters(model))