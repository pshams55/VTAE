#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:18:28 2018

@author: nsde
"""

#%%
import torch
import argparse, datetime
from torchvision import transforms

from sup.trainer import vae_trainer
from sup.data.mnist_data_loader import mnist_data_loader
from sup.data.perception_data_loader import perception_data_loader
from sup.helper.utility import model_summary
from sup.helper.encoder_decoder import get_encoder, get_decoder
from sup.models import get_model
from sup.helper.losses import vae_loss

import time

#%%
def argparser():
    """ Argument parser for the main script """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model settings
    ms = parser.add_argument_group('Model settings')
    ms.add_argument('--model', type=str, default='vae', help='model to train')
    ms.add_argument('--ed_type', type=str, default='mlp', help='encoder/decoder type')
    ms.add_argument('--stn_type', type=str, default='affinediff', help='transformation type to use')
    ms.add_argument('--beta', type=float, default=8.0, help='beta value for beta-vae model')
    
    # Training settings
    ts = parser.add_argument_group('Training settings')
    ts.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    ts.add_argument('--eval_epoch', type=int, default=1000, help='when to evaluate log(p(x))')
    ts.add_argument('--batch_size', type=int, default=1024, help='size of the batches')
    ts.add_argument('--warmup', type=int, default=100, help='number of warmup epochs for kl-terms')
    ts.add_argument('--lr', type=float, default=1e-3, help='learning rate for adam optimizer')
    
    # Hyper settings
    hp = parser.add_argument_group('Variational settings')
    hp.add_argument('--latent_dim', type=int, default=2, help='dimensionality of the latent space')
    hp.add_argument('--density', type=str, default='bernoulli', help='output density')    
    hp.add_argument('--eq_samples', type=int, default=1, help='number of MC samples over the expectation over E_q(z|x)')
    hp.add_argument('--iw_samples', type=int, default=1, help='number of importance weighted samples')
    
    # Dataset settings
    ds = parser.add_argument_group('Dataset settings')
    ds.add_argument('--classes','--list', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='classes to train on')
    ds.add_argument('--num_points', type=int, default=10000, help='number of points in each class')
    ds.add_argument('--logdir', type=str, default='beta_final8', help='where to store results')
    ds.add_argument('--dataset', type=str, default='mnist', help='dataset to use')
    
    # Parse and return
    args = parser.parse_args()
    return args

#%%
args = argparser()

transformations = transforms.Compose([ 
    transforms.ToTensor(), 
])

trainloader, testloader = mnist_data_loader(root='sup/data', 
                                            transform=transformations,
                                            download=True,
                                            classes=args.classes,
                                            num_points=args.num_points,
                                            batch_size=args.batch_size)
img_size = (1, 28, 28)

# Construct model
model_class = get_model(args.model)
model = model_class(input_shape = img_size,
                    latent_dim = args.latent_dim, 
                    encoder = get_encoder(args.ed_type), 
                    decoder = get_decoder(args.ed_type), 
                    outputdensity = args.density,
                    ST_type = args.stn_type)
    
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


data_train = next(iter(trainloader))[0].to(torch.float32).cuda()
data_train = data_train.reshape(-1, *img_size)

# Warmup 
for i in range(100):
    out = model(data_train, 1, 1)

#%% Forward pass    
start = time.time()
for i in range(1000):
    out = model(data_train, 1, 1)
end = time.time()
print('forward time', (end - start) / 1000)

#%% Backward pass
start = time.time()
for i in range(1000):
    optimizer.zero_grad()
    out = model(data_train, 1, 1)
    loss, _, _ = vae_loss(data, *out, 
                          1, 1, 
                          args.latent_dim, 
                          1, 1, 1.0,
                          'bernoulli')
    (-loss).backward()
    optimizer.step()
end = time.time()
print('backward time', (end - start) / 1000)
    


