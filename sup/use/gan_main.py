#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 06:24:18 2023

"""

#%%
from unsuper.data.data_loader import mnist_train_loader, mnist_test_loader
from unsuper import GAN

from torch import nn
from torchvision import transforms
import argparse
import numpy as np

#%%
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
    parser.add_argument('--learning-rate', action="store", default=1e-4, type=float, help='learning rate for optimizer')
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    # Input arguments
    args = argparser()
    
    # Get data loaders
    train_loader = mnist_train_loader(batch_size=128, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    test_loader = mnist_test_loader(batch_size=128, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    input_shape = (1, 28, 28)
    
    # Define generator and discriminator
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            self.model = nn.Sequential(
                            *self.block(args.latent_dim, 128, normalize=False),
                            *self.block(128, 256),
                            *self.block(256, 512),
                            *self.block(512, 1024),
                            nn.Linear(1024, int(np.prod(input_shape))),
                            nn.Tanh()
            )

        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *input_shape)
            return img
    
        def block(self, in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            self.model = nn.Sequential(
                    nn.Linear(int(np.prod(input_shape)), 512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
            )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
            return validity

    # Define model
    generator = Generator()
    discriminator = Discriminator()
    model = GAN(input_shape, args.latent_dim, generator, discriminator, device='cuda')
    
    # Train model
    model.train(trainloader=train_loader, n_epochs=args.n_epochs, 
                learning_rate=args.learning_rate, betas=(args.b1, args.b2))