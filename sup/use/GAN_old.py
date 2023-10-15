# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:12:48 2023

"""

#%%
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from torchvision.utils import save_image
import datetime
import os

#%%
class GAN(object):
    def __init__(self, input_shape, latent_dim, generator, discriminator, device='cpu', logdir=None):
        # Initialize
        self.input_shape = input_shape
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.logdir = './logs/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') \
                        if logdir is None else logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        
        # Initialize generator and discriminator
        self.generator = generator
        self.discriminator = discriminator
        assert isinstance(self.generator, nn.Module), 'Generator is not a nn.Module'
        assert isinstance(self.discriminator, nn.Module), 'Discriminator is not a nn.Module'
        
        # Loss function
        self.loss = torch.nn.BCELoss()
        
        # Transfer to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.loss.to(self.device)
    
    #%%
    def train(self, trainloader, n_epochs, learning_rate=1e-3, betas=(0.9, 0.999)):
        # Make optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=betas)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=betas)
        
        for epoch in range(n_epochs):
            progress_bar = tqdm(desc='Epoch ' + str(epoch), total=len(trainloader.dataset), 
                                unit='samples')
            total_g_loss, total_d_loss = 0, 0
            for i, (imgs, _) in enumerate(trainloader):
                # Adversarial ground truths
                valid = Variable(torch.Tensor(imgs.size(0), 1).fill_(1.0).to(self.device), requires_grad=False)
                fake = Variable(torch.Tensor(imgs.size(0), 1).fill_(0.0).to(self.device), requires_grad=False)
        
                # Configure input
                real_imgs = Variable(imgs.to(self.device))
                
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
            
                # Sample noise as generator input
                z = Variable(torch.randn(imgs.shape[0], self.latent_dim).to(self.device))
                
                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.loss(self.discriminator(real_imgs), valid)
                fake_loss = self.loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()
                
                # Update progress
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                progress_bar.update(imgs.size(0))
                progress_bar.set_postfix({'g_loss': g_loss.item(), 'd_loss': d_loss.item()})
            
            # Final progress
            progress_bar.set_postfix({'avg_g_loss': total_g_loss / len(trainloader),
                                      'avg_d_loss': total_d_loss / len(trainloader)})
            progress_bar.close()
            
            # Save some results
            self.snapshot(postfix=epoch)

    #%%
    def sample(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            samples = self.generator(z)
        return samples

    #%%            
    def snapshot(self, n=64, postfix=''):
        gen_data = self.sample(n)
        save_image(gen_data.data.cpu(), self.logdir + '/samples' + str(postfix) + '.png',
                   normalize=True)

#%%
if __name__ == '__main__':
    pass
                    