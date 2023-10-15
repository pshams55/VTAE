#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 08:13:22 2023

"""
#%%
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import time, os, datetime
from tensorboardX import SummaryWriter
from .helper.losses import vae_loss

#%%
class vae_trainer:
    """ Main class for training the vae models 
    Arguments:
        input_shape: shape of a single image
        model: model (of type torch.nn.Module) to train
        optimizer: optimizer (of type torch.optim.Optimizer) that will be used 
            for the training
    Methods:
        fit - for training the network
        save_embeddings - embeds data into the learned spaces, saves to tensorboard
    """
    def __init__(self, input_shape, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.outputdensity = model.outputdensity
        self.use_cuda = True
        
        # Get the device
        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        # Move model to gpu (if avaible)
        if torch.cuda.is_available() and self.use_cuda:
            self.model.cuda()
    
    #%%
    def fit(self, trainloader, n_epochs=10, warmup=1, logdir='',
            testloader=None, eq_samples=1, iw_samples=1, beta=1.0, eval_epoch=10000):
       
        # Assert that input is okay
        assert isinstance(trainloader, torch.utils.data.DataLoader), '''Trainloader
            should be an instance of torch.utils.data.DataLoader '''
        assert warmup <= n_epochs, ''' Warmup period need to be smaller than the
            number of epochs '''
        
        # Print stats
        print('Number of training points: ', len(trainloader.dataset))
        if testloader: print('Number of test points:     ', len(testloader.dataset))
        
        # Dir to log results
        logdir = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if logdir is None else logdir
        if not os.path.exists(logdir): os.makedirs(logdir)
        
        # Summary writer
        writer = SummaryWriter(log_dir=logdir)
        
        # Main loop
        start = time.time()
        for epoch in range(1, n_epochs+1):
            progress_bar = tqdm(desc='Epoch ' + str(epoch) + '/' + str(n_epochs), 
                                total=len(trainloader.dataset), unit='samples')
            train_loss = 0
            # Training loop
            self.model.train()
            for i, (data, _) in enumerate(trainloader):
                # Zero gradient
                self.optimizer.zero_grad()
            
                # Feed forward data
                data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
                switch = 1.0 if epoch > warmup else 0.0
                out = self.model(data, eq_samples, iw_samples, switch)
                
                # Calculat loss
                loss, recon_term, kl_terms = vae_loss(data, *out, 
                                                      eq_samples, iw_samples, 
                                                      self.model.latent_dim, 
                                                      epoch, warmup, beta,
                                                      self.outputdensity)
                train_loss += float(loss.item())
                
                # Backpropegate and optimize
                # We need to maximize the bound, so in this case we need to
                # minimize the negative bound
                (-loss).backward()
                self.optimizer.step()
                
                # Write to consol
                progress_bar.update(data.size(0))
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Save to tensorboard
                iteration = epoch*len(trainloader) + i
                writer.add_scalar('train/total_loss', loss, iteration)
                writer.add_scalar('train/recon_loss', recon_term, iteration)
                
                for j, kl_loss in enumerate(kl_terms):
                    writer.add_scalar('train/KL_loss' + str(j), kl_loss, iteration)
                del loss, recon_term, kl_loss, out
                
            progress_bar.set_postfix({'Average ELBO': train_loss / len(trainloader)})
            progress_bar.close()
            
            # Log for the training set
            with torch.no_grad():
                n = 10
                data_train = next(iter(trainloader))[0].to(torch.float32).to(self.device)
                data_train = data[:n].reshape(-1, *self.input_shape)
                recon_data_train = self.model(data_train)[0]
                writer.add_image('train/recon', make_grid(torch.cat([data_train, 
                             recon_data_train]).cpu(), nrow=n), global_step=epoch)
                samples = self.model.sample(n*n)    
                writer.add_image('samples/samples', make_grid(samples.cpu(), nrow=n), 
                                 global_step=epoch)
                del data_train, recon_data_train, samples
            
            if testloader:
                with torch.no_grad():
                    # Evaluate on test set (L1 log like)
                    self.model.eval()
                    test_loss, test_recon, test_kl = 0, 0, len(kl_terms)*[0]
                    for i, (data, _) in enumerate(testloader):
                        data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
                        out = self.model(data, 1, 1)    
                        loss, recon_term, kl_terms = vae_loss(data, *out, 1, 1, 
                                                              self.model.latent_dim, 
                                                              epoch, warmup, beta,
                                                              self.outputdensity)
                        test_loss += loss.item()
                        test_recon += recon_term.item()
                        test_kl = [l1+l2 for l1,l2 in zip(kl_terms, test_kl)]
            
                    writer.add_scalar('test/total_loss', test_loss, iteration)
                    writer.add_scalar('test/recon_loss', recon_term, iteration)
                    for j, kl_loss in enumerate(kl_terms):
                        writer.add_scalar('test/KL_loss' + str(j), kl_loss, iteration)
            
                    data_test = next(iter(testloader))[0].to(torch.float32).to(self.device)[:n]
                    data_test = data_test.reshape(-1, *self.input_shape)
                    recon_data_test = self.model(data_test)[0]
                    writer.add_image('test/recon', make_grid(torch.cat([data_test, 
                             recon_data_test]).cpu(), nrow=n), global_step=epoch)
                    if (epoch==n_epochs):
                        print('Final test loss', test_loss)
                    del data, out, loss, recon_term, kl_terms, data_test, recon_data_test

                    # Callback, if a model have something special to log
                    self.model.callback(writer, testloader, epoch)
                    
                    # If testset and we are at a eval epoch (or last epoch), 
                    # calculate L5000 (very expensive to do)
                    if (epoch % eval_epoch == 0) or (epoch==n_epochs):
                        progress_bar = tqdm(desc='Calculating log(p(x))', 
                                            total=len(testloader.dataset), unit='samples')
                        test_loss, test_recon, test_kl = 0, 0, self.model.latent_spaces*[0]
                        for i, (data, _) in enumerate(testloader):
                            data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
                            # We need to do this for each individual points, because
                            # iw_samples is high (running out of GPU memory)
                            for d in data:
                                out = self.model(d[None], 1, 1000)
                                loss, _, _ = vae_loss(d, *out, 1, 1000  ,
                                                      self.model.latent_dim, 
                                                      epoch, warmup, beta,
                                                      self.outputdensity)
                                test_loss += loss.item()
                                progress_bar.update()
                        progress_bar.close()
                        writer.add_scalar('test/L5000', test_loss, iteration)
                        
        print('Total train time', time.time() - start)
        
        # Save the embeddings
        print('Saving embeddings, maybe?')
        with torch.no_grad():
            try:
                self.save_embeddings(writer, trainloader, name='train')
            except Exception as e:
                print('Did not save embeddings for training set')
                print(e)
            if testloader: 
                try:
                    self.save_embeddings(writer, testloader, name='test')
                except Exception as e:
                    print('Did not save embeddings for test set')
                    print(e)

        # Close summary writer
        writer.close()
        
    #%%
    def save_embeddings(self, writer, loader, name='embedding'):
        # Constants
        N = len(loader.dataset)
        m = self.model.latent_spaces
        
        # Data structures for holding the embeddings
        all_data = torch.zeros(N, *self.input_shape, dtype=torch.float32)
        all_label = torch.zeros(N, dtype=torch.int32)
        all_latent = [ ]
        for j in range(m):
            all_latent.append(torch.zeros(N, self.model.latent_dim, dtype=torch.float32))
        
        # Loop over all data and get embeddings
        counter = 0
        for i, (data, label) in enumerate(loader):
            n = data.shape[0]
            data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
            label = label.to(self.device)
            z = self.model.latent_representation(data)
            all_data[counter:counter+n] = data.cpu()
            for j in range(m):
                all_latent[j][counter:counter+n] = z[j].cpu()
            all_label[counter:counter+n] = label.cpu()
            counter += n
            
        # Save the embeddings
        for j in range(m):
            
            # Embeddings with dim < 3 needs to be appended extra non-informative dimensions
            N, n = all_latent[j].shape
            if n == 1:
                all_latent[j] = torch.cat([all_latent[j], torch.zeros(N, 2)], dim = 1)
            if n == 2:
                all_latent[j] = torch.cat([all_latent[j], torch.zeros(N, 1)], dim = 1)
            
            # Maximum bound for the sprite image
            writer.add_embedding(mat = all_latent[j],
                                 metadata = all_label,
                                 label_img = all_data,
                                 tag = name + '_latent_space' + str(j))
        
        
