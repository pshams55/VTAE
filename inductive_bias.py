# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 07:30:05 2019

@author: nsde
"""

#%%
import torch
import argparse, datetime
from torchvision import transforms
import numpy as np

from sup.trainer import vae_trainer
from sup.data.mnist_data_loader import mnist_data_loader
from sup.data.perception_data_loader import perception_data_loader
from sup.helper.utility import model_summary
from sup.helper.encoder_decoder import get_encoder, get_decoder
from sup.models import get_model

#%%
def argparser():
    """ Argument parser for the main script """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model settings
    ms = parser.add_argument_group('Model settings')
    ms.add_argument('--model', type=str, default='vae', help='model to train')
    ms.add_argument('--ed_type', type=str, default='mlp', help='encoder/decoder type')
    ms.add_argument('--stn_type', type=str, default='affine', help='transformation type to use')
    ms.add_argument('--beta', type=float, default=1.0, help='beta value for beta-vae model')
    
    # Training settings
    ts = parser.add_argument_group('Training settings')
    ts.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    ts.add_argument('--eval_epoch', type=int, default=1000, help='when to evaluate log(p(x))')
    ts.add_argument('--batch_size', type=int, default=512, help='size of the batches')
    ts.add_argument('--warmup', type=int, default=50, help='number of warmup epochs for kl-terms')
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
    ds.add_argument('--logdir', type=str, default='idb_test', help='where to store results')
    ds.add_argument('--dataset', type=str, default='mnist', help='dataset to use')
    
    # Parse and return
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    # Input arguments
    args = argparser()
    
    # Logdir for results
    logdir = 'res/final'
    
    # Load data
    print('Loading data')
    transformations = transforms.Compose([ 
        #transforms.RandomAffine(degrees=20, translate=(0.1,0.1)), 
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
    model_class = get_model('vitae_ci')
    model = model_class(input_shape = img_size,
                        latent_dim = 2, 
                        encoder = get_encoder(args.ed_type), 
                        decoder = get_decoder(args.ed_type), 
                        outputdensity = args.density,
                        ST_type = args.stn_type)
    
    # Summary of model
    #model_summary(model)
    print('model 1')
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    Trainer = vae_trainer(img_size, model, optimizer)
    Trainer.fit(trainloader=trainloader, 
                n_epochs=args.n_epochs, 
                warmup=args.warmup, 
                logdir=logdir,
                testloader=testloader,
                eq_samples=args.eq_samples, 
                iw_samples=args.iw_samples, 
                beta=args.beta,
                eval_epoch=args.eval_epoch)
    
    # Save model
    torch.save(model.state_dict(), logdir + '/trained_model.pt')
    
    #%% build new dataset
    class sample_data(torch.utils.data.Dataset):
        def __init__(self, n, model=None):
            self.n = n
            if model:
                X=np.zeros((n, 1, 28, 28))
                y=np.zeros((n, ))
                self.latent = [np.zeros((n, 2)) for _ in range(model.latent_spaces)]
                for b in range(int(n/100)):
                    x, zs = model.special_sample(100)
                    X[100*b:100*(b+1)] = x.cpu()
                    for i in range(model.latent_spaces):
                        self.latent[i][100*b:100*(b+1)] = zs[i].cpu()
                self.data = torch.tensor(X)
                self.targets = torch.tensor(y)
        
        def fromfiles(self, name_x, name_y):
            self.data = torch.load(name_x)
            self.targets = torch.load(name_y)
        
        def __getitem__(self, index):
            img, target = self.data[index], int(self.targets[index])
            return img, target

        def __len__(self):
            return self.n
                
    train = sample_data(60000, model)
    torch.save(train.data, 'samples_x.pt')
    torch.save(train.targets, 'samples_y.pt')
    #train = sample_data(60000, None)
    #train.fromfiles('samples_x.pt', 'samples_y.pt')
    
    trainloader = torch.utils.data.DataLoader(train, batch_size = args.batch_size)
    #testloader = torch.utils.data.DataLoader(test, batch_size = 500)
    
    #%% train other models
    # Construct vae
    model_class2 = get_model('vae')
    model2 = model_class2(input_shape = img_size,
                          latent_dim = 4, 
                          encoder = get_encoder(args.ed_type), 
                          decoder = get_decoder(args.ed_type), 
                          outputdensity = args.density,
                          ST_type = args.stn_type)
    print('model 2')
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)
    Trainer2 = vae_trainer(img_size, model2, optimizer2)
    Trainer2.fit(trainloader=trainloader, 
                 n_epochs=args.n_epochs, 
                 warmup=args.warmup, 
                 logdir=logdir,
                 testloader=None,
                 eq_samples=args.eq_samples, 
                 iw_samples=args.iw_samples,
                 beta=1.0,
                 eval_epoch=args.eval_epoch)
    torch.save(model2.state_dict(), logdir + '/trained_model2.pt')
    
#   Construct beta-vae
    model_class3 = get_model('vae')
    model3 = model_class3(input_shape = img_size,
                          latent_dim = 4, 
                          encoder = get_encoder(args.ed_type), 
                          decoder = get_decoder(args.ed_type), 
                          outputdensity = args.density,
                          ST_type = args.stn_type)
    print('model 3')
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=args.lr)
    Trainer3 = vae_trainer(img_size, model3, optimizer3)
    Trainer3.fit(trainloader=trainloader, 
                 n_epochs=args.n_epochs, 
                 warmup=args.warmup, 
                 logdir=logdir,
                 testloader=None,
                 eq_samples=args.eq_samples, 
                 iw_samples=args.iw_samples,
                 beta=8.0,
                 eval_epoch=args.eval_epoch)
    torch.save(model3.state_dict(), logdir + '/trained_model3.pt')
    
    # Construct vitae (again)
    model_class4 = get_model('vitae_ci')
    model4 = model_class4(input_shape = img_size,
                          latent_dim = 2, 
                          encoder = get_encoder(args.ed_type), 
                          decoder = get_decoder(args.ed_type), 
                          outputdensity = args.density,
                          ST_type = args.stn_type)
    print('model 4')
    optimizer4 = torch.optim.Adam(model4.parameters(), lr=args.lr)
    Trainer4 = vae_trainer(img_size, model4, optimizer4)
    Trainer4.fit(trainloader=trainloader, 
                 n_epochs=args.n_epochs, 
                 warmup=args.warmup, 
                 logdir=logdir,
                 testloader=None,
                 eq_samples=args.eq_samples, 
                 iw_samples=args.iw_samples,
                 beta=1.0,
                 eval_epoch=args.eval_epoch)
    torch.save(model4.state_dict(), logdir + '/trained_model4.pt')
    
    #%% save latent codes
    latent1 = train.latent
    latent2 = [np.zeros((60000, 4))]
    latent3 = [np.zeros((60000, 4))]
    latent4 = [np.zeros((60000, 2)), np.zeros((60000, 2))]
    
    counter = 0
    for (X, _) in trainloader:
        X = X.reshape(-1, 1, 28, 28).to(torch.float32).cuda()
        _, _, _, zs2, _ = model2.semantics(X)
        _, _, _, zs3, _ = model3.semantics(X)
        _, _, _, zs4, _ = model4.semantics(X)
        n = X.shape[0]
        latent2[0][counter:counter+n] = zs2[0].detach().cpu()
        latent3[0][counter:counter+n] = zs3[0].detach().cpu()
        latent4[0][counter:counter+n] = zs4[0].detach().cpu()
        latent4[1][counter:counter+n] = zs4[1].detach().cpu()
        counter += n

    np.save(logdir + '/latent1', latent1)
    np.save(logdir + '/latent2', latent2)
    np.save(logdir + '/latent3', latent3)
    np.save(logdir + '/latent4', latent4)    
    
    
