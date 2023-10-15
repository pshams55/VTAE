#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:31:08 2019

@author: nsde
"""

#%%
import argparse, os, sys

#%%
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help='experiment to run')
    args = parser.parse_args()
    return args

#%%
experiments = [
    
    ## Stability experiments    
    # Affine stn    
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir a4 --stn_type affine --lr 1e-4",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir a3 --stn_type affine --lr 1e-3",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir a2 --stn_type affine --lr 1e-2",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir a1 --stn_type affine --lr 1e-1",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir a0 --stn_type affine --lr 1e-0",
    # Decompose affine stn
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir c4 --stn_type affinedecomp --lr 1e-4",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir c3 --stn_type affinedecomp --lr 1e-3",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir c2 --stn_type affinedecomp --lr 1e-2",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir c1 --stn_type affinedecomp --lr 1e-1",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir c0 --stn_type affinedecomp --lr 1e-0",
    # Diffio affine stn           
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir d4 --stn_type affinediff --lr 1e-4",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir d3 --stn_type affinediff --lr 1e-3",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir d2 --stn_type affinediff --lr 1e-2",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir d1 --stn_type affinediff --lr 1e-1",
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir d0 --stn_type affinediff --lr 1e-0",
    
    ]

if __name__ == '__main__':
    args = argparser()
    command = experiments[args.n]
    try:
        os.system(command)
    except Exception as e:
        print("Incountered error in command", args.n)
        print(e)
        sys.exit()

