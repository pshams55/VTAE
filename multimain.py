#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 07:59:17 2019

@author: nsde
"""

#%% 
import os 
import sys

counter = 0

#%%
def run(command, num=''):
    global counter
    counter += 1
    try:
        os.system(command)
    except Exception as e:
        print("Incountered error in command", counter)
        print(e)
        sys.exit()
    
 
#%% MNIST

# Affine
run("python main.py --model vae      --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist")
run("python main.py --model vitae_ci --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist --stn_type affine")
run("python main.py --model vitae_ui --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist --stn_type affine")
#
## Affine decomp
run("python main.py --model vae      --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist")
run("python main.py --model vitae_ci --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist --stn_type affinedecomp")
run("python main.py --model vitae_ui --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist --stn_type affinedecomp")
#
## Affine diff
run("python main.py --model vae      --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist")
run("python main.py --model vitae_ci --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist --stn_type affinediff")
run("python main.py --model vitae_ui --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist --stn_type affinediff")
#
## CPAB
run("python main.py --model vae      --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist")
run("python main.py --model vitae_ci --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist --stn_type cpab")
run("python main.py --model vitae_ui --n_epochs 5 --num_points 1000 --logdir ft --dataset mnist --stn_type cpab")
#
##%% PERCEPTION
## Affine
#run("python main.py --model vae      --n_epochs 5 --num_points 1000 --logdir ft --dataset perception")
#run("python main.py --model vitae_ci --n_epochs 5 --num_points 1000 --logdir ft --dataset perception --stn_type affine")
#run("python main.py --model vitae_ui --n_epochs 5 --num_points 1000 --logdir ft --dataset perception --stn_type affine")
#
## Affine decomp
#run("python main.py --model vae      --n_epochs 5 --num_points 1000 --logdir ft --dataset perception")
#run("python main.py --model vitae_ci --n_epochs 5 --num_points 1000 --logdir ft --dataset perception --stn_type affinedecomp")
#run("python main.py --model vitae_ui --n_epochs 5 --num_points 1000 --logdir ft --dataset perception --stn_type affinedecomp")
#
## Affine diff
#run("python main.py --model vae      --n_epochs 5 --num_points 1000 --logdir ft --dataset perception")
#run("python main.py --model vitae_ci --n_epochs 5 --num_points 1000 --logdir ft --dataset perception --stn_type affinediff")
#run("python main.py --model vitae_ui --n_epochs 5 --num_points 1000 --logdir ft --dataset perception --stn_type affinediff")
#
## CPAB
#run("python main.py --model vae      --n_epochs 5 --num_points 1000 --logdir ft --dataset perception")
#run("python main.py --model vitae_ci --n_epochs 5 --num_points 1000 --logdir ft --dataset perception --stn_type cpab")
#run("python main.py --model vitae_ui --n_epochs 5 --num_points 1000 --logdir ft --dataset perception --stn_type cpab")