#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:52:16 2023

"""

#%%
from .vae import VAE
from .vitae_ci import VITAE_CI
from .vitae_ui import VITAE_UI

#%%
def get_model(model_name):
    models = {'vae': VAE,
              'vitae_ci': VITAE_CI,
              'vitae_ui': VITAE_UI,
              }
    assert (model_name in models), 'Model not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[model_name]