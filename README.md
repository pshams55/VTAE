# VTAE
Variational Transformer Autoencoder with Manifolds Learning

## Introduction
This is a PyTorch implementation for our paper "Variational Transformer Autoencoder with Manifolds Learning". It is an open-source codebase for image interpolations and reconstructions.

## Installation
We test this repo with Python 3.8, PyTorch 1.9.0, and CUDA 11.1. However, it should be runnable with recent PyTorch versions (Pytorch >= 1.1.0).
```shell
python setup.py develop
```
In addition, we need to install KNN_CUDA on the wheel.
```shell
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Pre-trained Weight

The file tree we used for storing the pre-trained weights is like
```shell
logs
├── pretrained.pth.tar
```

**(1) Celeba-pretrained weights for CNNs backbone**

The Celeba-pretrained weights for CNN's backbone or the pretrained weights for the model.

## Training
Train by running a script in the terminal. train.py


In the train.py.
In case you want to speed up testing, enlarge GPUS for more GPUs, or enlarge the --tuple-size for more tuples on one GPU.
In case your GPU does not have enough memory, reduce --pos-num or --neg-num for fewer positives or negatives in one tuple.

## Testing
Test by running a script in the terminal. Script test.py



## References
If you find our work useful for your research, please consider citing our paper:
```bibtex
@article{shamsolmoali2023vtae,
  title={VTAE: Variational Transformer Autoencoder with Manifolds Learning},
  author={Shamsolmoali, Pourya and Zareapoor, Masoumeh and Zhou, Huiyu and Tao, Dacheng and Li, Xuelong},
  journal={IEEE Transactions on Image Processing},
  year={2023}
}
```
