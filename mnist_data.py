#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:06:24 2018

@author: alex
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class Normalize(object):
    
    def __init__(self, means, stds):
        """
        
        """
        
        self.means = means
        self.stds = stds
        self.std_eps = 10**-6 # if std is smaller than this number, std is not scaled.
        
        self.normalize_lambda = lambda xx: (xx - self.means) / np.where( self.stds > self.std_eps, self.stds, 1 )
        self.denormalize_lambda = lambda xx: ( xx * np.where( self.stds > self.std_eps, self.stds, 1 ) ) + self.means
        
        
    def normalize(self, inp):
            
        return self.normalize_lambda(inp)
    
    def denormalize(self, inp):
        
        return self.denormalize_lambda(inp)
        
    
    def test_normalization(self, xx):
        
        test_eps = np.finfo( xx.dtype.type ).eps * 10000
        
        assert np.all( np.abs(xx.mean(axis=0) - 0.0 ) < test_eps ), "Means are wrong"
        
        stds = xx.std(axis=0)
        assert np.all( np.abs( np.where(stds > test_eps, stds, 1) - 1.0) < test_eps   ), "Stds are wrong"
        
        print("Data is properly normalized to 0 mean, 1 variance")


def get_mnist_data( data_dir = None, normalized = False, train_batch_size=1, test_batch_size=1, train_shuffle=True):
    """
    Returns the iterators which can iterate over the training and test data.
    Inputs:
    ------------------
    
    data_dir: str
        Data dir. If none the standard one is used
        
    normalized: bool
        If false the data is between 0 and 1 as pytorch returns. Otherwise it
        if normalized to 0 mean and 1 variance.
    """

    if data_dir is None:
        data_dir = '/Users/alex/Yandex.Disk.localized/WorkDocuments/Projects/Silo/mnist_mlp/data'
    
    
    train_transform_list = [transforms.ToTensor(), transforms.Lambda(lambda aa: aa.numpy().reshape( ( 28*28,) ) ) ]

    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose( train_transform_list ))
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose( train_transform_list ))
    
    if normalized:
        dd = np.array( [ ii[0] for ii in train_set] )
        train_means = dd.mean(axis=0)
        train_stds = dd.std(axis=0)

        norm_object = Normalize(train_means, train_stds)
    
        train_transform_list.append( transforms.Lambda( norm_object.normalize_lambda ) )
        
        train_set.transform = transforms.Compose( train_transform_list )
        test_set.transform = transforms.Compose( train_transform_list )
        
        # Test the normalization
        ddn = np.array( [ ii[0] for ii in train_set] )
        norm_object.test_normalization(ddn)
    
    #torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, 
    #num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size = train_batch_size, shuffle = train_shuffle, num_workers=4 )   
    test_iter = torch.utils.data.DataLoader(test_set, batch_size = test_batch_size, shuffle = False, num_workers=4 )
     
    return train_iter, test_iter
