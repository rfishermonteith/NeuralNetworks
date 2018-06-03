# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 15:45:08 2018

@author: richardf
"""

def load_mnist():
    # Load the MNIST data
    f = gzip.open('../mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set