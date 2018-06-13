# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:45:02 2018

@author: rfishy1
"""

import numpy as np

def sigmoid(z):
    #output = sp.logistic.cdf(z)
    output = 1.0/(1.0+np.exp(-z))
    return output