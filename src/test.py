# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 20:11:36 2018

@author: rfishy1
"""

# Visualise the network behaviour
import numpy as np, NeuralNet as nn, matplotlib.pyplot as plt

np.random.seed(6)
net = nn.Network([1,200,50,2])

print(net.sim(np.array([[0]])))

inputs = np.array([np.arange(-10.0,10.0,0.1)])
outputs = net.sim(inputs)
    

plt.plot(np.transpose(inputs), np.transpose(outputs))