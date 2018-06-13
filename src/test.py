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

#%% Find the cost for a sin curve
np.random.seed(6)
net = nn.Network([1,20,2])

inputs = np.array([np.arange(-10.0,10.0,0.1)])
outputs = np.concatenate((0.5*np.sin(inputs)+0.5,0.5*np.cos(inputs)+0.5))

print(net.cost(inputs, outputs))

plt.plot(np.transpose(inputs), np.transpose(outputs))
plt.plot(np.transpose(inputs), np.transpose(net.sim(inputs)))


#%% Try training the network
np.random.seed(7)
net = nn.Network([1,10,1])

inputs = np.array([np.arange(-3.0,3.0,0.2)])
outputs =0.5*np.sin(inputs)+0.5

for k in range(10):
    cost = net.train(inputs, outputs, 1000)
    plt.plot(np.transpose(inputs), np.transpose(outputs))
    plt.plot(np.transpose(inputs), np.transpose(net.sim(inputs)))
    plt.pause(0.05)
    
    print(cost)
