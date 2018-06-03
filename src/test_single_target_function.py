# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:38:55 2018

@author: richardf
"""

import math, numpy as np, NeuralNet as nn, matplotlib.pyplot as plt

# Test whether the NN can learn a single target function
train_inputs = np.arange(0,1.001,0.5)
#train_outputs = -(train_inputs-0.5)**(2)/10.0+0.5
#train_outputs = train_inputs*(1.0)
train_outputs = np.abs(train_inputs -0.5)
                             
new_inputs = [np.arange(0,1,0.01)]

# Create the NN
net = nn.Network([5,5,1],1)

print(net.sim([[0]]))


print(net)

net.train(train_inputs, train_outputs, 500)

new_outputs = net.sim(new_inputs)

plt.plot(np.transpose(new_inputs), np.transpose(new_outputs))
plt.plot(train_inputs, train_outputs)
plt.legend(('Predictions','Training data'))

#%% Try create a network

net = nn.Network([2,1],1)

#net.layers[0].b[0] = -0.8
#net.layers[0].b[1] = 0
#net.layers[0].w[0] = -10
#net.layers[0].w[1] = 1

#net.layers[1].b[0] = 0
#net.layers[1].w[0,0] = 1
#net.layers[1].w[0,1] = 1

plt.plot(np.transpose(new_inputs), np.transpose(net.sim(new_inputs)))
#plt.plot(train_inputs, train_outputs)
plt.legend(('Predictions','Training data'))