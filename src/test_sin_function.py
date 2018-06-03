# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 15:47:23 2018

@author: richardf
"""
import math, numpy as np, NeuralNet as nn, matplotlib.pyplot as plt

# Test whether the NN can learn a sin function
train_inputs = np.random.rand(1,1000)
train_outputs = 0.5*np.sin(train_inputs*2*math.pi)+0.5

test_inputs = np.random.rand(1,10)
test_outputs =  0.5*np.sin(test_inputs*2*math.pi)+0.5

new_inputs = [np.arange(0,1,0.01)]

# Create the NN
net = nn.Network([20,5,1],1)

print(net.sim([[0]]))


print(net)

net.train(train_inputs, train_outputs, 500)

new_outputs = net.sim(new_inputs)

plt.plot(np.transpose(new_inputs), np.transpose(new_outputs))
plt.plot(np.transpose(train_inputs), np.transpose(train_outputs),'x')