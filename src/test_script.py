# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 13:09:52 2017

@author: richardf
"""

import NeuralNet as nn
import numpy as np
import matplotlib.pyplot as plt

# Code to test the cost function
#print cost(np.array([[3, 8, 9, 12]]).T, np.array([[4, 9, 7, 20]]).T)


# Code to test the network creation
# Fairly simple network

net = nn.Network([5,2],4)
training_input = np.matrix([[1, 1, 3], [1, 2, 6], [4, 6, 2], [2, 8, 4]])
training_output = np.matrix([[1, 0, 0.5], [0, 1, 0.8]])

# Simple 1-node network
'''
net = Network([1],1)
training_input = np.array([1])
training_output = np.array([0.6])
'''


net.train(training_input, training_output, 1000)
print np.concatenate([net.sim(training_input[:,n]) for n in range(np.size(training_input, 1))], 1) 

#%%
'''
Create a function-learning NN
'''

reload(nn)

# function: y = sin(x)*cos(x)+x**3
x = np.arange(0,1,0.01)
y = 0.25*(np.sin(7*x)*np.cos(10*x)+(x)**3+1)

# fig, ax = plt.subplots()
# ax.cla()
# plt.figure()
plt.ion()
fig,ax = plt.subplots(1,1)
ax.plot(x,y)

training_input = np.matrix(x)
training_output = np.matrix(y)

net = nn.Network([20,10,1],1)



plt.hold(True)

for i in range(2):
    net.train(training_input, training_output, 10)
    
    # Plot the current approximation
    out = net.sim(training_input)

    ax.plot(x,out.T)
    plt.draw()
    
#%%
net = nn.Network([20,10, 6, 30, 3,1],1)

reload(nn)

#a = 50
#net.layers[0].w[0] = a*(np.random.rand(1)-0.5)
#net.layers[0].w[1] = a*(np.random.rand(1)-0.5)
#net.layers[0].w[2] = a*(np.random.rand(1)-0.5)
#net.layers[0].w[3] = a*(np.random.rand(1)-0.5)
#net.layers[0].w[4] = a*(np.random.rand(1)-0.5)

#net.layers[0].b[0] = a*(np.random.rand(1)-0.5)
#net.layers[0].b[1] = a*(np.random.rand(1)-0.5)
#net.layers[0].b[2] = a*(np.random.rand(1)-0.5)
#net.layers[0].b[3] = a*(np.random.rand(1)-0.5)
#net.layers[0].b[4] = a*(np.random.rand(1)-0.5)

#net.layers[1].w = a*(np.random.rand(1,5)-0.5)
#net.layers[1].b = a*(np.random.rand(1)-0.5)
plt.cla()
plt.plot(x,net.sim(x).T)
