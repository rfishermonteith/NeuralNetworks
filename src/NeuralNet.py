# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 19:30:54 2018

@author: rfishy1
"""

import numpy as np, utils as ut

class Network:
    def __init__(self, layer_sizes):
        """Initialises a Network object
        layer_sizes is a list of the number of nodes in each layer
        The first element is the number of inputs"""
        
        num_nodes = layer_sizes[1:]
        num_inputs = layer_sizes[0:-1]
        self.layers = []
        for k in range(len(num_nodes)):
            self.layers.append(Layer(num_nodes[k], num_inputs[k]))
    
    def sim(self, inputs):
        a = inputs
        for k in self.layers:
            a = k.sim(a)
        return a
    
    def cost(self, inputs, outputs):
        a = self.sim(inputs)
        cost = 1/(2*inputs.shape[1])*np.sum(np.linalg.norm(outputs-a,axis=0),axis=0)
        return cost
    
class Layer:
    def __init__(self, num_nodes, num_inputs):
        """Initialises a Layer object"""
        self.nodes = []
        self.nodes = [Neuron(num_inputs) for x in range(num_nodes)]
    
    def sim(self, inputs):
        """Simulates the layer"""
        a = np.concatenate([x.sim(inputs) for x in self.nodes], axis=0)
        return a
    
class Neuron:
    """A neural network layer comprises a number of neurons"""
    
    def __init__(self, num_inputs):
        self.w = np.random.normal(size=[1,num_inputs])
        self.b = np.random.normal(size=[1,1])
        
    def sim(self, inputs):
        a = ut.sigmoid(np.dot(self.w,inputs)+self.b)
        return a
     

        
