'''
Created on Apr 5, 2016

@author: richardf
'''
import numpy as np

class Network:
    def __init__(self, layer_sizes, num_inputs):
        """Initialises a Network object
        layer_sizes is a list of the number of nodes in each layer (excluding the input layer)
        num_inputs is the number of input nodes"""
        
        self.layers = [0 for i in range(len(layer_sizes))]
        """Create Layer objects for each layer"""
        for i in range(len(layer_sizes)):
            if i == 0:
                num_inputs_layer = num_inputs
            else:
                num_inputs_layer = layer_sizes[i-1]
            new_layer = Layer(layer_sizes[i], num_inputs_layer)
            self.layers[i] = new_layer
            
    def sim(self, inputs):
        """Simulates the feedforward network
        inputs is a list of inputs which corresponds to the number of inputs in the input layer"""
        
        tempOutputs = inputs
        for i in self.layers:
            tempOutputs = i.sim(tempOutputs)
        outputs = tempOutputs
        return outputs
    

class Layer:
    def __init__(self, num_nodes, num_inputs):
        """Initialises the Layer object
        num_nodes are the number of Node objects to be in the layer
        num_inputs are the number of outputs from the previous layer"""
        self.nodes = [0 for i in range(num_nodes)]
        for i in range(num_nodes):
            self.nodes[i] = (Node(num_inputs))
            
    def sim(self, inputs):
        """Simulates the layer
        inputs is a list of inputs to the layer"""
        outputs = []
        
        for i in self.nodes:
            outputs.append(i.sim(inputs))
        return outputs        

class Node:
    def __init__(self, num_inputs):
        """Initialise the Node object
        num_inputs is the number of inputs to the node"""
        
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn(1) 
        
    def sim(self, inputs):
        """Simulates the node
        inputs is a list of inputs to the node"""
        z = np.dot(self.weights, inputs)+self.bias
        output = sigmoid(z)
        return output
    
def sigmoid(z):
    output = 1.0/(1.0+np.exp(-z))
    return output

def cost(simulated_outputs, actual_outputs):
    n = len(simulated_outputs)
    if len(simulated_outputs.shape) == 1:
        cost_value = 1.0/n * np.sum(np.subtract(simulated_outputs,actual_outputs)**2)
    else:
        cost_value = 1.0/n * np.sum(np.linalg.norm(np.subtract(simulated_outputs,actual_outputs), axis=1)**2)
    return cost_value



# Code to test the cost function
print cost(np.array([[3, 8, 9, 12]]).T, np.array([[4, 9, 7, 20]]).T)


# Code to test the network creation
net = Network([3, 4],2)
print net.sim([1, 2])