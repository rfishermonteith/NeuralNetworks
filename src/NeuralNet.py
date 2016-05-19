'''
Created on Apr 5, 2016

@author: richardf
'''
import numpy as np
#import scipy.stats as sp

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
         
        # Set step size for numerical integration   
        self.h_param = 0.000001
        # Set learning rate value
        self.learning_rate = 0.4
            
    def sim(self, inputs):
        """Simulates the feedforward network
        inputs is a list of inputs which corresponds to the number of inputs in the input layer"""
        
        tempOutputs = inputs
        for i in self.layers:
            tempOutputs = i.sim(tempOutputs)
        outputs = np.concatenate(tempOutputs)
        return outputs
    
    def train(self, inputs, outputs, num_epochs):
        """Trains the network
        inputs is a 2darray of inputs, rows are samples, columns are elements
        outputs is a 2darray of corresponding network outputs, rows are samples, columns are elements
        num_epochs is the number of epochs to run
        """
                
        # Loop through the epochs
        for i in range(num_epochs):
            # Find current cost
            current_outputs = [self.sim(inputs[:,n]) for n in range(np.size(inputs, 1))]
            current_outputs = np.concatenate(current_outputs,1)
            current_cost = cost(current_outputs, outputs)
            print "Cost for epoch %d: %f" % (i,current_cost)
            # Find the partial derivative of Cost w.r.t. each parameter
            for j in self.layers:
                for k in range(j.w.shape[0]):
                    # Weights
                    for m in range(j.w.shape[1]):
                        # Change weight by self.h_param
                        j.w[k,m] += self.h_param
                        # Simulate network
                        perturbed_outputs = [self.sim(inputs[:,n]) for n in range(np.size(inputs, 1))]
                        perturbed_outputs = np.concatenate(perturbed_outputs, 1)
                        # Find cost
                        perturbed_cost = cost(perturbed_outputs, outputs)
                        # Calculate partial cost
                        partial_cost = (perturbed_cost-current_cost)/self.h_param
                        # Return value of k.weight[k]
                        j.w[k,m] -= self.h_param
                        # Calculate update
                        j.w_to_update[k,m] = j.w[k,m] - self.learning_rate*partial_cost
                        
                    # Bias
                    # Change bias by self.h_param
                    j.b[k] += self.h_param
                    # Simulate network
                    perturbed_outputs = [self.sim(inputs[:,n]) for n in range(np.size(inputs, 1))]
                    perturbed_outputs = np.concatenate(perturbed_outputs, 1)
                    # Find cost
                    perturbed_cost = cost(perturbed_outputs, outputs)
                    # Calculate partial cost
                    partial_cost = (perturbed_cost-current_cost)/self.h_param
                    # Return value of k.bias[k]
                    j.b[k] -= self.h_param
                    # Calculate update
                    j.b_to_update[k] = j.b[k] - self.learning_rate*partial_cost
                    
            # Update the parameters  
            for j in self.layers:
                j.update_parameters()
                '''
                for k in j.nodes:
                    k.update_parameters()
                '''

class Layer:
    def __init__(self, num_nodes, num_inputs):
        """Initialises the Layer object
        num_nodes is the number of Nodes to be in the layer
        num_inputs is the number of outputs from the previous layer"""
        
        self.w = np.matrix(np.random.rand(num_nodes, num_inputs))
        self.w_to_update = np.zeros_like(self.w)
        self.b = np.matrix.transpose(np.matrix(np.random.rand(num_nodes)))
        self.b_to_update = np.zeros_like(self.b)
        '''
        self.nodes = [0 for i in range(num_nodes)]
        for i in range(num_nodes):
            self.nodes[i] = (Node(num_inputs))
        '''
    def sim(self, inputs):
        """Simulates the layer
        inputs is a list of inputs to the layer"""
        outputs = []
        
        z = self.w*inputs+self.b
        
        outputs = sigmoid(z)
        '''
        for i in self.nodes:
            outputs.append(i.sim(inputs))
        '''
        return outputs   
    def update_parameters(self):
        """Updates the weights and biases of the Layer"""
        self.w = np.matrix(self.w_to_update)
        self.b = np.matrix(self.b_to_update)      

def sigmoid(z):
    #output = sp.logistic.cdf(z)
    output = 1.0/(1.0+np.exp(-z))
    return output

def cost(simulated_outputs, actual_outputs):
    n = simulated_outputs.shape[1]
    if len(simulated_outputs.shape) == 1:
        cost_value = 1.0/(2.0*n) * np.sum(np.subtract(simulated_outputs,actual_outputs)**2)
    else:
        cost_value = 1.0/(2.0*n) * np.sum(np.linalg.norm(np.subtract(simulated_outputs,actual_outputs), axis=0)**2)
    return cost_value



# Code to test the cost function
#print cost(np.array([[3, 8, 9, 12]]).T, np.array([[4, 9, 7, 20]]).T)


# Code to test the network creation
# Fairly simple network

net = Network([5,2],4)
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
