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
        self.h_param = 0.0000001
        # Set learning rate value
        self.learning_rate = 3.0
            
    def sim(self, inputs):
        """Simulates the feedforward network
        inputs is a 2darray of inputs which corresponds to the number of inputs in the input layer
        Columns in the inputs are samples
        Rows are elements
        Ie a single input should be a single column"""
        
        tempOutputs = inputs
        for i in self.layers:
            tempOutputs = i.sim(tempOutputs)
        outputs = np.concatenate(tempOutputs)
        return outputs
    
    def train(self, inputs, outputs, num_epochs):
        """Trains the network
        inputs is a 2darray of inputs, rows are elements, columns are samples
        outputs is a 2darray of corresponding network outputs, rows are elements, columns are samples
        num_epochs is the number of epochs to run
        """
                
        # Loop through the epochs
        for i in range(num_epochs):
            # Find current cost
            #current_outputs = [self.sim(inputs[:,n]) for n in range(np.shape(inputs)[1])]
            #current_outputs = np.concatenate(current_outputs,1)
            current_outputs = self.sim(inputs)
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
                        #perturbed_outputs = [self.sim(inputs[:,n]) for n in range(np.size(inputs, 1))]
                        #perturbed_outputs = np.concatenate(perturbed_outputs, 1)
                        perturbed_outputs = self.sim(inputs)
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
                    #perturbed_outputs = [self.sim(inputs[:,n]) for n in range(np.size(inputs, 1))]
                    #perturbed_outputs = np.concatenate(perturbed_outputs, 1)
                    perturbed_outputs = self.sim(inputs)
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
    def __str__(self):
        """Print the network"""
        
        # Loop through the layers
        a = ""
        for i in range(len(self.layers)):
            a+=str(self.layers[i])+"\n"
        #print(a)
        return a

class Layer:
    def __init__(self, num_nodes, num_inputs):
        """Initialises the Layer object
        num_nodes is the number of Nodes to be in the layer
        num_inputs is the number of outputs from the previous layer
        
        The weights w will be of size [num_nodes x num_inputs]
        The biases b will be of size [num_nodes x 1]"""
        a = 10
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.w = np.matrix(a*(np.random.normal(0,1,size = [num_nodes, num_inputs])))
        self.w_to_update = np.zeros_like(self.w)
        self.b = np.matrix.transpose(a*(np.matrix(np.random.normal(0,1,num_nodes))))
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
        
        outputs = np.tanh(z)
        # outputs = z
        '''
        for i in self.nodes:
            outputs.append(i.sim(inputs))
        '''
        return outputs   
    def update_parameters(self):
        """Updates the weights and biases of the Layer"""
        self.w = np.matrix(self.w_to_update)
        self.b = np.matrix(self.b_to_update)   
    
    def __str__(self):
        """Returns the details of this layer"""
        
        return "Layer has %s inputs and %s nodes, b: %s, w: %s" % (self.num_inputs, self.num_nodes, str(self.b), str(self.w))

def sigmoid(z):
    #output = sp.logistic.cdf(z)
    output = 1.0/(1.0+np.exp(-z))
    return output

def cost(simulated_outputs, actual_outputs):
    n = simulated_outputs.shape[1]
    if len(simulated_outputs.shape) == 1:
        cost_value = 1.0/(2.0*n) * np.sum(np.subtract(simulated_outputs,actual_outputs)**2)
    else:
        cost_value = 1.0/(2.0*n) * np.sum(np.linalg.norm(np.subtract(simulated_outputs,actual_outputs), axis=0))
    return cost_value
