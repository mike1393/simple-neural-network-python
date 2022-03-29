# Third Party Packages
import numpy as np
# Build-in Packages
import random
# Local Packages
from neural_network.activation_function import sigmoid, d_sigmoid

class NeuralNetwork:
    # class members:
    # @neurons: a list of neuron numbers at each layer
    # @weights: a list of weight matrix between layers
    # @biases: a list of bias matrix for layers[1:]
    def __init__(self, neurons):
        self.neurons = neurons
        self.layer_num = len(self.neurons)
        self.weights = [np.random.randn(j,i) for i,j in zip(self.neurons[:-1], self.neurons[1:])]
        self.biases = [np.random.randn(j,1) for j in self.neurons[1:]]
    
    def feed_forward(self, activations):
        # update the activation value from first layer 
        # all the way to last layer
        for weight, bias in zip(self.weights, self.biases):
            activations = np.dot(weight, activations) + bias
        return activations

    # back proporgation
    def back_propagation(self):
        pass

    # Stochastic Gradient Descent
    def sgd(self):
        pass

    def d_cost(self):
        pass
    
