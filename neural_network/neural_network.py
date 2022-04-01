# Third Party Packages
import numpy as np
# Build-in Packages
import random
import pickle
import os
# Local Packages
import neural_network.activation_function as af

def model_loader(path):
    with open(path , 'rb') as f:
        return pickle.load(f)

class NeuralNetwork:
    # class members:
    # @neurons: a list of neuron numbers at each layer
    # @weights: a list of weight matrix between layers
    # @biases: a list of bias matrix for layers[1:]
    def __init__(self, neurons, af_name):
        self.neurons = neurons
        self.layer_num = len(self.neurons)
        self.weights = [np.random.randn(j,i) for i,j in zip(self.neurons[:-1], self.neurons[1:])]
        self.biases = [np.random.randn(j,1) for j in self.neurons[1:]]
        self.af, self.d_af = af.activate_method[af_name]

    
    def feed_forward(self, input_layer):
        # update the activation value from first layer 
        # all the way to last layer
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, input_layer) + bias
            input_layer = self.af(z)
        return input_layer

    # back proporgation find the gradient from input data
    def back_propagation(self, x,y):
        # feed forward
        # feed x into the network
        # store each layer into a list as we move forward
        # store each z into a list as we move forward
        layer = x
        layers = [x]
        z_for_each_layer = []
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, layer) + bias
            z_for_each_layer.append(z)
            layer = self.af(z)
            layers.append(layer)
        # backward pass
        # initialize zero matrix for nabla_w and nabla_b
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        # calculate d_cost_z = d_a_z * d_cost_a
        d_cost_z = (layers[-1] - y)*self.d_af(z_for_each_layer[-1])
        # calculate the last layer nabla for weight and bias
        nabla_w[-1] = np.dot( d_cost_z, layers[-2].transpose())
        nabla_b[-1] = d_cost_z
        for idx in range(2, self.layer_num):
            # d_cost_z^L-i = d_a_z^L-i * d_cost_a^L-i
            # d_cost_a^L-i = d_z_a^L-i * d_cost_z
            d_cost_z = np.dot(self.weights[-idx+1].transpose(), d_cost_z) * self.d_af(z_for_each_layer[-idx])
            nabla_w[-idx] = np.dot( d_cost_z, layers[-idx-1].transpose())
            nabla_b[-idx] = d_cost_z
        return (nabla_w, nabla_b)
    
    # update weights and biases from mini batches
    def update_feature(self, mini_batch, eta):
        # initialize an overall nabla for weights and biases
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        # for each training data
        for x,y in mini_batch:
            # get a mini nabla and add to overall nabla
            nabla_w_per_data, nabla_b_per_data = self.back_propagation(x,y)
            nabla_w = [w+nw for w, nw in zip(nabla_w, nabla_w_per_data)]
            nabla_b = [b+nb for b, nb in zip(nabla_b, nabla_b_per_data)]
        # once done, update weights and biases with
        # averaged nabla * learning rate eta
        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    # Stochastic Gradient Descent
    def sgd(self, training_data, epochs, batch_size, eta, validate_data=None):
        learning_curve = [0 for _ in range(epochs)]
        # get the length of the data
        number_of_training = len(training_data)
        # for each epochs
        for j in range(epochs):
            # shuffle training data and make it into batches
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size]
                            for k in range(0, number_of_training, batch_size)]
            # for each batches
            for batch in mini_batches:
                # update weights and biases
                self.update_feature(batch, eta)
            # if test exist
            if validate_data is not None:
                # show training result
                learning_curve[j] = round(self.success_rate(validate_data),2)
                print(f"Epoch {j}: {learning_curve[j]} %")
            # else show complete
            else:
                print(f"Epoch {j} completed!")
        return learning_curve

    def success_rate(self, test_data):
        # get the testing result index for each testing data
        # by applying feedforward to them
        testing_result = [(np.argmax(self.feed_forward(data)), truth)
                        for data, truth in test_data]
        # use the index to get the value from truth vector
        # sum them up and return the value 
        return sum(int(y[idx]) for idx,y in testing_result)/len(test_data)*100
    
    def classify(self, data):
        result = self.feed_forward(data)
        idx = np.argmax(result)
        return (idx, result[idx])

    def evaluate_testing_data(self, test_data):
        evaluation = []
        for img_idx, data in enumerate(test_data):
            x,y = data
            output_layer = self.feed_forward(x)
            cost = np.sum(np.square(output_layer - y))
            predict = np.argmax(output_layer)
            confidence = output_layer[predict][0]
            result = "Correct" if y[predict] == 1 else "Wrong"
            evaluation.append((cost, img_idx, predict, confidence, result))
        return evaluation
        


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            print(f"Model saved at {path}")