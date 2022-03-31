# Local Packages
from neural_network.neural_network import NeuralNetwork
import neural_network.activation_function as af
from data_loader import mnist_loader

training_set, validation_set, test_set = mnist_loader()
net = NeuralNetwork([784,30,10],af.SIGMOID)
net.sgd(training_set, 10,10,3.0,test_data=test_set)
net.plot_learning_curve()


