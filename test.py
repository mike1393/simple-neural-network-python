from neural_network.neural_network import NeuralNetwork
from data_loader import mnist_loader

training_set, validation_set, test_set = mnist_loader()
net = NeuralNetwork([784,30,10])
net.sgd(training_set, 10,10,3.0,test_data=test_set)


