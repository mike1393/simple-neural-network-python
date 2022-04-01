# Third Party Packages
# Build-in Packages
import os
# Local Packages
from neural_network.neural_network import NeuralNetwork
import neural_network.activation_function as af
from neural_network.data_loader import mnist_loader

output_file_path = os.path.join(os.path.dirname(__file__),'result\model_pkl')
training_set, validation_set, test_set = mnist_loader()
net = NeuralNetwork([784,30,10],af.SIGMOID)
net.sgd(training_set, 10,10,3.0,validate_data=validation_set)
net.save(output_file_path)

