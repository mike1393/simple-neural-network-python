# Third Party Packages
import matplotlib.pyplot as plt
import numpy as np
# Build-in Packages
import os
import pickle
# Local Packages
from neural_network.neural_network import NeuralNetwork, model_loader
import neural_network.activation_function as af
from data_loader import mnist_loader
output_file_path = os.path.join(os.path.dirname(__file__),'model_pkl')
training_set, validation_set, test_set = mnist_loader()
# net = NeuralNetwork([784,30,10],af.SIGMOID)
# net.sgd(training_set, 10,10,3.0)
# net.save(output_file_path)
net2 = model_loader(output_file_path)
for i in range(5):
    x, y = test_set[i]
    result, confidence = net2.classify(x)
    img = np.reshape(x,(28,28))
    fig = plt.figure
    plt.title(f"Result: {result}, Confidence: {confidence*100}%")
    plt.imshow(img, cmap='gray')
    plt.show()


