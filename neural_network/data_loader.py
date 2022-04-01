# Third Party Packages
import numpy as np
# Build-in Packages
import pickle
import gzip
import os
import neural_network

PACKAGE_DIR = os.path.dirname(neural_network.__file__)
DATA_DIR = os.path.join(PACKAGE_DIR, "..\data")
MNIST_DIR = os.path.join(DATA_DIR, "mnist.pkl.gz")

#encode the result to a list with all elements equals to zero
#except the result index is 1.0
def result_encoder(y):
    encoder = np.zeros((10,1))
    encoder[y] = 1.0
    return encoder

def mnist_loader():
    mnist_data = []
    with gzip.open(MNIST_DIR) as f:
        data_set = pickle.load(f, encoding='latin1')
        for i in range(len(data_set)):
            img_set, result_set = data_set[i]
            # transpose each img data into a vector
            img_list = [np.reshape(img,(784,1)) for img in img_set]
            # encode each result into a vector
            result_list = [result_encoder(result) for result in result_set]
            # Create a zip from img vector and result vector
            mnist_data.append(list(zip(img_list, result_list)))
    return mnist_data