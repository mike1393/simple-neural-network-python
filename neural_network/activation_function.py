# Third Party Packages
import numpy as np
# Build-in Packages
import random
# Local Packages

# sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# the derivative of sigmoid function
def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    return max(0, z)

def d_relu(z):
    return 0 if z <=0 else 1
