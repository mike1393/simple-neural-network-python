# Third Party Packages
import numpy as np
# Build-in Packages
SIGMOID = "SIGMOID"
RELU = "RELU"
TANH="TANH"

# sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# the derivative of sigmoid function
def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    z = np.exp(z)
    return np.where(z>0,z,1.0*(z-1))

def d_relu(z):
    z = np.exp(z)
    return np.where(z>0,1.,1.0*z)

def htan(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def d_htan(z):
    return 1-np.square(htan(z))

# activation type
activate_method={
SIGMOID : (sigmoid, d_sigmoid),
RELU : (relu, d_relu),
TANH:(htan,d_htan)
}