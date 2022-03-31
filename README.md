# simple-neural-network-python
This is a python project for self-validation of understanding of how neural networks work.<br>
I finished this project by following the tutorials from [3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks) and [Yujian Tang](https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/).<br>
* Data<br>
The dataset used in this repo is the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/)<br>
* File Structure<br>
The file structure of the repo:
```
.
├─data
│   └─mnist.pkl.gz
│
├─neural_network
│     ├─activation_function.py
│     ├─neural_network.py
│     └─__init__.py
│ 
├─data_loader.py
├─Pipfile
├─Pipfile.lock
├─README.md
├─test.py
```
* Environment<br>
The virtual environment was made with [pipenv](https://pipenv.pypa.io/en/latest/)<br>
The environment setting is listed in ```./Pipfile``` and ```./Pipfile.lock```.<br>
Follow the guide [here](https://docs.python-guide.org/dev/virtualenvs/) if you want to learn pipenv<br>
* Neural Network<br>
This is a simple implementation of a fully connected neural network.<br>
The detailed implementation was in ```./neural_network/neural_network.py```.<br>
The user specifies two inputs to create the neural network object.<br>
The first one is the structure of your network using a list of integers, where each number represents the number of neurons for each layer.<br>
The second one is the type of activation function. The implementation can be foun at ```./neural_network/activation_function.py```<br>
  * For example, if we want to craete a network with 2 hidden layers, each hidden layer consists of 16 neurona.<br>
And all layers use sigmoid function as activation function.<br>
  ```python
  from neural_network.neural_network import NeuralNetwork
  import neural_network.activation_function as af
  net = NeuralNetwork([784,16,16,10], af.SIGMOID)
  ```

