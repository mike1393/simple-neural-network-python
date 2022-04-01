# simple-neural-network-python
This is a python project for making sure that I REALLY understand of how neural networks work.<br>
I built this project by following the tutorials from [3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks) and [Yujian Tang](https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/).<br> Be sure to check them out if you are interested in this work.<br>
#### :books: **Data**<br>
The dataset used in this repo is the famous :tada:[MNIST dataset](http://yann.lecun.com/exdb/mnist/):tada:<br>
#### :open_file_folder: **File Structure**<br>
The file structure of the repo:
```
.
├─data
│   └─mnist.pkl.gz
│
├─neural_network
│     ├─activation_function.py
│     ├─neural_network.py
│     ├─data_loader.py
│     └─__init__.py
│ 
├─result
│   ├─hidden30_sigmoid_best_6.png
│   ├─hidden30_sigmoid_worst_6.png
│   └─model_pkl
│ 
├─find_best_and_worst.py
├─save_model.py
├─Pipfile
├─Pipfile.lock
├─README.md
```
#### :computer: **Virtual Environment**<br>
The virtual environment was made with [pipenv](https://pipenv.pypa.io/en/latest/)<br>
The environment setting is listed in [./Pipfile](https://github.com/mike1393/simple-neural-network-python/blob/main/Pipfile) and [./Pipfile.lock](https://github.com/mike1393/simple-neural-network-python/blob/main/Pipfile.lock).<br>
Follow the guide [here](https://docs.python-guide.org/dev/virtualenvs/) if you want to learn pipenv<br>
#### :boom: **Neural Network**<br>
This is a simple implementation of a fully connected neural network.<br>
The detailed implementation was in [./neural_network/neural_network.py](https://github.com/mike1393/simple-neural-network-python/blob/main/neural_network/neural_network.py).<br>
The user specifies two inputs to create the neural network object.<br>
The first one is the structure of your network using a list of integers, where each number represents the number of neurons for each layer.<br>
The second one is the type of activation function. The implementation can be foun at [./neural_network/activation_function.py](https://github.com/mike1393/simple-neural-network-python/blob/main/neural_network/activation_function.py)<br>
  * For example, if we want to craete a network with 2 hidden layers, each hidden layer consists of 16 neurona.<br>
And all layers use sigmoid function as activation function.<br>
  ```python
  from neural_network.neural_network import NeuralNetwork
  import neural_network.activation_function as af
  net = NeuralNetwork([784,16,16,10], af.SIGMOID)
  ```
Check out [save_model.py](https://github.com/mike1393/simple-neural-network-python/blob/main/save_model.py) or [find_best_and_worst.py](https://github.com/mike1393/simple-neural-network-python/blob/main/find_best_and_worst.py) to see more examples in how to use the pacakge.<br>
#### :chart_with_upwards_trend: **Results**
  * 6 images that performs the **best**<br>
  ![Best 6](https://raw.githubusercontent.com/mike1393/simple-neural-network-python/main/result/hidden30_sigmoid_best_6.png)
  * 6 images that performs the **worst**<br>
  ![Worst 6](https://raw.githubusercontent.com/mike1393/simple-neural-network-python/main/result/hidden30_sigmoid_worst_6.png)
