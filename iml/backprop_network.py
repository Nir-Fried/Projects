import random
import numpy as np
from scipy.special import softmax
import math
import collections
import matplotlib.pyplot as plt 

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x)

                        for x, y in zip(sizes[:-1], sizes[1:])]



    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data,section):

        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  """

        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))
        train_acc = []
        train_loss = []
        test_acc = []
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            print ("Epoch {0} test accuracy: {1}".format(j, self.one_label_accuracy(test_data)))
            train_acc.append(self.one_hot_accuracy(training_data))
            train_loss.append(self.loss(training_data))
            test_acc.append(self.one_label_accuracy(test_data))

        if section == 2:
            return train_acc, train_loss, test_acc

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        stochastic gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """The function receives as input a 784 dimensional 
        vector x and a one-hot vector y.
        The function should return a tuple of two lists (db, dw) 
        as described in the assignment pdf. """

        v, z = self.forward_pass(x)
        partialb, partialW = self.backwards_pass(y, v, z)

        return partialb, partialW

    def forward_pass(self, x): 

        y = x.copy()
        v = []
        z = [y]
        L = self.num_layers

        for i in range(L-2):
            v.append(np.dot(self.weights[i], y) + self.biases[i])
            y = relu(v[-1])
            z.append(y.copy())
        
        v.append(np.dot(self.weights[L-2], y) + self.biases[L-2])
        y = softmax(v[-1])
        z.append(y.copy())
        
        return v, z

    def backwards_pass(self, y, v, z):
        
        L = self.num_layers
        delta = collections.deque()
        partialW = collections.deque()
        partialb = collections.deque()

        #insert layer L-1
        delta.appendleft(z[-1]-y)
        partialW.appendleft(np.dot(delta[0], z[-2].T))
        partialb.appendleft(delta[0].copy())

        #insert layer L-2
        delta.appendleft(np.dot(self.weights[L-2].T, delta[0]))
        partialb.appendleft(np.multiply(delta[0], relu_derivative(v[-2])))
        partialW.appendleft(np.dot(partialb[0], z[-3].T))

        for i in range(L-3,0,-1):
            delta.appendleft(np.dot(self.weights[i].T, np.multiply(delta[0], relu_derivative(v[i+1]))))
            partialb.appendleft(np.multiply(delta[0], relu_derivative(v[i])))
            partialW.appendleft(np.dot(partialb[0], z[i-1].T))

        return list(partialb), list(partialW)
    
    def one_label_accuracy(self, data):

        """Return accuracy of network on data with numeric labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), y)

         for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results)/float(len(data))



    def one_hot_accuracy(self,data):

        """Return accuracy of network on data with one-hot labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), np.argmax(y))

                          for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))





    def network_output_before_softmax(self, x):

        """Return the output of the network before softmax if ``x`` is input."""

        layer = 0

        for b, w in zip(self.biases, self.weights):

            if layer == len(self.weights) - 1:

                x = np.dot(w, x) + b

            else:

                x = relu(np.dot(w, x)+b)

            layer += 1

        return x



    def loss(self, data):

        """Return the CE loss of the network on the data"""

        loss_list = []

        for (x, y) in data:

            net_output_before_softmax = self.network_output_before_softmax(x)

            net_output_after_softmax = self.output_softmax(net_output_before_softmax)

            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])

        return sum(loss_list) / float(len(data))

    def output_softmax(self, output_activations):

        """Return output after softmax given output before softmax"""

        return softmax(output_activations)


def relu(z):
    return np.maximum(0, z)

def relu_derivative(z): #return the derivative of relu function
    return (z > 0) * 1
    

