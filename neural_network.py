import numpy as np
import random


class NeuralNetwork(object):

    def __init__(self, layer_sizes):
        """Neural network consisting of 'self.num_layers' layers. Each layer has
        a specific number of neurons specified in 'layer_sizes', which defines
        the architecture of the NN.
        Weights initialized using Gaussian distribution with mean 0 & st. dev. 1
        over the square root of the number of weights connecting to the same neuron."""
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(size2, size1) / np.sqrt(size1)
                            for size1, size2 in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(size, 1) for size in layer_sizes[1:]]

    def feedforward(self, a):
        """Return the network's output if 'a' is the input."""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, batch_size, learn_rate, test_data=None):
        """Implements the method of stochastic gradient descent, training the network
        by passing over the training data multiple times ('epochs'), each time using
        subsets of data of size 'batch_size'."""
        training_data = list(training_data) # mogoče lahko dam ven in dam v loadmnistdata
        n = len(training_data)
        if test_data:
            test_data = list(test_data) # isto
            n_test = len(test_data)

        for i in range(epochs):
            # create random batches for this epoch
            random.shuffle(training_data)
            batches = [training_data[j:j+batch_size] for j in range(0, n, batch_size)]
            # update batch
            for batch in batches:
                self.update_batch(batch, learn_rate)
            # evaluate learning progress
            if test_data:
                print("Epoch {} : {} / {}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(i))

    def backpropagation(self, x, y):
        """Backpropagation algorithm."""
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sig_deriv = sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sig_deriv
            grad_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
            grad_b[-layer] = delta

        return grad_w, grad_b

    def update_batch(self, batch, learn_rate):
        """Update the weights & biases of the network according to gradient descent of
        a single batch using backpropagation."""
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        batch_size = len(batch)

        for x, y in batch:
            delta_grad_w, delta_grad_b = self.backpropagation(x, y)
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]
            grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]

        self.weights = [w - (learn_rate/batch_size) * gw
                            for w, gw in zip(self.weights, grad_w)
                       ]
        self.biases = [b - (learn_rate/batch_size) * gb
                            for b, gb in zip(self.biases, grad_b)
                      ]

    def evaluate(self, test_data):
        """Return the number of correctly classified test inputs."""
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        """Return vector of partial derivatives of quadratic cost function (f(a) = 1/2 (a-y)^2)
         in respect to output activations."""
        return output_activations - y


# ------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------

def sigmoid(z):
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Return derivative of sigmoid function."""
    return sigmoid(z) * (1-sigmoid(z))
