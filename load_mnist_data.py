import gzip
import pickle

import numpy as np


# Partition MNIST data (70k entries, containing pairs of 784-dimensional numpy arrays representing images
# and their classification) into a training set (50k entries), validation set (10k entries) and test set
# (10k entries) using pickle. Also reformat data for use in neural network.
def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

    train_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    train_results = [num2vector(y) for y in training_data[1]]
    training_data = zip(train_inputs, train_results)

    validat_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validat_results = validation_data[1]
    validation_data = zip(validat_inputs, validat_results)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_results = test_data[1]
    test_data = zip(test_inputs, test_results)

    return training_data, validation_data, test_data


# Take a number from 0 to 9 and represent it as a vector containing '1' at the numbers corresponding index
def num2vector(num):
    vector = np.zeros((10, 1))
    vector[num] = 1
    return vector
