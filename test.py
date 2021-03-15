import load_mnist_data
import neural_network

training_data, validation_data, test_data = load_mnist_data.load_data()

net = neural_network.NeuralNetwork([784, 40, 20, 10])
net.stochastic_gradient_descent(training_data, 40, 10, 2.0, test_data=test_data)

