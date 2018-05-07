import numpy
import random
import math

class ANN:
    # Constructor
    def __init__(self, input_nodes, output_nodes, hidden_nodes, beta, etta):
        # Number of nodes on each layer
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes

        # Sigma function constant
        self.beta = beta

        # Learning rate
        self.etta = etta

        # Initialize arrays of weight
        self.hidden_layer_weights = numpy.zeros((self.hidden_nodes, self.input_nodes))
        self.output_layer_weights = numpy.zeros((self.output_nodes, self.hidden_nodes))

        # Get random weights
        self.fill_weights()

    # Fill weighs arrays, with random values
    def fill_weights(self):
        # Use seed, to get static random values
        random.seed(322)

        for i in range(self.hidden_nodes):
            for j in range(self.input_nodes):
                self.hidden_layer_weights[i, j] = random.uniform(-1, 1)

        for i in range(self.output_nodes):
            for j in range(self.hidden_nodes):
                self.output_layer_weights[i, j] = random.uniform(-1, 1)


    # Activation function
    def sigma(self, x):
        return 1 / (1 - math.exp(-self.beta * x))

    # Get output of NN, with given input
    def predict(self, in_values):
        pass

    # Fitness function
    def fit(self, in_values, expected_values):
        pass



a = ANN(2, 2, 2, 2, 2)
print(a.hidden_layer_weights)