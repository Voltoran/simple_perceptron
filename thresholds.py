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
        random.seed(1048)
        self.hidden_layer_weights = numpy.asarray([[1, -1], [-1, 1], [1, 1]])

        for i in range(self.output_nodes):
            for j in range(self.hidden_nodes):
                self.output_layer_weights[i, j] = 1

    # Get output of NN, with given input
    # Return output of each neuron layer
    def predict(self, in_values):
        hidden_layer_output = numpy.zeros(self.hidden_nodes)
        finish_layer_output = numpy.zeros(self.output_nodes)

        # Get hidden layer output
        for i in range(self.hidden_nodes):
            temp = 0.0
            for j in range(self.input_nodes):
                temp += self.hidden_layer_weights[i, j] * in_values[j]
            temp -= 0.5
            if temp > 0.0:
                hidden_layer_output[i] = 1
            else:
                hidden_layer_output[i] = 0

        # Get output layer output
        for i in range(self.output_nodes):
            temp = 0.0
            for j in range(self.hidden_nodes):
                temp += self.output_layer_weights[i, j] * hidden_layer_output[j]
            temp -= 0.5
            if temp > 0.0:
                finish_layer_output[i] = 1
            else:
                finish_layer_output[i] = 0

        return hidden_layer_output, finish_layer_output

    # Fitness function
    def fit(self, in_values, expected_values):
        output_error = numpy.zeros(self.output_nodes)

        hidden_layer_output, finish_layer_output = self.predict(in_values)

        for i in range(self.output_nodes):
            output_error[i] = expected_values[i] - finish_layer_output[i]

        for i in range(self.output_nodes):
            delta = output_error[i]
            for j in range(self.hidden_nodes):
                self.output_layer_weights[i, j] += delta * self.etta * hidden_layer_output[j]

        return output_error


new_web = ANN(2, 1, 3, 8, 0.1)
print("Init out w:\n", new_web.output_layer_weights)
print("Init hidden w:\n", new_web.hidden_layer_weights)
print()
N = 50
for i in range(N):
        k = random.randint(1, N)

        if k % 4 == 0:
            x = numpy.array([1, 0])
            y = numpy.array([1])
        elif k % 4 == 1:
            x = numpy.array([0, 1])
            y = numpy.array([1])
        elif k % 4 == 2:
            x = numpy.array([0, 0])
            y = numpy.array([0])
        else:
            x = numpy.array([1, 1])
            y = numpy.array([0])

        error = new_web.fit(x, y)
        # if i % 1 == 0:
        #     print("Error: ", error[0], "( iteration ", i, ")")
print("Out out w:\n", new_web.output_layer_weights)
print("Out hidden w:\n", new_web.hidden_layer_weights)
print()

print("[1,0] -> 1: ", new_web.predict(numpy.array([1, 0]))[1])
print("[0,1] -> 1: ", new_web.predict(numpy.array([0, 1]))[1])
print("[1,1] -> 0: ", new_web.predict(numpy.array([1, 1]))[1])
print("[0,0] -> 0: ", new_web.predict(numpy.array([0, 0]))[1])

