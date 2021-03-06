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
        random.seed(10000)

        for i in range(self.hidden_nodes):
            for j in range(self.input_nodes):
                self.hidden_layer_weights[i, j] = random.uniform(-1, 1)

        for i in range(self.output_nodes):
            for j in range(self.hidden_nodes):
                self.output_layer_weights[i, j] = random.uniform(-1, 1)

    # Activation function
    def sigma(self, x):
        return 1 / (1 + math.exp(-self.beta * x))

    # Activation function derivative
    def desigma(self, x):
        # x = self.sigma(x)
        return self.beta * x * (1 - x)

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
            hidden_layer_output[i] = self.sigma(temp)

        # Get output layer output
        for i in range(self.output_nodes):
            temp = 0.0
            for j in range(self.hidden_nodes):
                temp += self.output_layer_weights[i, j] * hidden_layer_output[j]
            finish_layer_output[i] = self.sigma(temp)

        return hidden_layer_output, finish_layer_output

    # Fitness function
    def fit(self, in_values, expected_values):
        output_error = numpy.zeros(self.output_nodes)
        delta_output_array = numpy.zeros(self.output_nodes)

        hidden_layer_output, finish_layer_output = self.predict(in_values)


        # Output error calculation
        for i in range(self.output_nodes):
            output_error[i] = expected_values[i] - finish_layer_output[i]
        
        # Deltas for output layer
        for i in range(self.output_nodes):
            delta_output_array[i] = self.desigma(finish_layer_output[i]) * output_error[i]

        # Calculate deltas for hiddent layer, and rescale hidden layer weights
        for i in range(self.hidden_nodes):
            summ = 0
            for k in range(self.output_nodes):
                summ += delta_output_array[k] * self.output_layer_weights[k, i]

            # Delta for hiddent layer
            delta_hidden = self.desigma(hidden_layer_output[i]) * summ

            # Rescale hidden weights
            for j in range(self.input_nodes):
                self.hidden_layer_weights[i, j] += delta_hidden * self.etta * in_values[j]

        # Rescale output weights
        for i in range(self.output_nodes):
            for j in range(self.hidden_nodes):
                self.output_layer_weights[i, j] += delta_output_array[i] * self.etta * hidden_layer_output[j]

        return output_error


new_web = ANN(2, 1, 3, 5, 0.1)
N = 10001
print("Init out w:\n", new_web.output_layer_weights)
print("Init hidden w:\n", new_web.hidden_layer_weights)
print()

for i in range(N):
        k = i
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
        # if i % 1000 == 0:
            # print("Error: ", error[0], "( iteration ", i, ")")

print("Out out w:\n", new_web.output_layer_weights)
print("Out hidden w:\n", new_web.hidden_layer_weights)
print()


print("[1,0] -> 1: ", new_web.predict(numpy.array([1, 0]))[1])
print("[0,1] -> 1: ", new_web.predict(numpy.array([0, 1]))[1])

print("[1,1] -> 0: ", new_web.predict(numpy.array([1, 1]))[1])
print("[0,0] -> 0: ", new_web.predict(numpy.array([0, 0]))[1])

