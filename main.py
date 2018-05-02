import numpy as np
import math
import random


class Web:

    def __init__(self, x_count, y_count, hls, etta):
        self.x_count = x_count
        self.y_count = y_count
        self.hide_layer_size = hls
        self.output_layer_size = y_count
        self.etta = etta

        self.w1 = np.random.rand(self.hide_layer_size, self.x_count)
        self.w2 = np.random.rand(self.hide_layer_size, self.output_layer_size)

        self.U = np.zeros(self.hide_layer_size)
        self.A = np.zeros(self.output_layer_size)

        self.E2 = np.zeros(self.hide_layer_size)
        self.E = np.zeros(self.output_layer_size)

        self.O1 = np.zeros(self.hide_layer_size)
        self.O2 = np.zeros(self.output_layer_size)

    def activation_function(self, x, beta=1):
        return 1 / (1 + math.exp(-beta * x))

    def fit(self, x_train, y_train):
        for h in range(self.hide_layer_size):
            self.O1[h] = np.sum(np.multiply(self.w1[h], x_train))
            self.U[h] = self.activation_function(self.O1[h])

        for m in range(self.output_layer_size):
            self.O2[m] = np.sum(np.multiply(self.w2[m], self.U))
            self.A[m] = self.activation_function(self.O2[m])

        self.E = np.subtract(self.A, y_train)

        O1_sigm = np.zeros(self.hide_layer_size)
        O2_sigm = np.zeros(self.output_layer_size)

        for k in range(self.output_layer_size):
            sigm = self.activation_function(self.O2[k])
            O2_sigm[k] = sigm * (1 - sigm)

        for k in range(self.hide_layer_size):
            sigm = self.activation_function(self.O1[k])
            O1_sigm[k] = sigm * (1 - sigm)

        for h in range(self.hide_layer_size):
            self.E2[h] = np.sum(np.multiply(self.E, np.multiply(O2_sigm, self.w2[h])))

        for h in range(self.hide_layer_size):
            a1 = np.multiply(self.E2[h], self.etta)
            a2 = np.multiply(O1_sigm[h], x_train)
            self.w1[h] = np.subtract(self.w1[h], np.multiply(a1, a2))

        for m in range(self.output_layer_size):
            b1 = np.multiply(self.E[m], self.etta)
            b2 = np.multiply(O2_sigm[m], self.U)
            self.w2[m] = np.subtract(self.w2[m], np.multiply(b1, b2))

    def predict(self, x_data):
        for h in range(self.hide_layer_size):
            self.O1[h] = np.sum(np.multiply(self.w1[h], x_data))
            self.U[h] = self.activation_function(self.O1[h])

        for m in range(self.output_layer_size):
            self.O2[m] = np.sum(np.multiply(self.w2[m], self.U))
            self.A[m] = self.activation_function(self.O2[m])

        return self.A


new_web = Web(2, 1, 5, 1)


test_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]

y_test_list = [np.array([0]), np.array([1]), np.array([1]), np.array([0])]

N = 1000
for i in range(1000):
        if i % 4 == 0:
            x = np.array([0, 0])
            y = np.array([0])
            new_web.fit(x, y)
        elif i % 4 == 1:
            x = np.array([0, 1])
            y = np.array([1])
            new_web.fit(x, y)
        elif i % 4 == 2:
            x = np.array([1, 0])
            y = np.array([1])
            new_web.fit(x, y)
        else:
            x = np.array([1, 1])
            y = np.array([0])
            new_web.fit(x, y)

print("-> ", new_web.predict(np.array([1, 0])))