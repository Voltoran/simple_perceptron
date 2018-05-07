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

        self.w1 = np.zeros((self.hide_layer_size, self.x_count))
        self.w2 = np.zeros((self.output_layer_size, self.hide_layer_size))


        for h in range(self.hide_layer_size):
            for n in range(self.x_count):
                self.w1[h, n] = random.uniform(-1, 1)

        for m in range(self.output_layer_size):
            for h in range(self.hide_layer_size):
                self.w2[m, h] = random.uniform(-1, 1)


        self.U = np.zeros(self.hide_layer_size)
        self.A = np.zeros(self.output_layer_size)

        self.E2 = np.zeros(self.hide_layer_size)
        self.E = np.zeros(self.output_layer_size)

        self.O1 = np.zeros(self.hide_layer_size)
        self.O2 = np.zeros(self.output_layer_size)

    def activation_function(self, x, beta=1):
        return 1 / (1 + math.exp(-beta * x))


    def fit(self, x_train, y_train):
        # for h in range(self.hide_layer_size):
        #     self.O1[h] = np.sum(np.multiply(self.w1[h], x_train))
        #     self.U[h] = self.activation_function(self.O1[h])

        for h in range(self.hide_layer_size):
            for n in range(self.x_count):
                self.O1[h] += self.w1[h, n] * x_train[n]
            self.U[h] = self.activation_function(self.O1[h])

        #
        # for m in range(self.output_layer_size):
        #     self.O2[m] = np.sum(np.multiply(self.w2[:, m], self.U))
        #     self.A[m] = self.activation_function(self.O2[m])

        for m in range(self.output_layer_size):
            for h in range(self.hide_layer_size):
                self.O2[m] += self.w2[m, h] * self.U[h]
            self.A[m] = self.activation_function(self.O2[m])

        self.E = np.subtract(self.A, y_train)
        print("E1 ", self.E)

        O1_sigm = np.zeros(self.hide_layer_size)
        O2_sigm = np.zeros(self.output_layer_size)

        for k in range(self.output_layer_size):
            sigm = self.activation_function(self.O2[k])
            O2_sigm[k] = 1 * sigm * (1 - sigm)

        for k in range(self.hide_layer_size):
            sigm = self.activation_function(self.O1[k])
            O1_sigm[k] = 1 * sigm * (1 - sigm)

        # for h in range(self.hide_layer_size):
        #     self.E2[h] = np.sum(np.multiply(self.E, np.multiply(O2_sigm, self.w2[h])))

        for m in range(self.output_layer_size):
            for h in range(self.hide_layer_size):
                self.E2[h] += self.E[m] * O2_sigm[m] * self.w2[m, h]
        print("E2 ", self.E2)
        # W1
        # for n in range(self.x_count):
        #     a1 = np.multiply(self.E2, self.etta)
        #     a2 = np.multiply(O1_sigm, x_train[n])
        #     self.w1[:, n] = np.subtract(self.w1[:, n], np.multiply(a1, a2))
        for h in range(self.hide_layer_size):
            for n in range(self.x_count):
                print("delta 1 ", self.etta * self.E2[h] * O1_sigm[h] * x_train[n])
                self.w1[h, n] -= self.etta * self.E2[h] * O1_sigm[h] * x_train[n]


        # W2
        # for h in range(self.hide_layer_size):
        #     b1 = np.multiply(self.E, self.etta)
        #     b2 = np.multiply(O2_sigm, self.U[h])
        #     self.w2[h] = np.subtract(self.w2[h], np.multiply(b1, b2))

        for m in range(self.output_layer_size):
            for h in range(self.hide_layer_size):
                print("delta 2", self.etta * self.E[m] * O2_sigm[m] * self.U[h])
                self.w2[m, h] -= self.etta * self.E[m] * O2_sigm[m] * self.U[h]

    def predict(self, x_data):
        # for h in range(self.hide_layer_size):
        #     self.O1[h] = np.sum(np.multiply(self.w1[h], x_train))
        #     self.U[h] = self.activation_function(self.O1[h])

        for h in range(self.hide_layer_size):
            for n in range(self.x_count):
                self.O1[h] += self.w1[h, n] * x_data[n]
            self.U[h] = self.activation_function(self.O1[h])

        #
        # for m in range(self.output_layer_size):
        #     self.O2[m] = np.sum(np.multiply(self.w2[:, m], self.U))
        #     self.A[m] = self.activation_function(self.O2[m])

        for m in range(self.output_layer_size):
            for h in range(self.hide_layer_size):
                self.O2[m] += self.w2[m, h] * self.U[h]
            self.A[m] = self.activation_function(self.O2[m])

        return self.A


new_web = Web(2, 1, 2, 0.01)

# N = 20
# print("w1 **********\n", new_web.w1)
# print("w2 **********\n", new_web.w2)
# print("")
# for i in range(N):
#         k = random.randint(1, N)
#
#         if k % 4 == 0:
#             x = np.array([0, 0])
#             y = np.array([0])
#         elif k % 4 == 1:
#             x = np.array([0, 1])
#             y = np.array([1])
#         elif k % 4 == 2:
#             x = np.array([1, 0])
#             y = np.array([1])
#         else:
#             x = np.array([1, 1])
#             y = np.array([0])
#
#         new_web.fit(x, y)
#
#         print("w1 **********\n", new_web.w1)
#         print("w2 **********\n", new_web.w2)
#         print("")
# print("\nEND")
# print("w1 **********\n", new_web.w1)
# print("w2 **********\n", new_web.w2)
#
# print("-> 1 ", new_web.predict(np.array([1, 0])))
# print("-> 1 ", new_web.predict(np.array([0, 1])))
#
# print("-> 0 ", new_web.predict(np.array([1, 1])))
# print("-> 0 ", new_web.predict(np.array([0, 0])))
