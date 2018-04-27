import numpy as np
import math


class Web:

    def __init__(self, x_count, y_count, hls, etta):
        self.x_count = x_count
        self.y_count = y_count
        self.hide_layer_size = hls
        self.output_layer_size = y_count
        self.etta = etta

        self.w1 = np.random.rand(self.x_count)
        self.w2 = np.random.rand(self.hide_layer_size)

        self.U = np.zeros(self.hide_layer_size)
        self.A = np.zeros(self.output_layer_size)

        self.E2 = np.zeros(self.hide_layer_size)
        self.E = np.zeros(self.output_layer_size)

        self.O1 = np.zeros(self.hide_layer_size)
        self.O2 = np.zeros(self.output_layer_size)

    def activation_function(self, x, beta):
        return 1 / (1 + math.exp(-beta * x))

    def fit(self, x_train, y_train):
        for h in range(self.hide_layer_size):
            self.O1[h] = np.sum(np.multiply(self.w1, x_train))
            self.U[h] = self.activation_function(self.O1[h], 1)

        for m in range(self.output_layer_size):
            self.O2[m] = np.sum(np.multiply(self.w2, self.U))
            self.A[m] = self.activation_function(self.O2[m], 1)

        self.E = np.subtract(self.A, y_train)

        print("U", self.U)
        print("A", self.A)
        print("O1", self.O1)
        print("O2", self.O2)
        print("E", self.E)


        E_sum = np.sum(self.E)
        print("E sum", self.E)

        O1_sigm = np.zeros(self.hide_layer_size)
        O2_sigm = np.zeros(self.output_layer_size)

        for k in range(self.output_layer_size):
            sigm = self.activation_function(self.O2[k], 1)
            O2_sigm[k] = sigm * (1 - sigm)

        for k in range(self.hide_layer_size):
            sigm = self.activation_function(self.O1[k], 1)
            O1_sigm[k] = sigm * (1 - sigm)

        print("O1 sigm", O1_sigm)
        print("O2 sigm", O2_sigm)

        self.E2 = np.multiply(E_sum, np.multiply(O2_sigm, self.w2))
        print("E2", self.E2 )

        a1 = np.multiply(self.E2, self.etta)
        a2 = np.multiply(O1_sigm, x_train)

        self.w1 = np.subtract(self.w1, np.multiply(a1, a2))


        b1 = np.multiply(self.E, self.etta)
        b2 = np.multiply(O2_sigm, self.U)
        self.w2 = np.subtract(self.w2, np.multiply(b1, b2))

    def predict(self, x_data):
        for h in range(self.hide_layer_size):
            self.O1[h] = np.sum(np.multiply(self.w1, x_data))
            self.U[h] = self.activation_function(self.O1[h], 20)

        for m in range(self.output_layer_size):
            self.O2[m] = np.sum(np.multiply(self.w2, self.U))
            self.A[m] = self.activation_function(self.O2[m], 20)


        return self.A


new_web = Web(2, 1, 3, 1)
# print("\nW1", new_web.w1)
# print("W2", new_web.w2)

# x_train = np.array([0, 0])
# y_train = np.array([0])
# new_web.fit(x_train, y_train)
# print("\nW1", new_web.w1)
# print("W2", new_web.w2)

# x_train = np.array([0, 1])
# y_train = np.array([1])
# new_web.fit(x_train, y_train)
#
# x_train = np.array([1, 0])
# y_train = np.array([1])
# new_web.fit(x_train, y_train)
# print("\nW1", new_web.w1)
# print("W2", new_web.w2)

# x_train = np.array([1, 1])
# y_train = np.array([0])
# new_web.fit(x_train, y_train)

print(new_web.predict(np.array([0, 1])))


# a = np.array([1, 2, 3, 4])
# b = np.array([2, 2, 2, 2])
#
# print(np.multiply(a, b))