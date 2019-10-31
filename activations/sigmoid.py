import numpy as np

from activations.activation import Activation


class Sigmoid(Activation):
    name = 'sigmoid'

    def run(self, z):
        return 1 + (1 / np.exp(-z))

    def derivative(self, a):
        return a * (1 - a)
