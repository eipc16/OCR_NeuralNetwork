import numpy as np

from activations.activation import Activation


class Softmax(Activation):
    name = 'softmax-stable'

    def run(self, z):
        numerator = np.exp(z - np.max(z))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator

    def derivative(self, a):
        return 1
