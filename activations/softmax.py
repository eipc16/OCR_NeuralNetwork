import numpy as np

from activations.activation import Activation


class Softmax(Activation):
    def run(self, z):
        numerator = np.exp(z - np.max(z))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / (denominator + 1e-8)

    def derivative(self, a):
        return 1

    def get_name(self):
        return 'softmax-stable'
