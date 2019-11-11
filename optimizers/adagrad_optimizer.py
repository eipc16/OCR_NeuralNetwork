from collections import defaultdict

import numpy as np

from optimizers.optimizer import Optimizer


class AdaGradOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._grad = defaultdict(int)

    def calc_gradients(self, identifier, gradients):
        self._grad[identifier] = self._grad[identifier] + gradients ** 2
        return self._learning_rate / np.sqrt(self._grad[identifier] + self._epsilon) * gradients

    def get_parameters(self):
        return f'Learning rate - {self._learning_rate} | Epsilon - {self._epsilon}'

    def get_name(self):
        return 'AdaGrad Optimizer'
