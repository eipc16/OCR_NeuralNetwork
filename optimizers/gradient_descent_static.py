import numpy as np

from optimizers.optimizer import Optimizer


class StaticGradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self._learning_rate = learning_rate

    def calc_gradients(self, identifier, gradients):
        return self._learning_rate * gradients

    def get_parameters(self):
        return f'Learning rate - {self._learning_rate}'

    def get_name(self):
        return 'Static Gradient Descent'
