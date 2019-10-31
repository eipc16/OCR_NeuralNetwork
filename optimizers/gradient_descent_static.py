import numpy as np

from optimizers.optimizer import Optimizer


class StaticGradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self._learning_rate = learning_rate

    def update(self, activation, error, cost):
        return self._update(activation, self._delta, self._weights, self._biases, error, cost)

    def _update(self, activation, delta, weights, biases, error, cost):
        weights += self._learning_rate * activation.T @ delta
        biases += self._learning_rate * np.sum(delta, axis=0, keepdims=True)
        return weights, biases

    def copy(self):
        return StaticGradientDescent(self._learning_rate)
