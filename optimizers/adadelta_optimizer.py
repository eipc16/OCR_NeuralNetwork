from collections import defaultdict

import numpy as np

from optimizers.optimizer import Optimizer


class AdaDeltaOptimizer(Optimizer):
    def __init__(self, rho=0.95, epsilon=1e-8):
        self._rho = rho
        self._epsilon = epsilon
        self._error, self._error_delta = self._init_parameters()

    def calc_gradients(self, identifier, gradients):
        self._error[identifier] = self._rho * self._error[identifier] + (1 - self._rho) * gradients ** 2
        rms = np.sqrt(self._error_delta[identifier] + self._epsilon)
        gradients_delta = rms / np.sqrt(self._error[identifier] + self._epsilon) * gradients
        self._error_delta[identifier] = self._rho * self._error_delta[identifier] + (1 - self._rho) * gradients_delta ** 2
        return gradients_delta

    def get_parameters(self):
        return f'Rho - {self._rho} | Epsilon: {self._epsilon}'

    def get_name(self):
        return 'AdaDelta Optimizer'

    @staticmethod
    def _init_parameters():
        return defaultdict(int), defaultdict(int)
