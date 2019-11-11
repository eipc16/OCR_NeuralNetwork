from collections import defaultdict

import numpy as np

from optimizers.optimizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta_01=0.0, beta_02=0.999, epsilon=1e-8):
        self._learning_rate = learning_rate
        self._beta_01 = beta_01
        self._beta_02 = beta_02
        self._epsilon = epsilon
        self._m, self._v, self._t = self._init_parameters()

    def calc_gradients(self, identifier, gradients):
        self._m[identifier] = self._beta_01 * self._m[identifier] + (1 - self._beta_01) * gradients
        self._v[identifier] = self._beta_02 * self._v[identifier] + (1 - self._beta_02) * gradients ** 2

        self._t[identifier] += 1

        m_est = self._m[identifier] / (1 - self._beta_01 ** self._t[identifier])
        v_est = self._v[identifier] / (1 - self._beta_02 ** self._t[identifier])

        return self._learning_rate / (np.sqrt(v_est) + self._epsilon) * m_est

    def get_parameters(self):
        return  f'Learning rate - {self._learning_rate} | ' \
                f'Epsilon - {self._epsilon} | ' \
                f'Beta 1 - {self._beta_01} | Beta_2 - {self._beta_02}'

    def get_name(self):
        return 'Adam Optimizer'

    @staticmethod
    def _init_parameters():
        return defaultdict(int), defaultdict(int), defaultdict(int)
