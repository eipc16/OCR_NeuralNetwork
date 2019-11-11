from collections import defaultdict

from optimizers.optimizer import Optimizer


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.5):
        super().__init__()
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._velocity_array = defaultdict(int)

    def calc_gradients(self, identifier, gradients):
        new_gradients = self._momentum * self._velocity_array[identifier]
        new_gradients += self._learning_rate * gradients
        self._velocity_array[identifier] = new_gradients
        return new_gradients

    def get_parameters(self):
        return f'Learning rate - {self._learning_rate} | Momentum - {self._momentum}'

    def get_name(self):
        return 'Momentum Optimizer'
