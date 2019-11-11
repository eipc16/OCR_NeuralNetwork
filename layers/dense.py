from activations.sigmoid import Sigmoid
from initializers.xavier_initializer import XavierInitializer
from initializers.zero_initializer import ZeroInitializer
from layers.layer import Layer
from optimizers.optimizer import Optimizer

import numpy as np


class Dense(Layer):
    def __init__(self, layer_size, weight_initializer=XavierInitializer(), activation_func=Sigmoid(),
                 bias_initializer=ZeroInitializer(), layer_name='dense'):
        super().__init__(layer_size, layer_name)
        self._weight_initializer = weight_initializer
        self._activation_func = activation_func
        self._bias_initializer = bias_initializer
        self._weights = None
        self._biases = None
        self._activations = None
        self._delta = None
        self._optimizer = None

    def __call__(self, previous_layer_size, optimizer):
        self._weights = self._weight_initializer((previous_layer_size, self._layer_size))
        self._biases = self._bias_initializer((1, self._layer_size))
        self._optimizer = optimizer

    def get_error(self):
        return self._delta @ self._weights.T

    def update_delta(self, error):
        self._delta = error * self._activation_func.derivative(self._activations)

    def feed(self, x):
        self._activations = self._activation_func.run(x @ self._weights + self._biases)
        return self._activations

    def update(self, x, error, cost):
        self._weights += self._optimizer.calc_gradients(id(self._weights), x.T @ self._delta)
        self._biases += self._optimizer.calc_gradients(id(self._biases),
                                                       np.sum(self._delta, axis=0, keepdims=True))
        return self._activations
