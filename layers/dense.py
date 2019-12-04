from activations.sigmoid import Sigmoid
from initializers.xavier_initializer import XavierInitializer
from initializers.zero_initializer import ZeroInitializer
from layers.layer import Layer

import numpy as np


class Dense(Layer):
    def __init__(self, layer_size, weight_initializer=XavierInitializer(), activation_func=Sigmoid(),
                 bias_initializer=ZeroInitializer(), layer_name='dense'):
        super().__init__(layer_name)
        self._weight_initializer = weight_initializer
        self._activation_func = activation_func
        self._bias_initializer = bias_initializer
        self._layer_size = layer_size
        self._z = None

    def __call__(self, previous_layer_shape, optimizer, calc_error=True):
        self._weights = self._weight_initializer((previous_layer_shape, self._layer_size))
        self._biases = self._bias_initializer((1, self._layer_size))
        self._optimizer = optimizer
        self._calc_error = calc_error
        return self._layer_size

    def feed(self, input_layer):
        self._input_layer = input_layer
        self._z = input_layer @ self._weights + self._biases
        self._output_layer = self._activation_func.run(self._z)
        return self._output_layer

    def back(self, error):
        delta_error = self._get_delta(error)
        if self._calc_error:
            error = self._get_error(delta_error)

        self._weights += self._optimizer.calc_gradients(id(self._weights), self._input_layer.T @ delta_error)
        self._biases += self._optimizer.calc_gradients(id(self._biases),
                                                       np.sum(delta_error, axis=0, keepdims=True))
        return error

    def _get_error(self, delta):
        return delta @ self._weights.T

    def _get_delta(self, error):
        return error * self._activation_func.derivative(self._z)
