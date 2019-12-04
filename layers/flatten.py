import numpy as np

from layers.layer import Layer


class Flatten(Layer):
    def __init__(self, layer_name='flatten'):
        super().__init__(layer_name)

    def __call__(self, previous_layer_shape, optimizer, calc_error=False):
        self._input_shape = previous_layer_shape
        self._output_shape = np.product(previous_layer_shape)
        return self._output_shape

    def feed(self, input_layer):
        out = input_layer.reshape(-1, self._output_shape)
        return out

    def back(self, error):
        error = error.reshape(-1, *self._input_shape)
        return error
