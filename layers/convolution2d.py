import numpy as np
from numpy.lib.stride_tricks import as_strided

from activations.relu import ReLu
from initializers.xavier_initializer import XavierInitializer
from initializers.zero_initializer import ZeroInitializer
from layers.layer import Layer


class Convolution2D(Layer):
    def __init__(self, num_of_filters, kernel, stride=(1, 1), kernel_initializer=XavierInitializer(),
                 bias_initializer=ZeroInitializer(),
                 activation_func=ReLu(), layer_name='Convo2D'):
        super().__init__(layer_name)
        self._num_of_filters = num_of_filters
        self._kernel = kernel
        self._stride = stride
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._activation_func = activation_func
        self._z = None

    def __call__(self, previous_layer_shape, optimizer, calc_error=False):
        self._input_shape = previous_layer_shape
        num_of_channels = previous_layer_shape[2]
        self._weights = self._kernel_initializer((self._kernel[0], self._kernel[1],
                                                  num_of_channels, self._num_of_filters))
        self._biases = self._bias_initializer((1, 1, 1, self._num_of_filters))
        self._optimizer = optimizer
        self._calc_error = calc_error
        nH, nW = Layer.calc_output_shape(previous_layer_shape, self._kernel, self._stride)
        self._output_shape = (nH, nW, self._num_of_filters)
        return self._output_shape

    def convolve2d(self, input_layer, num_of_inputs, out_shape):
        xH, xW, filters = out_shape
        stride_height, stride_width = self._stride
        kernel_1, kernel_2, num_of_channels, _ = self._weights.shape
        stride_1, stride_2, stride_3, stride_4 = input_layer.strides
        view_shape = (num_of_inputs, xH, xW, kernel_1, kernel_2, filters)
        strides_shape = (stride_1, stride_2 * stride_height, stride_3 * stride_width, stride_2, stride_3, stride_4)
        # we create a view of the layer of shape view_shape, width_strides of shape strides_shape
        # the last argument tells numpy to not modify source matrix
        strided_layer_view = as_strided(input_layer.astype(np.float16), shape=view_shape, strides=strides_shape, writeable=False)\
            .astype(np.float16)

        # (self._kernel[0], self._kernel[1], previous_layer_shape[2], self._num_of_filters)
        #   h - kernel height, w - kernel width, W - previous_layer_shape[2], f - num_of_filters, c - number of channels
        # M - number of inputs, H - layer HEIGHT, W - layer WIDTH
        return np.einsum('MHWhwc,hwcf->MHWf', strided_layer_view.astype(np.float16), self._weights.astype(np.float16))\
            .astype(np.float16) + self._biases

    def feed(self, input_layer):
        self._input_layer = input_layer.astype(np.float16)
        self._z = self.convolve2d(input_layer, input_layer.shape[0], self._output_shape)
        self._output_layer = self._activation_func.run(self._z)
        return self._output_layer.astype(np.float16)

    def _get_error(self, delta, num_of_inputs, out_shape):
        mH, mW = out_shape
        kernel_1, kernel_2, _, filters = self._weights.shape
        stride_1, stride_2, stride_3, stride_4 = delta.strides
        view_shape = (num_of_inputs, mH, mW, kernel_1, kernel_2, filters)
        strides_shape = (stride_1, stride_2, stride_3, stride_2, stride_3, stride_4)
        strided_delta_view = as_strided(delta, shape=view_shape, strides=strides_shape, writeable=False)\
            .astype(np.float16)
        return np.einsum('MHWhwf,hwcf->MHWc', strided_delta_view.astype(np.float16),
                         np.rot90(self._weights, 2, axes=(0, 1)).astype(np.float16))\
            .astype(np.float16)

    def _get_delta(self, error):
        delta_layer = error * self._activation_func.derivative(self._z)
        delta_layer = np.insert(delta_layer, np.repeat(np.arange(1, delta_layer.shape[1]), self._stride[0] - 1),
                                values=0, axis=1).astype(np.float16)
        delta_layer = np.insert(delta_layer, np.repeat(np.arange(1, delta_layer.shape[2]), self._stride[1] - 1),
                                values=0, axis=2).astype(np.float16)
        return delta_layer

    def back(self, error):
        layer_delta = self._get_delta(error).astype(np.float16)
        if self._calc_error:
            mH, mW, _ = self._output_shape
            error = self._get_error(layer_delta, layer_delta.shape[0], (mH, mW)).astype(np.float16)

        # (ker_1, ker_2, 1)
        kernel_1, kernel_2, num_of_channels, _ = self._weights.shape
        # (batch_size, 28, 28)
        layer_1, layer_2, layer_3 = layer_delta.shape
        stride_1, stride_2, stride_3, stride_4 = self._input_layer.strides
        view_shape = (kernel_1, kernel_2, num_of_channels, layer_1, layer_2, layer_3)
        strides_shape = (stride_2, stride_3, stride_4, stride_1, stride_2, stride_3)
        strided_input_view = as_strided(self._input_layer.astype(np.float16), shape=view_shape, strides=strides_shape,
                                        writeable=False)\
            .astype(np.float16)
        delta_weights = np.einsum('HWcMhw,MHwf->HWcf', strided_input_view.astype(np.float16),
                                  layer_delta.astype(np.float16)).astype(np.float16)
        delta_biases = np.sum(layer_delta, axis=(0, 1, 2), keepdims=True, dtype=np.float16)

        self._weights += self._optimizer.calc_gradients(id(self._weights), delta_weights)
        self._biases += self._optimizer.calc_gradients(id(self._biases), delta_biases)

        return error
