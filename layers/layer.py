import numpy as np


class Layer(object):
    def __init__(self, layer_name='abstract-layer'):
        self._name = layer_name
        self._input_layer = np.array([])
        self._output_layer = np.array([])

    def __call__(self, previous_layer_size, optimizer, calc_error=True):
        raise NotImplementedError

    def feed(self, input_layer):
        raise NotImplementedError

    def back(self, error):
        raise NotImplementedError

    @staticmethod
    def calc_output_shape(input_layer, size, stride):
        result = []
        for i in range(len(input_layer) - 1):
            result.append(int((input_layer[i] - size[i]) / stride[i]) + 1)
        return result

    @staticmethod
    def pad(input_layer, pad):
        '''
        Function for padding matrix.
        :param input_layer: input matrix of shape (N, xH, xW, C) ,
                where N - numOfInputs, xH - input height, xW - input width, C - number of channels
        :param pad: (h_pad, w_pad), h_pad: height padding_size, w_pad: width padding_size
        :return: padded_matrix of shape (N, xH + 2 * h_pad, xW + 2 * w_pad, C), with 0 on edges
        '''
        return np.pad(input_layer, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'constant', constant_values=0)

    @staticmethod
    def _remove_hpad(input_layer, h_pad):
        return input_layer[:, h_pad:-h_pad]

    @staticmethod
    def _remove_wpad(input_layer, w_pad):
        return input_layer[:, :,  w_pad:-w_pad]

    @staticmethod
    def remove_padding(input_layer, pad):
        if pad[0]:
            return Layer._remove_hpad(input_layer, pad[0])
        elif pad[1]:
            return Layer._remove_wpad(input_layer, pad[1])
        else:
            return input_layer

    def get_name(self):
        return self._name

