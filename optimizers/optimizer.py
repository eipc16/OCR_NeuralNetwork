import numpy as np


class Optimizer(object):
    def calc_gradients(self, identifier, gradients):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError
