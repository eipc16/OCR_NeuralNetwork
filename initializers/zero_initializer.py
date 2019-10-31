import numpy as np

from initializers.initializer import Initializer


class ZeroInitializer(Initializer):
    def __call__(self, shape):
        return np.zeros(shape=shape)
