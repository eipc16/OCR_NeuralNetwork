import numpy as np

from initializers.initializer import Initializer


class HeInitializer(Initializer):
    def __call__(self, shape):
        return np.random.randn(shape[0], shape[1]) * np.sqrt(1 / (shape[0] + shape[1]))

    def get_name(self):
        return f'he-initializer-'
