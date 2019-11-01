import numpy as np

from initializers.initializer import Initializer


class RangeInitializer(Initializer):
    def __init__(self, lower_bound, upper_bound):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def __call__(self, shape):
        return np.random.uniform(low=self._lower_bound, high=self._upper_bound, size=shape)

    def get_name(self):
        return f'range-low={self._lower_bound}-high={self._upper_bound}'
