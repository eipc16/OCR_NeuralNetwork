import numpy as np

from initializers.initializer import Initializer


class HeInitializer(Initializer):
    def __call__(self, shape):
        fan_in, fan_out = self.compute_fans(shape)
        return np.random.randn(fan_in, fan_out) * np.sqrt(2 / (fan_in + fan_out))

    def get_name(self):
        return f'he-initializer-'
