import numpy as np

from initializers.initializer import Initializer


class XavierInitializer(Initializer):
    def __init__(self, gain=6):
        self._gain = gain

    def __call__(self, shape):
        fan_in, fan_out = self.compute_fans(shape)
        bound = np.sqrt(self._gain / (fan_in + fan_out))
        return np.random.uniform(-bound, bound, size=shape)

    def get_name(self):
        return f'xavier-gain={self._gain}'
