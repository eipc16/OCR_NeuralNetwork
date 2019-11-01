import numpy as np

from initializers.initializer import Initializer


class XavierInitializer(Initializer):
    def __call__(self, shape):
        bound = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-bound, bound, size=shape)
