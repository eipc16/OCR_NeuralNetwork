import numpy as np

from initializers.initializer import Initializer


class NormalInitializer(Initializer):

    def __init__(self, loc, scale, a):
        self.loc = loc
        self.scale = scale
        self.a = a

    def __call__(self, shape):
        _, fan_out = self.compute_fans(shape)
        weights = np.random.normal(self.loc, self.scale, size=shape)
        return weights * np.sqrt(self.a / fan_out)

    def get_name(self):
        return f'normal-distribution-loc={self.loc}-scale={self.scale}-a={self.a}'
