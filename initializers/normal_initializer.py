import numpy as np

from initializers.initializer import Initializer


class NormalInitializer(Initializer):

    def __init__(self, loc, scale, a):
        self.loc = loc
        self.scale = scale
        self.a = a

    def __call__(self, shape):
        weights = np.random.normal(self.loc, self.scale, size=shape)
        return weights * np.sqrt(self.a / shape[1])

    def get_name(self):
        return f'normal-distribution-loc={self.loc}-scale={self.scale}-a={self.a}'
