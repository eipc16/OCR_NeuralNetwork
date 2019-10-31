from numpy import np


class Optimizer(object):
    def __init__(self):
        self._delta = np.array([])
        self._weights = np.array([])
        self._biases = np.array([])
    '''
    For future implementation of Adaptive Gradient Descent when we have to keep track of learning_rate
    '''
    def copy(self):
        raise NotImplementedError

    def update(self, activation, error, cost):
        raise NotImplementedError

    def _update(self, activation, delta, weights, biases, error, cost):
        raise NotImplementedError

    def with_delta(self, delta):
        self._delta = delta
        return self

    def with_weights(self, weights):
        self._weights = weights
        return self

    def with_biases(self, biases):
        self._biases = biases
        return self
