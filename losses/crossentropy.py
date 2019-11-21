import numpy as np

from losses.loss import Loss


class CrossEntropy(Loss):
    @staticmethod
    def _calc_error(y, y_pred):
        error = np.array(y_pred)
        error[range(y.shape[0]), y] -= 1
        return -error / y.shape[0]

    @staticmethod
    def _calc_cost(y, y_pred):
        probabilities = -np.log(y_pred[range(y.shape[0]), y])
        return np.mean(probabilities)

    def __call__(self, y, y_pred):
        y = np.argmax(y, axis=1)
        error = CrossEntropy._calc_error(y, y_pred)
        cost = CrossEntropy._calc_cost(y, y_pred)
        return error, cost

    def get_name(self):
        return 'CrossEntropy'
