import numpy as np

from metrics.metric import Metric


class NormalAccuracy(Metric):
    name = 'normal-accuracy'

    @staticmethod
    def calculate(y_1, y_2):
        y_1, y_2 = np.asarray(y_1), np.asarray(y_2)
        results = np.where(np.argmax(y_1, axis=1) == np.argmax(y_2, axis=1), 1, 0)
        return np.mean(results)
