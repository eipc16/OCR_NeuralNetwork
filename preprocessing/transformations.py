import numpy as np


def convert_to_one_hot(array):
    return np.eye(np.max(array) + 1, dtype=np.int32)[array]


def reshape(image, shape):
    flattened = np.reshape(image, (-1, *shape)).astype(np.float16)
    return flattened / 255.0
