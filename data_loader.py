import numpy as np
import pickle as pkl

def _load_file(file_name):
    file = open(file_name, mode='rb')
    data = pkl.load(file, encoding='latin1')
    file.close()
    return data

def reshape_and_vectorize(dataset):
    inputs = np.array([np.reshape(x, (784,)) for x in dataset[0]])
    results = np.array([vectorized_result(y) for y in dataset[1]])
    return inputs, results

def load_data_wrapper(file_name):
    tr_d, va_d, te_d = _load_file(file_name)
    training_data = reshape_and_vectorize(tr_d)
    validation_data = reshape_and_vectorize(va_d)
    test_data = reshape_and_vectorize(te_d)
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e