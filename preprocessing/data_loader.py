from keras.datasets import mnist

from preprocessing.transformations import convert_to_one_hot, flatten_image

TRAINING_SIZE = 50000


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_val = flatten_image(X_train[TRAINING_SIZE:])
    y_val = convert_to_one_hot(y_train[TRAINING_SIZE:])

    X_train = flatten_image(X_train[:TRAINING_SIZE])
    y_train = convert_to_one_hot(y_train[:TRAINING_SIZE])

    X_test = flatten_image(X_test)
    y_test = convert_to_one_hot(y_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
