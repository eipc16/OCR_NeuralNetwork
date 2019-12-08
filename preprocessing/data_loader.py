from keras.datasets import mnist

from preprocessing.transformations import convert_to_one_hot, reshape

TRAINING_SIZE = 5000
VALIDATION_SIZE = 1000
TEST_SIZE = 1000

def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_val = reshape(X_train[TRAINING_SIZE:TRAINING_SIZE + VALIDATION_SIZE], (28, 28, 1))
    y_val = convert_to_one_hot(y_train[TRAINING_SIZE:TRAINING_SIZE + VALIDATION_SIZE])

    X_train = reshape(X_train[:TRAINING_SIZE], (28, 28, 1))
    y_train = convert_to_one_hot(y_train[:TRAINING_SIZE])

    X_test = reshape(X_test[:TEST_SIZE], (28, 28, 1))
    y_test = convert_to_one_hot(y_test[:TEST_SIZE])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
