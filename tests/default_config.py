from preprocessing.data_loader import get_data

default_parameters = {
    'epochs': 40,
    'batch_size': 32,
    'learning_rate': 0.0002,
}

(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data()