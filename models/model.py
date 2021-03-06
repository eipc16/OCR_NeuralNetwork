import numpy as np

from metrics.normal_accuracy import NormalAccuracy
from metrics.time_metric import Timer
from models.state import NetworkState


class NeuralNetwork:
    def __init__(self, loss, optimizer, layers=None, callbacks=None):
        self._loss = loss
        self._optimizer = optimizer
        self._layers = layers if layers else []
        self._callbacks = callbacks if layers else []
        self._state = None

    def _run_batch(self, x_batch, y_batch):
        y_pred = self.predict(x_batch)
        error, cost = self._loss(y_batch, y_pred)
        self._back_propagation(error)
        return y_pred, cost

    def _back_propagation(self, error):
        for layer in self._layers[::-1][:-1]:
            error = layer.back(error)

    @staticmethod
    def _shuffle_data(x_train, y_train):
        data_size = x_train.shape[0]
        random_indexes = np.random.permutation(data_size)
        return x_train[random_indexes], y_train[random_indexes]

    @staticmethod
    def _prepare_batches(x_train, y_train, batch_size):
        data_size = x_train.shape[0]

        x_batch = [x_train[k:k + batch_size] for k in range(0, data_size, batch_size)]
        y_batch = [y_train[k:k + batch_size] for k in range(0, data_size, batch_size)]

        return zip(x_batch, y_batch)

    def _prepare_layers(self, input_size):
        size = input_size
        for layer in self._layers:
            size = layer(size, self._optimizer)

    @Timer(dump_to_file=True, message='Training time')
    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=1000, batch_size=32):
        self._state = NetworkState()
        self._prepare_layers(x_train.shape[1:])

        self._state.batch_size = batch_size
        self._state.epochs = epochs

        self._run_callbacks('on_training_begin')

        for epoch in range(1, epochs + 1):
            self._state.current_epoch = epoch
            self._run_callbacks('on_epoch_begin')
            x_train, y_train = self._shuffle_data(x_train, y_train)
            batches = self._prepare_batches(x_train, y_train, batch_size)

            costs, predictions = [], []
            for batch in batches:
                self._state.current_batch = batch
                self._run_callbacks('on_batch_begin')
                y_pred, cost = self._run_batch(batch[0], batch[1])
                predictions.extend(y_pred)
                costs.append(cost)

                self._run_callbacks('on_batch_end')

            train_pred = self.predict(x_train[:500])
            _, train_cost = self._loss(y_train[:500], train_pred)
            self._state.current_training_accuracy = NormalAccuracy.calculate(y_train[:500], train_pred)
            self._state.current_training_cost = np.mean(train_cost)

            if x_val is not None and y_val is not None:
                self._run_callbacks('on_validation_test_begin')
                val_pred = self.predict(x_val)
                val_error, val_cost = self._loss(y_val, val_pred)
                self._state.current_validation_accuracy = NormalAccuracy.calculate(y_val, val_pred)
                self._state.current_validation_cost = val_cost
                self._run_callbacks('on_validation_test_end')

            self._run_callbacks('on_epoch_end')

        self._run_callbacks('on_training_end')

    def test(self, x, y):
        self._run_callbacks('on_test_begin')
        predictions = self.predict(x)
        accuracy = NormalAccuracy.calculate(predictions, y)
        self._state.test_accuracy = accuracy
        self._run_callbacks('on_test_end')

    def predict(self, x):
        for layer in self._layers:
            x = layer.feed(x)
        return x

    def get_state(self):
        return self._state

    def get_layers(self):
        return self._layers

    def get_optimizer(self):
        return self._optimizer

    def _run_callbacks(self, name):
        for callback in self._callbacks:
            callback.__getattribute__(name)(self)

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

