import time

import matplotlib.pyplot as plt

from callbacks.callback import Callback
from utils.pickle_utils import save_file


class PlotCallback(Callback):
    def __init__(self, destination_path):
        self.val_accuracies = []
        self.val_costs = []
        self.training_accuracies = []
        self.training_costs = []
        self.epochs = []
        self.path = destination_path
        self._start_time = 0

    def on_training_begin(self, model):
        self.val_accuracies = []
        self.val_costs = []
        self.training_accuracies = []
        self.training_costs = []
        self.epochs = []
        self._start_time = time.time()

    def on_epoch_end(self, model):
        self.epochs.append(model.get_state().current_epoch)
        self.training_accuracies.append(model.get_state().current_training_accuracy)
        self.training_costs.append(model.get_state().current_training_cost)

    def on_validation_test_end(self, model):
        self.val_accuracies.append(model.get_state().current_validation_accuracy)
        self.val_costs.append(model.get_state().current_validation_cost)

    def on_training_end(self, model):
        batch_size = model.get_state().batch_size
        epochs = model.get_state().epochs
        learning_rate = model.get_state().learning_rate
        time_diff = (time.time() - self._start_time)

        save_file(self.path + '/training/', f'training_accr_batch={batch_size}_epochs={epochs}_lr={learning_rate}.pkl',
                  (zip(self.epochs, self.training_accuracies), time_diff))

        save_file(self.path + '/training/', f'training_costs_batch={batch_size}_epochs={epochs}_lr={learning_rate}.pkl',
                  (zip(self.epochs, self.training_costs), time_diff))

        save_file(self.path + '/validation/', f'validation_costs_batch={batch_size}_epochs={epochs}_lr={learning_rate}.pkl',
                  (zip(self.epochs, self.val_costs), time_diff))

        save_file(self.path + '/validation/', f'validation_accr_batch={batch_size}_epochs={epochs}_lr={learning_rate}.pkl',
                  (zip(self.epochs, self.val_accuracies), time_diff))

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.canvas.set_window_title(f'Batch size: {model.get_state().batch_size}')
        ax.grid(True)
        ax.set_xlabel('epochs')
        ax.set_ylabel('Accuracy')

        ax.plot(self.epochs, self.training_accuracies, label='training')

        if len(self.training_accuracies) == len(self.epochs):
            ax.plot(self.epochs, self.val_accuracies, label='validation')

        ax.legend()
        plt.show()
