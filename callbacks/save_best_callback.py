import os

from callbacks.callback import Callback
from utils.pickle_utils import save_file


class SaveBestCallback(Callback):
    def __init__(self, destination_path, file_name):
        self.destination_path = destination_path
        self.file_name = file_name
        self.best_accuracy = 0

    def on_training_begin(self, model):
        self.best_accuracy = 0

    def on_epoch_end(self, model):
        state = model.get_state()

        if state.current_validation_accuracy > self.best_accuracy:
            self.best_accuracy = state.current_validation_accuracy
            save_file(self.destination_path, self.file_name, model)
