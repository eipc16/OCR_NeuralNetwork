import pickle as pkl

from callbacks.callback import Callback


class SaveBestCallback(Callback):
    def on_epoch_end(self, model):
        pass
