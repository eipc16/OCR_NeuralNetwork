class Callback(object):
    def on_training_begin(self, model):
        pass

    def on_training_end(self, model):
        pass

    def on_batch_begin(self, model):
        pass

    def on_batch_end(self, model):
        pass

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        pass

    def on_validation_test_begin(self, model):
        pass

    def on_validation_test_end(self, model):
        pass

    def on_test_begin(self, model):
        pass

    def on_test_end(self, model):
        pass
