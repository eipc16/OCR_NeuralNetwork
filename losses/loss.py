class Loss(object):
    def __call__(self, y, y_pred):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError
