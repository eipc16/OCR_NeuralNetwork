class Initializer(object):
    def __call__(self, shape):
        raise NotImplementedError

    def get_name(self):
        return 'none'
