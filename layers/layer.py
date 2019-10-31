class Layer(object):
    def __init__(self, layer_size, layer_name='abstract-layer'):
        self._name = layer_name
        self._layer_size = layer_size

    def __call__(self, previous_layer_size, optimizer):
        raise NotImplementedError

    def get_error(self):
        raise NotImplementedError

    def update_delta(self, error):
        raise NotImplementedError

    def feed(self, x):
        raise NotImplementedError

    def update(self, x, error, cost):
        raise NotImplementedError

    def get_name(self):
        return self._name

    def get_size(self):
        return self._layer_size
