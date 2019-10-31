class Activation(object):
    name = 'activation'

    def run(self, z):
        raise NotImplementedError

    def derivative(self, a):
        raise NotImplementedError
