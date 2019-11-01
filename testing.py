from activations.relu import ReLu
from activations.sigmoid import Sigmoid
from activations.softmax import Softmax
from initializers.range_initializer import RangeInitializer
from initializers.xavier_initializer import XavierInitializer
from initializers.zero_initializer import ZeroInitializer
from layers.dense import Dense
from tests.batch_size_tests import test_batch_sizes
from tests.neuron_tests import test_layer_configs
from tests.test_activations import test_activation_functions
from tests.weights_tests import test_weight_initializers

test_weight_initializers([
    XavierInitializer(1),
    XavierInitializer(6),
    RangeInitializer(-1, 1),
    RangeInitializer(-2, 2),
    ZeroInitializer(),
    RangeInitializer(-0.05, 0.05)
])

test_activation_functions([Sigmoid(), ReLu()])

test_layer_configs([
    {
        'name': 'one_layer_5_neurons',
        'layers': [
            Dense(layer_size=5, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ]
    },
    {
        'name': 'one_layer_1_neurons',
        'layers': [
            Dense(layer_size=1, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ]
    },
    {
        'name': 'one_layer_50_neurons',
        'layers': [
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ]
    },
    {
        'name': 'one_layer_10_neurons',
        'layers': [
            Dense(layer_size=10, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ]
    },
    {
        'name': 'one_layer_28_neurons',
        'layers': [
            Dense(layer_size=28, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ]
    },
    {
        'name': 'one_layer_300_neurons',
        'layers': [
            Dense(layer_size=300, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ]
    },
    {
        'name': 'two_layers_100_50_neurons',
        'layers': [
            Dense(layer_size=100, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ]
    },
    {
        'name': 'two_layers_100_10_neurons',
        'layers': [
            Dense(layer_size=100, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ]
    },
])

test_batch_sizes([50000, 2048, 1024, 100, 32, 1])
