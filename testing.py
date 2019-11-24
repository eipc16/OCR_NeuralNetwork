from activations.relu import ReLu
from activations.sigmoid import Sigmoid
from activations.softmax import Softmax
from initializers.he_initializer import HeInitializer
from initializers.normal_initializer import NormalInitializer
from initializers.range_initializer import RangeInitializer
from initializers.xavier_initializer import XavierInitializer
from initializers.zero_initializer import ZeroInitializer
from layers.dense import Dense
from losses.crossentropy import CrossEntropy
from losses.mse import MeanSquaredError
from optimizers.adadelta_optimizer import AdaDeltaOptimizer
from optimizers.adagrad_optimizer import AdaGradOptimizer
from optimizers.adam_optimizer import AdamOptimizer
from optimizers.gradient_descent_static import StaticGradientDescent
from optimizers.momentum_optimizer import MomentumOptimizer
from tests.batch_size_tests import test_batch_sizes
from tests.cost_tests import perform_cost_and_last_layer_tests
from tests.initializer_tests import perform_initializer_test
from tests.neuron_tests import test_layer_configs
from tests.optimizer_tests import perform_optimizer_test
from tests.test_activations import test_activation_functions
from tests.weights_tests import test_weight_initializers

# test_weight_initializers([
#     XavierInitializer(1),
#     XavierInitializer(6),
#     RangeInitializer(-1, 1),
#     RangeInitializer(-2, 2),
#     ZeroInitializer(),
#     RangeInitializer(-0.05, 0.05)
# ])
#
# test_activation_functions([Sigmoid(), ReLu()])
#
# test_layer_configs([
#     {
#         'name': 'one_layer_5_neurons',
#         'layers': [
#             Dense(layer_size=5, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_1_neurons',
#         'layers': [
#             Dense(layer_size=1, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_50_neurons',
#         'layers': [
#             Dense(layer_size=50, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_10_neurons',
#         'layers': [
#             Dense(layer_size=10, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_28_neurons',
#         'layers': [
#             Dense(layer_size=28, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'one_layer_300_neurons',
#         'layers': [
#             Dense(layer_size=300, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'two_layers_100_50_neurons',
#         'layers': [
#             Dense(layer_size=100, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=50, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
#     {
#         'name': 'two_layers_100_10_neurons',
#         'layers': [
#             Dense(layer_size=100, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=ReLu(), weight_initializer=XavierInitializer()),
#             Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
#         ]
#     },
# ])
#
# test_batch_sizes([50000, 2048, 1024, 100, 32, 1])


initializers = [
    XavierInitializer(gain=6),
    HeInitializer(),
    NormalInitializer(loc=0, scale=1, a=10)
]

perform_initializer_test(initializers)

optimizers = [
    StaticGradientDescent(learning_rate=0.001),
    AdamOptimizer(learning_rate=0.001),
    MomentumOptimizer(learning_rate=0.001),
    AdaGradOptimizer(learning_rate=0.001),
    AdaDeltaOptimizer()
]

perform_optimizer_test(optimizers)

cost_experiments = [
    [CrossEntropy(), Softmax()],
    [CrossEntropy(), Sigmoid()],
    [MeanSquaredError(), Softmax()],
    [MeanSquaredError(), Sigmoid()]
]

perform_cost_and_last_layer_tests(cost_experiments)

