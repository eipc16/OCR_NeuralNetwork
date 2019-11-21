from activations.sigmoid import Sigmoid
from activations.softmax import Softmax
from losses.crossentropy import CrossEntropy
from losses.mse import MeanSquaredError
from tests.cost_tests import perform_cost_and_last_layer_tests

cost_experiments = [
    [CrossEntropy(), Softmax()],
    [CrossEntropy(), Sigmoid()],
    [MeanSquaredError(), Softmax()],
    [MeanSquaredError(), Sigmoid()]
]

perform_cost_and_last_layer_tests(cost_experiments)
