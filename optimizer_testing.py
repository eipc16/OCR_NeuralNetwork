from optimizers.adadelta_optimizer import AdaDeltaOptimizer
from optimizers.adagrad_optimizer import AdaGradOptimizer
from optimizers.adam_optimizer import AdamOptimizer
from optimizers.gradient_descent_static import StaticGradientDescent
from optimizers.momentum_optimizer import MomentumOptimizer
from tests.optimizer_tests import perform_optimizer_test

optimizers = [
    StaticGradientDescent(learning_rate=0.001),
    AdamOptimizer(learning_rate=0.001),
    MomentumOptimizer(learning_rate=0.001),
    AdaGradOptimizer(learning_rate=0.001),
    AdaDeltaOptimizer()
]

perform_optimizer_test(optimizers)
