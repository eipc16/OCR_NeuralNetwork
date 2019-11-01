from activations.softmax import Softmax
from callbacks.logger_callback import LoggerCallback
from callbacks.plot_callback import PlotCallback
from initializers.xavier_initializer import XavierInitializer
from layers.dense import Dense
from losses.crossentropy import CrossEntropy
from models.model import NeuralNetwork
from optimizers.gradient_descent_static import StaticGradientDescent
from tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_activation_function(activation):
    model = NeuralNetwork(
        optimizer=StaticGradientDescent(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=activation, weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f'./results/activations/{activation.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def test_activation_functions(activation_functions):
    for activation in activation_functions:
        test_single_activation_function(activation)
