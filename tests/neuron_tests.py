from activations.relu import ReLu
from activations.softmax import Softmax
from callbacks.logger_callback import LoggerCallback
from callbacks.plot_callback import PlotCallback
from initializers.xavier_initializer import XavierInitializer
from layers.dense import Dense
from losses.crossentropy import CrossEntropy
from models.model import NeuralNetwork
from optimizers.gradient_descent_static import StaticGradientDescent
from tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_layer(layer_config):
    model = NeuralNetwork(
        optimizer=StaticGradientDescent(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=layer_config['layers'],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f"./results/layers/{layer_config['name']}")
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def test_layer_configs(layer_configs):
    for layer_config in layer_configs:
        test_single_layer(layer_config)
