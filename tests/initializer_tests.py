from activations.relu import ReLu
from activations.softmax import Softmax
from callbacks.logger_callback import LoggerCallback
from callbacks.plot_callback import PlotCallback
from layers.dense import Dense
from losses.crossentropy import CrossEntropy
from models.model import NeuralNetwork
from optimizers.gradient_descent_static import StaticGradientDescent
from tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_initializer(initializer):
    model = NeuralNetwork(
        optimizer=StaticGradientDescent(),
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=initializer),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=initializer)
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f'./lab_3/initializers/{initializer.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def perform_initializer_test(initializer_list):
    for initializer in initializer_list:
        test_single_initializer(initializer)

