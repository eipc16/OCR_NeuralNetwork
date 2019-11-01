from activations.relu import ReLu
from activations.softmax import Softmax
from callbacks.logger_callback import LoggerCallback
from callbacks.plot_callback import PlotCallback
from layers.dense import Dense
from losses.crossentropy import CrossEntropy
from models.model import NeuralNetwork
from optimizers.gradient_descent_static import StaticGradientDescent
from tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_weight_initializer(weight_initializer):
    model = NeuralNetwork(
        optimizer=StaticGradientDescent(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=weight_initializer),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=weight_initializer)
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f'./results/weigh_initializer/{weight_initializer.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def test_weight_initializers(initializers):
    for initializer in initializers:
        test_single_weight_initializer(initializer)
