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


def single_batch_size_test(batch_size):
    model = NeuralNetwork(
        optimizer=StaticGradientDescent(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=[
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback('./results/batch_size/')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=batch_size)

    model.test(X_test, y_test)


def test_batch_sizes(batch_size_array):
    for batch_size in batch_size_array:
        single_batch_size_test(batch_size)
