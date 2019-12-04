import numpy as np

from activations.relu import ReLu
from activations.softmax import Softmax
from callbacks.logger_callback import LoggerCallback
from callbacks.plot_callback import PlotCallback
from layers.convolution2d import Convolution2D
from layers.dense import Dense
from layers.flatten import Flatten
from layers.max_pooling import MaxPooling2D
from losses.crossentropy import CrossEntropy
from models.model import NeuralNetwork
from optimizers.adam_optimizer import AdamOptimizer
from optimizers.gradient_descent_static import StaticGradientDescent
from tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_initializer_with_convo(initializer):
    model = NeuralNetwork(
        optimizer=AdamOptimizer(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=[
            Convolution2D(num_of_filters=4, kernel=[3, 3], activation_func=ReLu()),
            MaxPooling2D(pool_size=(2, 2), stride=(2, 2)),
            Flatten(),
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

def test_single_initializer(initializer):
    model = NeuralNetwork(
        optimizer=AdamOptimizer(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=[
            Flatten(),
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
        test_single_initializer_with_convo(initializer)
        test_single_initializer(initializer)

