import numpy as np

from activations.relu import ReLu
from activations.softmax import Softmax
from callbacks.logger_callback import LoggerCallback
from callbacks.plot_callback import PlotCallback
from initializers.he_initializer import HeInitializer
from layers.convolution2d import Convolution2D
from layers.dense import Dense
from layers.flatten import Flatten
from layers.max_pooling import MaxPooling2D
from losses.crossentropy import CrossEntropy
from models.model import NeuralNetwork
from optimizers.adam_optimizer import AdamOptimizer
from optimizers.gradient_descent_static import StaticGradientDescent
from tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_signle_convo_network(test):
    model = NeuralNetwork(
        optimizer=AdamOptimizer(learning_rate=default_parameters['learning_rate']),
        loss=CrossEntropy(),
        layers=test['layers'],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f"./lab_4_2/{test['test_name']}")
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)

conv_tests = [
    {
        'test_name': 'normal_C6x6-F4_MP1x1_F_D50_D10',
        'layers': [
            Convolution2D(num_of_filters=4, kernel=(6, 6), activation_func=ReLu()),
            MaxPooling2D(pool_size=(1, 1), stride=(1, 1)),
            Flatten(),
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
        ]
    },
    {
        'test_name': 'normal_C6x6-F4_MP4x4_F_D50_D10',
        'layers': [
            Convolution2D(num_of_filters=4, kernel=(6, 6), activation_func=ReLu()),
            MaxPooling2D(pool_size=(4, 4), stride=(4, 4)),
            Flatten(),
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
        ]
    },
    {
        'test_name': 'normal_C6x6-F4_MP7x7_F_D50_D10',
        'layers': [
            Convolution2D(num_of_filters=4, kernel=(6, 6), activation_func=ReLu()),
            MaxPooling2D(pool_size=(7, 7), stride=(7, 7)),
            Flatten(),
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
        ]
    },
    # {
    #     'test_name': 'normal_F_D50_D10',
    #     'layers': [
    #         Flatten(),
    #         Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
    #         Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
    #     ]
    # },
    # {
    #     'test_name': 'normal_C8x8-F4_MP2x2_F_D50_D10',
    #     'layers': [
    #         Convolution2D(num_of_filters=4, kernel=(8, 8), activation_func=ReLu()),
    #         MaxPooling2D(pool_size=(2, 2), stride=(2, 2)),
    #         Flatten(),
    #         Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
    #         Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
    #     ]
    # },
    # {
    #     'test_name': 'normal_C6x6-F4_MP2x2_F_D50_D10',
    #     'layers': [
    #         Convolution2D(num_of_filters=4, kernel=(6, 6), activation_func=ReLu()),
    #         MaxPooling2D(pool_size=(2, 2), stride=(2, 2)),
    #         Flatten(),
    #         Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
    #         Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
    #     ]
    # },
    # {
    #     'test_name': 'normal_C1x1-F4_MP2x2_F_D50_D10',
    #     'layers': [
    #         Convolution2D(num_of_filters=4, kernel=(6, 6), activation_func=ReLu()),
    #         MaxPooling2D(pool_size=(2, 2), stride=(2, 2)),
    #         Flatten(),
    #         Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
    #         Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
    #     ]
    # },
    # {
    #     'test_name': 'normal_C3x3-F1_MP2x2_F_D50_D10',
    #     'layers': [
    #         Convolution2D(num_of_filters=1, kernel=(3, 3), activation_func=ReLu()),
    #         MaxPooling2D(pool_size=(2, 2), stride=(2, 2)),
    #         Flatten(),
    #         Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
    #         Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
    #     ]
    # },
    {
        'test_name': 'normal_C3x3-F2_MP2x2_F_D50_D10',
        'layers': [
            Convolution2D(num_of_filters=2, kernel=(3, 3), activation_func=ReLu()),
            MaxPooling2D(pool_size=(2, 2), stride=(2, 2)),
            Flatten(),
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
        ]
    },
    {
        'test_name': 'normal_C3x3-F4_MP2x2_F_D50_D10',
        'layers': [
            Convolution2D(num_of_filters=4, kernel=(3, 3), activation_func=ReLu()),
            MaxPooling2D(pool_size=(2, 2), stride=(2, 2)),
            Flatten(),
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
        ]
    },
    {
        'test_name': 'normal_C3x3-F8_MP2x2_F_D50_D10',
        'layers': [
            Convolution2D(num_of_filters=8, kernel=(3, 3), activation_func=ReLu()),
            MaxPooling2D(pool_size=(2, 2), stride=(2, 2)),
            Flatten(),
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
            Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
        ]
    },
    # {
    #     'test_name': 'normal_C3x3-F8_MP4x4_F_D50_D10',
    #     'layers': [
    #         Convolution2D(num_of_filters=8, kernel=(3, 3), activation_func=ReLu()),
    #         MaxPooling2D(pool_size=(4, 4), stride=(4, 4)),
    #         Flatten(),
    #         Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
    #         Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
    #     ]
    # },
    # {
    #     'test_name': 'normal_C3x3-F8_MP7x7_F_D50_D10',
    #     'layers': [
    #         Convolution2D(num_of_filters=8, kernel=(3, 3), activation_func=ReLu()),
    #         MaxPooling2D(pool_size=(7, 7), stride=(7, 7)),
    #         Flatten(),
    #         Dense(layer_size=50, activation_func=ReLu(), weight_initializer=HeInitializer()),
    #         Dense(layer_size=10, activation_func=Softmax(), weight_initializer=HeInitializer())
    #     ]
    # },
]

def perform_convo_tests(test_list):
    for test in test_list:
        test_signle_convo_network(test)

perform_convo_tests(conv_tests)