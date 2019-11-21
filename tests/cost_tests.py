from activations.relu import ReLu
from callbacks.logger_callback import LoggerCallback
from callbacks.plot_callback import PlotCallback
from initializers.xavier_initializer import XavierInitializer
from models.model import NeuralNetwork
from optimizers.gradient_descent_static import StaticGradientDescent
from layers.dense import Dense
from tests.default_config import default_parameters, X_train, y_train, X_val, y_val, X_test, y_test


def test_single_cost_and_last_layer(cost_func, last_layer):
    model = NeuralNetwork(
        optimizer=StaticGradientDescent(),
        loss=cost_func,
        layers=[
            Dense(layer_size=50, activation_func=ReLu(), weight_initializer=XavierInitializer()),
            Dense(layer_size=10, activation_func=last_layer, weight_initializer=XavierInitializer())
        ],
        callbacks=[
            LoggerCallback(),
            PlotCallback(f'./lab_3/cost/func={cost_func.get_name()}&last_layer={last_layer.get_name()}')
        ]
    )

    model.fit(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
              epochs=default_parameters['epochs'], batch_size=default_parameters['batch_size'])

    model.test(X_test, y_test)


def perform_cost_and_last_layer_tests(data_list):
    for data in data_list:
        test_single_cost_and_last_layer(data[0], data[1])
