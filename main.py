from activations.relu import ReLu
from activations.softmax import Softmax
from callbacks.logger_callback import LoggerCallback
from callbacks.save_best_callback import SaveBestCallback
from initializers.xavier_initializer import XavierInitializer
from layers.dense import Dense
from losses.crossentropy import CrossEntropy
from metrics.normal_accuracy import NormalAccuracy
from models.model import NeuralNetwork
from optimizers.gradient_descent_static import StaticGradientDescent
from preprocessing.data_loader import get_data

model = NeuralNetwork(
    optimizer=StaticGradientDescent(learning_rate=0.01),
    loss=CrossEntropy(),
    layers=[
        Dense(layer_size=100, activation_func=ReLu(), weight_initializer=XavierInitializer()),
        Dense(layer_size=10, activation_func=Softmax(), weight_initializer=XavierInitializer())
    ],
    callbacks=[
        SaveBestCallback('./results/01_10_2019_13:00', 'best_model.pkl'),
        LoggerCallback()
    ]
)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data()

model.fit(
    x_train=X_train, y_train=y_train,
    x_val=X_val, y_val=y_val,
    epochs=10,
    batch_size=32
)

predictions = model.predict(X_test)

test_acc = NormalAccuracy.calculate(predictions, y_test)

print(f'Final accuracy: {test_acc:.2f}%')
