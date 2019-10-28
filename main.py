import numpy as np
import pickle as pkl

from data_loader import load_data_wrapper
from NeuralNetwork import NeuralNetwork

training, validation, test = load_data_wrapper('mnist.pkl')

nn = NeuralNetwork([training[0].shape[1], 384, 100, training[1].shape[1]])

T_X, T_y = training
V_X, V_y = validation
T_X, T_y = np.squeeze(T_X), np.squeeze(T_y)
V_X, V_y = np.squeeze(V_X), np.squeeze(V_y)

Test_X, Test_y = test
Test_X, Test_y = np.squeeze(Test_X), np.squeeze(Test_y)

nn.fit(T_X, T_y, epochs=1000, X_validate=V_X, y_validate=V_y)
predictions = np.zeros(Test_X.shape[0])

correct = 0
out = nn.predict(T_X)
labels = np.argmax(T_y, axis=1)
accr = (np.sum(np.where(out == labels, 1, 0)) / T_y.shape[0]) * 100;
print(f'Accuraccy {accr}%')
