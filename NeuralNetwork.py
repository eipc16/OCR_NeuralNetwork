import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def square_loss(y_pred, label):
    return np.mean(np.power(y_pred - label, 2))

def cross_entropy(y_pred, label):
    y_temp = label * np.log(y_pred)
    return -np.sum(y_temp)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=1, keepdims=True)

def fn_loss(y, y_pred):
    error = y - y_pred
    cost = np.mean(np.square(error))
    error = error / y.shape[0]
    return error, cost

class NeuralNetwork:
    def __init__(self, layers):
        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative
        self.loss = cross_entropy
        self._eta = 0.2
        self._batch_size = 32
        self.weights = []
        self.layers = layers
        self.weights = [np.random.randn(size1, size2) * np.sqrt(2 / size2) for size1, size2 in zip(layers[1:], layers[:-1])]
        self.biases = [np.random.randn(1, size) for size in layers[1:]]
    
    def forward(self, a):
        # a = np.reshape(a, (len(a), 1))
        self.activations, self.z = [], []
        self.activations.append(a)

        last_layer = len(self.weights)

        for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            z = a @ weights.T + biases

            if(i == last_layer - 1):
                a = softmax(z)
            else:
                a = self.activation(z)

            self.z.append(z)
            self.activations.append(a)
        return a

    def backpropagation(self, x, error):
        delta = error * self.activation_derivative(self.z[-1])
        update = self.activations[-2].T @ delta
        
        for i in range(len(self.layers) - 2, -1, -1):         
            self.weights[i] = self.weights[i] - self._eta * update.T
            self.biases[i] = self.biases[i] - self._eta * np.sum(delta, axis=0, keepdims=True)

            if i != 0:
                delta = delta @ self.weights[i] * self.activation_derivative(self.z[i - 1])
                update = self.activations[i - 1].T @ delta

    def fit(self, X_data, y_data, X_validate=None, y_validate=None, epochs=10000):
        n = X_data.shape[0]
        
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_data, y_data = X_data[perm], y_data[perm]

            batches_X = [X_data[k:k + self._batch_size] for k in range(0, n, self._batch_size)]
            batches_y = [y_data[k:k + self._batch_size] for k in range(0, n, self._batch_size)]
            for i in range(len(batches_X)):
                X, y = batches_X[i], batches_y[i]
                a = self.forward(X)
                error, cost = fn_loss(a, y)
                self.backpropagation(X, error)

                # if i % 10 == 0:
                #     print(f'Epoch: {epoch}/{epochs} | Batch: {i}/{len(batches_X)} | Cost: {cost}')


            if X_validate is not None and y_validate is not None:
                validate_prediction = self.predict(X_validate)
                labels = np.argmax(y_validate, axis=1)
                accr = (np.sum(np.where(validate_prediction == labels, 1, 0)) / y_validate.shape[0]) * 100;
                print(f'Epoch {epoch} | Cost: {cost} |  Validation accuraccy {accr}%')
            else:
                print(f'Epoch {epoch} | cost: {cost}')

    def predict(self, x):
        a = self.forward(x)
        return np.argmax(a, axis=1)

    @staticmethod
    def check_prediction(y_pred, output):
        return np.sum(y_pred == output) / len(y_pred)

    def loss(self, y_pred, output):
        return self.loss(y_pred, output) 

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights;