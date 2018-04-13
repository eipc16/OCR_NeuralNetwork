import numpy as np

def logistic(x):
    np.seterr( over='ignore' )
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers):
        self.activation = logistic
        self.activation_deriv = logistic_derivative

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i]
                                ))-1)*0.25)
        self.weights.append((2*np.random.random((layers[i] + 1, layers[i +
                            1]))-1)*0.25)


    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        ones = np.ones([X.shape[0], 1])
        X = np.concatenate((X, ones), axis=1)
        min_err = np.inf
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                hidden_inputs = np.ones([self.weights[l].shape[1] + 1])
                hidden_inputs[0:-1] = self.activation(np.dot(a[l], self.weights[l]))
                a.append(hidden_inputs)
            error = y[i] - a[-1][:-1]

            if(np.abs(np.mean(error)) < min_err):
                min_err = np.abs(np.mean(error))
                #output = open('best_err.pkl', 'wb')
                #pkl.dump(self.weights, output)
                #output.close()
            print('Iteration: ' + str(k) + '/' + str(epochs))

            deltas = [error * self.activation_deriv(a[-1][:-1])]
            l = len(a) - 2

            # The last layer before the output is handled separately because of
            # the lack of bias node in output
            deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))

            for l in range(len(a) -3, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1][:-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights)-1):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta[:,:-1])
            # Handle last layer separately because it doesn't have a bias unit
            i+=1
            layer = np.atleast_2d(a[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        a = np.array(x)
        for l in range(0, len(self.weights)):
            temp = np.ones(a.shape[0]+1)
            temp[0:-1] = a
            a = self.activation(np.dot(temp, self.weights[l]))
        return a

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights;
