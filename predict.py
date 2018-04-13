import numpy as np
import pickle as pkl
from NeuralNetwork import NeuralNetwork
from data_compressor import compress_data

num_epochs = 100000

def predict(x):
    hog_test_x = compress_data(x)

    x_train, y_train = pkl.load(open('hog_train.pkl', mode='rb'))
    nn = NeuralNetwork([x_train.shape[1], 384, y_train.shape[1]])
    print('Starting training with test sample size: ' + str(hog_test_x.shape[0]) + ' and number of iterations: ' + str(num_epochs))
    nn.fit(x_train[10000:], y_train[10000:], epochs=num_epochs)
    print('Training has been successfully completed.')
    #nn.setweights(pkl.load(open('NN_weights.pkl')))

    output = open('NN_weights.pkl', 'wb')
    pkl.dump(nn.get_weights(), output)
    output.close()
    #test, test, test, test, test

    predictions = np.zeros(x.shape[0])
    print('Starting prediction with test sample: ' + str(hog_test_x.shape[0]))
    for i in range(x.shape[0]):
        out = nn.predict(hog_test_x[i])
        predictions[i] = np.argmax(out)

    print('The prediction proccess has been completed.')
    return predictions
