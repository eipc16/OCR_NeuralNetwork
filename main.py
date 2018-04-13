from predict import predict
import numpy as np
import pickle as pkl
from data_compressor import compress_training_set
from sklearn.metrics import classification_report

TEST_SIZE = 2500;

#compress_training_set('train.pkl', 'hog_train.pkl')

x_test, y_test = pkl.load(open('train.pkl', mode='rb'))

x_test = x_test[0:TEST_SIZE]
y_test = y_test[0:TEST_SIZE]

result = predict(x_test)

correct = 0

for i in range(result.shape[0]):
    if(result[i] == y_test[i]):
        correct += 1

acc = correct / y_test.shape[0]

print(classification_report(y_test, result))
print('%.2f' % (acc * 100))
