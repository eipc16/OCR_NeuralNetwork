import pickle as pkl
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelBinarizer
import math

def compress_hog(x):
    size = int(math.sqrt(x.shape[0]))
    img = np.reshape(x, (size, size))
    hog_image = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=False, block_norm='L1')
    return hog_image

def compress_data(x_data):
    return np.array(list(map(lambda x: compress_hog(x), x_data)))

def compress_training_set(in_file, out_file):
    print("Compressing file: '" + in_file + "'.")
    x_train, y_train = pkl.load(open(in_file, mode='rb'))
    hog_x = compress_data(x_train)
    y_train = LabelBinarizer().fit_transform(y_train)
    output = open(out_file, mode='wb')
    pkl.dump((hog_x, y_train), output)
    output.close()
    print("Successful compression to file: '" + out_file + "'.")
