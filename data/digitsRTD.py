from utils.constants import PATH_DIGITS_RTD

import numpy as np
import pickle

def load_dataset():
    with open('xtrain', 'rb') as fp:
        x_train = pickle.load(fp)

    with open('xtest', 'rb') as fp:
        x_test = pickle.load(fp)

    y_train = np.load('ytrain.npy')
    y_test = np.load('ytest.npy')
    return x_train, y_train, x_test, y_test