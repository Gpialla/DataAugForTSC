# Imports
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def z_norm(x_train, x_test):
    """
    Z-Normalization for the data

    Args:
        x_train (np.array): X_train
        x_test (np.array): X_test

    Returns:
        (np.array, np.array): (X_train, X_test), both normalized
    """
    # TODO: randomize x_train
    
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train_prep = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test_prep = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
    return x_train_prep, x_test_prep

def feature_scaling(x_train, x_test):
    """
    Args:
        x_train (np.array)
        x_test (np.array)

    Returns:
        x_train_norm, x_test_norm: x_train & x_test normalized btw -1 and 1
    """
    max_ = np.max(x_train)
    min_ = np.min(x_train)
    x_train_norm = 2. * (x_train - min_) / (max_ - min_) - 1.

    # x_test min and max are 'unknown'
    x_test_norm = 2. * (x_test - min_) / (max_ - min_) - 1.
    return x_train_norm, x_test_norm


def labels_encoding(y_train, y_test, format=None):
    """
    Encoding for the labels
    Format:
    - None : Default encoding -> 0, 1, 2, ..., use with sparse categorical cross entropy
    - OHE  : One Hot Encoding, use with categorical cross entropy

    Args:
        y_train (np.array): Train labels
        y_test (np.array): Test labels
        format (string): The format for the encoding

    Returns:
        (np.array, np.array, int): The encoded labels, the number of classes
    """

    # init the encoder
    if format == None:
        encoder = LabelEncoder()
    elif format == "OHE":
        encoder = OneHotEncoder(sparse=False)
        # Change data format by expanding dimension
        y_train = np.expand_dims(y_train, axis=1)
        y_test  = np.expand_dims(y_test, axis=1)
    else:
        raise ValueError("Error wrong parameter, either None or OHE expected!")
    
    # Concat train and test
    y_train_test = np.concatenate((y_train, y_test), axis=0)
    # Count num_classes
    num_classes = len(np.unique(y_train_test))
    # Fit the encoder & transform data
    new_y_train_test = encoder.fit_transform(y_train_test)
    # Resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]

    return new_y_train, new_y_test, num_classes, encoder