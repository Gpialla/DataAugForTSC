# Imports
import os
import numpy as np

from utils.constants import PATH_ADV_P
from data.ucr_archive import UCR_ARCHIVE_2018_DATASETS

DS_VERSIONS = ["bim", "gm", "gm-wo-const", "sgm-wo-const"]

def load_dataset(ds_name, DS_version):
    """
    Args:
        ds_name (str): The dataset's name
        DS_version (str): Version of the dataset

    Returns:
        tuple: x_train, y_train, x_test, y_test
    """

    if not ds_name in UCR_ARCHIVE_2018_DATASETS:
        raise ValueError("Parameter 'ds_name' '%s' is an unknown dataset.\nTry with one of the following: '%s'" % (UCR_version, UCR_ARCHIVE_2018_DATASETS))
    
    if not DS_version in DS_VERSIONS:
        raise ValueError("Parameter 'DS_version' '%s' is an unknown version.\nTry with one of the following: '%s'" % (DS_version, DS_VERSIONS))
    
    # Load raw data
    path_dataset = os.path.join(PATH_ADV_P, DS_version, ds_name)
    x_train = np.load(os.path.join(path_dataset, "x_train.npy"))
    y_train = np.load(os.path.join(path_dataset, "y_train.npy"))
    x_test  = np.load(os.path.join(path_dataset, "x_test.npy"))
    y_test  = np.load(os.path.join(path_dataset, "y_test.npy"))
    # Data encoding
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test  = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test
