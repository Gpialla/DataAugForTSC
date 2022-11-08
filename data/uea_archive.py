# Imports
import os
import numpy as np

from sktime.datasets import load_from_tsfile

from utils.constants import PATH_UEA_ARCHIVE_2018

# Constants
UEA_ARCHIVE_2018_DATASETS = ['ArticularyWordRecognition', 'FingerMovements', 'JapaneseVowels', 'InsectWingbeat', 'MotorImagery', 'HandMovementDirection', 'Epilepsy', 'UWaveGestureLibrary', 'Cricket', 'BasicMotions', 'AtrialFibrillation', 'Heartbeat', 'ERing', 'SelfRegulationSCP1', 'RacketSports', 'CharacterTrajectories', 'NATOPS', 'SelfRegulationSCP2', 'EigenWorms', 'PenDigits', 'Handwriting', 'StandWalkJump', 'PEMS-SF', 'EthanolConcentration', 'FaceDetection', 'LSST', 'SpokenArabicDigits', 'Libras', 'DuckDuckGeese', 'PhonemeSpectra']

def read_uea(filename, delimiter="\t"):
    """
    Read UEA ts file

    Args:
        filename (string): Path of file to read

    Returns:
        (np.array, np.array): X, Y data
    """
    if not os.path.exists(filename):
        raise ValueError("File does not exists '%s'." % (filename))

    x, y = load_from_tsfile(filename)
    return x, y.astype(int)

def load_dataset(ds_name):
    """
    Args:
        ds_name (str): The dataset's name
        UCR_version (int, optional): The version of the UCR archive. Defaults to 2018.

    Returns:
        tuple: x_train, y_train, x_test, y_test
    """

    # Load raw data

    path_dataset = os.path.join(PATH_UEA_ARCHIVE_2018, ds_name)
    x_train, y_train = read_uea(os.path.join(path_dataset, ds_name + '_TRAIN.ts'), delimiter=',')
    x_test,  y_test  = read_uea(os.path.join(path_dataset, ds_name + '_TEST.ts'), delimiter=',')

    return x_train, y_train, x_test, y_test
