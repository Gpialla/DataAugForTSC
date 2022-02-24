# Imports
import os
import numpy as np

from utils.constants import PATH_UCR_ARCHIVE_2015, PATH_UCR_ARCHIVE_2018

# Constants
UCR_VERSIONS = [2015, 2018]
UCR_ARCHIVE_2015_DATASETS = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays', 'Earthquakes', 'ElectricDevices', 'FISH', 'FaceAll', 'FaceFour', 'FacesUCR', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OSULeaf', 'OliveOil', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'UWaveGestureLibraryAll', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'synthetic_control', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'yoga']
UCR_ARCHIVE_2018_DATASETS = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'ArrowHead', 'BME', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'ECG200', 'ECG5000', 'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes', 'ElectricDevices', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OSULeaf', 'OliveOil', 'PLAID', 'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']


def read_ucr(filename, delimiter="\t"):
    """
    Read UCR tsv file and split data 

    Args:
        filename (string): Path of file to read

    Returns:
        (np.array, np.array): X, Y data
    """
    if not os.path.exists(filename):
        raise ValueError("File does not exists '%s'." % (filename))

    data = np.loadtxt(filename, delimiter=delimiter)
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def load_dataset(ds_name, UCR_version=2018):
    """
    Args:
        ds_name (str): The dataset's name
        UCR_version (int, optional): The version of the UCR archive. Defaults to 2018.

    Returns:
        tuple: x_train, y_train, x_test, y_test
    """

    if not UCR_version in UCR_VERSIONS:
        raise ValueError("Parameter 'UCR_version' is '%s' and should be in '%s'" % (UCR_version, UCR_VERSIONS))
    if UCR_VERSIONS==2015:
        if not ds_name in UCR_ARCHIVE_2015_DATASETS:
            raise ValueError("Parameter 'ds_name' '%s' is an unknown dataset.\nTry with one of the following: '%s'" % (UCR_version, UCR_ARCHIVE_2015_DATASETS))
    elif UCR_VERSIONS==2018:
        if not ds_name in UCR_ARCHIVE_2018_DATASETS:
            raise ValueError("Parameter 'ds_name' '%s' is an unknown dataset.\nTry with one of the following: '%s'" % (UCR_version, UCR_ARCHIVE_2018_DATASETS))
    
    # Load raw data
    if UCR_version == 2015:
        path_dataset = os.path.join(PATH_UCR_ARCHIVE_2015, ds_name)
        x_train, y_train = read_ucr(os.path.join(path_dataset, ds_name + '_TRAIN'), delimiter=',')
        x_test,  y_test  = read_ucr(os.path.join(path_dataset, ds_name + '_TEST'), delimiter=',')
    elif UCR_version == 2018:
        path_dataset = os.path.join(PATH_UCR_ARCHIVE_2018, ds_name)
        x_train, y_train = read_ucr(os.path.join(path_dataset, ds_name + '_TRAIN.tsv'), delimiter='\t')
        x_test,  y_test  = read_ucr(os.path.join(path_dataset, ds_name + '_TEST.tsv'), delimiter='\t')
    # Data encoding
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test  = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test
