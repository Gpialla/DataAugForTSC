import os

UCR_DATASETS_2018 = [
    'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
    'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
    'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
    'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
    'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200',
    'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
    'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR',
    'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
    'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
    'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
    'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
    'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
    'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
    'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
    'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
    'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
    'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
    'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
    'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
    'PLAID', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
    'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
    'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll',
    'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
    'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
    'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
    'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
    'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
]

<<<<<<< HEAD
UEA_ARCHIVE_2018_DATASETS = [
    'ArticularyWordRecognition', 'FingerMovements', 'JapaneseVowels', 
    'InsectWingbeat', 'MotorImagery', 'HandMovementDirection', 'Epilepsy', 
    'UWaveGestureLibrary', 'Cricket', 'BasicMotions', 'AtrialFibrillation', 
    'Heartbeat', 'ERing', 'SelfRegulationSCP1', 'RacketSports', 
    'CharacterTrajectories', 'NATOPS', 'SelfRegulationSCP2', 'EigenWorms', 
    'PenDigits', 'Handwriting', 'StandWalkJump', 'PEMS-SF', 
    'EthanolConcentration', 'FaceDetection', 'LSST', 'SpokenArabicDigits', 
    'Libras', 'DuckDuckGeese', 'PhonemeSpectra'
]


AUG_METHODS      = ["scaling_multi", "windowwarp_multi"]
=======
AUG_METHODS      = []
>>>>>>> 404fa2162f872a68c895f97db16fa2a2e8d16bae
AUG_EACH_EPCH    = False             
MULTI_AUG        = False
MULTI_AUG_METHOD = 'MULTI'         
ONLY_AUG_DATA    = False           
EXP_NAME         = "DA_p_test"
NUM_ITR          = 1
CLSSF_NAME       = "fcn"
ARCHIVE_NAME     = "adv_p"
ARCHIVE_VERSIONS = ["bim", "gm", "gm-wo-const", "sgm-wo-const"]

EPOCHS      = 900
BATCH_SIZE  = 64

for ds_name in UCR_DATASETS_2018:
    for archive_version in ARCHIVE_VERSIONS:
        #for itr in range(NUM_ITR):
        command = "sbatch sbatch-main.sh {} {} {} {} {} {} {} {} {} {}".format(EXP_NAME, ARCHIVE_NAME, archive_version, ds_name, AUG_EACH_EPCH, ONLY_AUG_DATA, MULTI_AUG_METHOD, CLSSF_NAME, EPOCHS, BATCH_SIZE)
        os.system(command)
        print("Run command " + command)