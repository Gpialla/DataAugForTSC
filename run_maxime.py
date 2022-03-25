import os
import subprocess

UCR_2018_DATASETS = [
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

AUG_METHODS      = ["scaling", "windowwarp", "dgw", "rgw"]
AUG_EACH_EPCH    = False
MULTI_AUG        = True
MULTI_AUG_METHOD = 'MULTI'
ONLY_AUG_DATA    = False
EXP_NAME = "Multi_NoAEE_NoOAD"
NUM_ITR = 5
CLSSF_NAME = "inception"

EPOCHS      = 300
BATCH_SIZE  = 64

for ds_name in UCR_2018_DATASETS:
    if not MULTI_AUG:
        for aug in AUG_METHODS:
            for itr in range(NUM_ITR):
                command = "python3 main.py --exp_name {} --ds_name {} \
                        --aug_method {} --aug_each_epch {} --only_aug_data {}\
                        --model {} --num_epochs {} --batch_size {} --iter {}"\
                        .format(EXP_NAME, ds_name, aug, AUG_EACH_EPCH, ONLY_AUG_DATA, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
                print("Run command " + command)
                path_log = "./logs_%s_%s_%i"%(aug, ds_name, itr)
                print("outputs in: " + path_log)
                with open(path_log, "w") as f:
                    p = subprocess.run(command, shell=True, stdout=f)
                print("Return code ", p.returncode)
    else:
        # Several aug at the time
        aug_str = ""
        for aug in AUG_METHODS: aug_str += aug + ' '
        aug_str=aug_str[:-1]

        for itr in range(NUM_ITR):
            command = "python3 main.py --exp_name {} --ds_name {} \
                    --aug_method {} --aug_each_epch {} --only_aug_data {} --multi_aug_method {}\
                    --model {} --num_epochs {} --batch_size {} --iter {}"\
                    .format(EXP_NAME, ds_name, aug_str, AUG_EACH_EPCH, ONLY_AUG_DATA, MULTI_AUG_METHOD, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
            print("Run command " + command)
            p = subprocess.run(command, shell=True)
            print("Return code ", p.returncode)
