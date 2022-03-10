import os
from pyexpat import model



DS_2018 = [
	'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
	'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
	'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
	'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction'
]

AUG_METHODS = ["scaling", "windowwarp"]
MULTI_AUG = True
EXP_NAME = "TrainMultiAug"
NUM_ITR = 1
CLSSF_NAME = "inception"

EPOCHS      = 700
BATCH_SIZE  = 64

for ds_name in DS_2018[0]:
    if not MULTI_AUG:
        for aug in AUG_METHODS:
            for itr in range(NUM_ITR):
                command = "sbatch sbatch-main.sh {} {} {} {} {} {} {}".format(EXP_NAME, ds_name, aug, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
                os.system(command)
                print("Run command " + command)
    else:
        # Several aug at the time
        aug_str = str()
        for aug in AUG_METHODS: aug_str += aug + ' '
        aug_str=aug_str[:-1]

        for itr in range(NUM_ITR):
            command = "sbatch sbatch-main.sh {} {} {} {} {} {} {}".format(EXP_NAME, ds_name, aug_str, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
            os.system(command)
            print("Run command " + command)