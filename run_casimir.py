import os
import subprocess

DS_2018 = [
	'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
	'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
	'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
	'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction'
]

DS_2018 = ['Coffee']

AUG_METHODS = ["scaling", "windowwarp"]
MULTI_AUG = True
EXP_NAME = "TrainMultiAug"
NUM_ITR = 2
CLSSF_NAME = "inception"

EPOCHS      = 900
BATCH_SIZE  = 64

for ds_name in DS_2018:
    if not MULTI_AUG:
        for aug in AUG_METHODS:
            for itr in range(NUM_ITR):
                command = "python3 main.py --exp_name {} --ds_name {} --aug_method {} --model {} --num_epochs {} --batch_size {} --iter {}".format(EXP_NAME, ds_name, aug_str, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
                print("Run command " + command)
                p = subprocess.run(command, shell=True)
                print("Return code ", p.returncode)
    else:
        # Several aug at the time
        aug_str = ""
        for aug in AUG_METHODS: aug_str += aug + ' '
        aug_str=aug_str[:-1]

        for itr in range(NUM_ITR):
            command = "python3 main.py --exp_name {} --ds_name {} --aug_method {} --model {} --num_epochs {} --batch_size {} --iter {}".format(EXP_NAME, ds_name, aug_str, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
            print("Run command " + command)
            p = subprocess.run(command, shell=True)
            print("Return code ", p.returncode)
