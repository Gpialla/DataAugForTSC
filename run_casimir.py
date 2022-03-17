import os
import subprocess

DS_2018 = [
	'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
	'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
	'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
	'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction'
]

AUG_METHODS = ["scaling", "windowwarp", "dgw", "rgw"]
AUG_EACH_EPCH = True
MULTI_AUG     = False
EXP_NAME = "TrainOneAug_NotEachEpch"
NUM_ITR = 5
CLSSF_NAME = "inception"


EPOCHS      = 900
BATCH_SIZE  = 64

for ds_name in DS_2018:
    if not MULTI_AUG:
        for aug in AUG_METHODS:
            for itr in range(NUM_ITR):
                command = "python3 main.py --exp_name {} --ds_name {} --aug_method {} --aug_each_epch {} --model {} --num_epochs {} --batch_size {} --iter {}".format(EXP_NAME, ds_name, aug, AUG_EACH_EPCH, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
                print("Run command " + command)
                path_log = "%s_%s_%i"%(aug, ds_name, itr)
                print("outputs in: " + path_log)
                with open(path_log, "w") as f:
                    p = subprocess.run(command, stdout=f)
                print("Return code ", p.returncode)
    else:
        # Several aug at the time
        aug_str = ""
        for aug in AUG_METHODS: aug_str += aug + ' '
        aug_str=aug_str[:-1]

        for itr in range(NUM_ITR):
            command = "python3 main.py --exp_name {} --ds_name {} --aug_method {} --aug_each_epch {} --model {} --num_epochs {} --batch_size {} --iter {}".format(EXP_NAME, ds_name, aug_str, AUG_EACH_EPCH, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
            print("Run command " + command)
            p = subprocess.run(command, shell=True)
            print("Return code ", p.returncode)
