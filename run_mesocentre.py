import os

DS_2018 = [
	'ACSF1', 
    'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
	'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
	'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
	'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction'
]

AUG_METHODS      = ["dgw", "rgw", "scaling", "windowwarp"]
AUG_EACH_EPCH    = False             
MULTI_AUG        = False
MULTI_AUG_METHOD = 'MULTI'         
ONLY_AUG_DATA    = True           
EXP_NAME         = "OneAug_Aug_Only"
NUM_ITR          = 1
CLSSF_NAME       = "inception"

EPOCHS      = 900
BATCH_SIZE  = 64

for ds_name in DS_2018:
    if not MULTI_AUG:
        for aug in AUG_METHODS:
            for itr in range(NUM_ITR):
                command = "sbatch sbatch-main.sh {} {} {} {} {} {} {} {} {} {}".format(EXP_NAME, ds_name, aug, AUG_EACH_EPCH, ONLY_AUG_DATA, MULTI_AUG_METHOD, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
                os.system(command)
                print("Run command " + command)
    else:
        # Several aug at the time
        aug_str = "'"
        for aug in AUG_METHODS: aug_str += aug + ' '
        aug_str=aug_str[:-1]+"'"

        for itr in range(NUM_ITR):
            command = "sbatch sbatch-main.sh {} {} {} {} {} {} {} {} {} {}".format(EXP_NAME, ds_name, aug_str, AUG_EACH_EPCH, ONLY_AUG_DATA, MULTI_AUG_METHOD, CLSSF_NAME, EPOCHS, BATCH_SIZE, itr)
            os.system(command)
            print("Run command " + command)