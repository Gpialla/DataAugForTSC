import os



DS_2018 = [
	'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
	'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
	'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
	'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction'
]

NUM_ITR = 5
CLSSF_NAME = "inception"

for ds_name in DS_2018:
    for i in range(NUM_ITR):
        command = "sbatch sbatch-main.sh {} {} {} {}".format("UCRArchive_2018", ds_name, "inception", str(i))
        os.system(command)
        print("Run command " + command)
