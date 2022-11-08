import argparse
import os
import time

import tensorflow as tf

from models.helper import MODEL_LIST

from data_aug.helper import AUG_METHODS, MultiAugMethod
from data.helper import PREPROCESSINGS_NAMES
from data.data_preprocessing import labels_encoding
from data.helper import load_ucr_dataset, get_preprocessing_by_name

from utils.constants import DEFAULT_OUTPUT_DIR
from utils.utils import save_predictions

def training(args):

    # Load data
    x_train, y_train, x_test, y_test = load_ucr_dataset(args.ds_name, args.ucr_version)
    # Preprocessing
    preproc = get_preprocessing_by_name(args.preproc)
    x_train, x_test = preproc(x_train, x_test)
    y_train, y_test, n_classes, _ = labels_encoding(y_train, y_test, format="OHE")
 
    # Get output directory
    OUTPUT_DIR = os.path.join(args.output_dir, args.exp_name, args.model, str(args.aug_method), args.ds_name, "Itr_%i" % args.iter)

    # Make predictions using best model
    PATH_BEST_WEIGHTS = os.path.join(OUTPUT_DIR, "best_weights.h5")
    model = tf.keras.models.load_model(PATH_BEST_WEIGHTS)
    y_pred_train = model.predict(x_train)
    #y_pred_test  = model.predict(x_test)
    save_predictions(y_pred_train, os.path.join(OUTPUT_DIR, "y_preds_train.npy"))
    #save_predictions(y_pred_test,  os.path.join(OUTPUT_DIR, "y_preds_test.npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training dl models for Time Series Classification.")

    parser.add_argument("--exp_name", type=str, help="Unique name identifier for the experiment")
    # Args for datasets
    parser.add_argument("--ucr_version", type=int, default=2018, choices=[2015, 2018], help="The name of the dataset.")
    parser.add_argument("--ds_name", type=str, help="The dataset's name")
    parser.add_argument("--aug_method", type=str, default=None, choices=AUG_METHODS.keys(), nargs='+')
    parser.add_argument("--multi_aug_method", type=str, default='MULTI', choices=('MULTI', 'MIXED'))
    parser.add_argument("--aug_each_epch", choices=('True', 'False'), default='True', help="New data aug after each epoch")
    parser.add_argument("--only_aug_data", choices=('True', 'False'), default='False', help="Use only augmented data")   
    parser.add_argument("--preproc", default="z_norm", choices=PREPROCESSINGS_NAMES, help="Method used to preprocess the data")
    parser.add_argument("--shuffle", choices=('True', 'False'), default='True', help="Shuffle data at the end of each epoch")
    # Args for directories
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    # Training params
    parser.add_argument("--model", choices=MODEL_LIST.keys())
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss", type=str, default="categorical_crossentropy")

    parser.add_argument("--iter", type=int, default=0, help="The iteration index")
    args = parser.parse_args()

    # Converting param to Enum value
    if args.multi_aug_method == 'MULTI':
        args.multi_aug_method = MultiAugMethod.MULTI
    elif args.multi_aug_method == 'MIXED':
        args.multi_aug_method = MultiAugMethod.MIXED
    
    args.aug_method = args.aug_method[0]

    # Handling text to boolean
    args.aug_each_epch = args.aug_each_epch == 'True'
    args.only_aug_data = args.only_aug_data == 'True'
    args.shuffle       = args.shuffle       == 'True'

    training(args)
