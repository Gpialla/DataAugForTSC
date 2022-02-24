import argparse
import os

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

from models.helper import MODEL_LIST, get_model_by_name

from data_aug.helper import AUG_METHODS, get_aug_by_name, SequenceDataAugmentation
from data.helper import PREPROCESSINGS_NAMES
from data.data_preprocessing import labels_encoding
from data.helper import load_ucr_dataset, get_preprocessing_by_name

from utils.constants import DEFAULT_OUTPUT_DIR
from utils.utils import create_dir_if_not_exists

def training(args):

    # Load data
    x_train, y_train, x_test, y_test = load_ucr_dataset(args.ds_name, args.ucr_version)
    # Preprocessing
    preproc = get_preprocessing_by_name(args.preproc)
    x_train, x_test = preproc(x_train, x_test)
    y_train, y_test, n_classes, _ = labels_encoding(y_train, y_test, format="OHE")
    
    # Load data as sequence
    seq_data = SequenceDataAugmentation(x_train, y_train, args.batch_size, aug_method=args.aug_method, shuffle=args.shuffle)

    # For each iteration
    for i_iter in range(args.num_iters):
        # Create output directory

        OUTPUT_DIR = os.path.join(args.output_dir, args.exp_name, args.model, str(args.aug_method), args.ds_name, "Itr_%i" % i_iter)
        create_dir_if_not_exists(OUTPUT_DIR)

        # Load model
        model = get_model_by_name(args.model)       # Get the model
        model = model(x_train[0].shape, n_classes)  # Initialize the model
        model.compile(optimizer=args.optimizer, loss=args.loss, metrics=['accuracy'])

        # Callbacks
        model_checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_weights.h5"), monitor='loss', save_best_only=True)
        csv_logger       = CSVLogger(os.path.join(OUTPUT_DIR, 'history.csv'), append=True)
        reduce_lr        = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=50, mode='auto', min_lr=1e-4)
        early_stop       = EarlyStopping(monitor="loss", patience=100)
        callbacks        = [model_checkpoint, reduce_lr, early_stop, csv_logger]

        # Start training
        model.save_weights(os.path.join(OUTPUT_DIR, "init_weights.h5"))
        model.fit(
            seq_data,
            epochs=args.num_epochs,
            callbacks=callbacks,
            verbose=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training dl models for Time Series Classification.")

    parser.add_argument("--exp_name", type=str, help="Unique name identifier for the experiment")
    # Args for datasets
    parser.add_argument("--ucr_version", type=int, default=2018, choices=[2015, 2018], help="The name of the dataset.")
    parser.add_argument("--ds_name", type=str, help="The dataset's name")
    parser.add_argument("--aug_method", type=str, default=None, choices=AUG_METHODS.keys())
    parser.add_argument("--preproc", default="z_norm", choices=PREPROCESSINGS_NAMES, help="Method used to preprocess the data")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle data at the end of each epoch")

    # Args for directories
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    # Training params
    parser.add_argument("--model", choices=MODEL_LIST.keys())
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss", type=str, default="categorical_crossentropy")

    parser.add_argument("--num_iters", type=int, default=1, help="The number of training to do.")
    args = parser.parse_args()

    training(args)