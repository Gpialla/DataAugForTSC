import os
import json
import numpy as np
import pandas as pd

def create_dir_if_not_exists(path):
    """
    Create a directory if it does not exists
    Args:
        path (str): The path of the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_keras_history(history, filepath):
    """
    Save a keras history in the specified file
    Args:
        history (keras.history): Keras history
        filepath (str): The file path to save the history
    """

    df = pd.DataFrame(history.history)
    df.to_csv(filepath)


def save_predictions(preds, filepath):
    np.save(filepath, preds)

def save_records(records, filepath):
    with open(filepath, 'w') as f:
        json.dump(str(records), f)
