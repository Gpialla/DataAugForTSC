import os

def create_dir_if_not_exists(path):
    """
    Create a directory if it does not exists

    Args:
        path (str): The path of the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)