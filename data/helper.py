# Imports
from data.ucr_archive import load_dataset
from data.ucr_archive import UCR_ARCHIVE_2015_DATASETS, UCR_ARCHIVE_2018_DATASETS, UCR_VERSIONS
from data.adv_p_dataset import load_dataset as load_adv_p

PREPROCESSINGS_NAMES = ["z_norm", "feature_scaling"]

def get_preprocessing_by_name(name):
    """
    Args:
        name (str): The name of the method.

    Returns:
        function: Returns the corresponding preprocessing method.
    """
    if name=="z_norm":
        from data.data_preprocessing import z_norm
        return z_norm
    elif name=="feature_scaling":
        from data.data_preprocessing import feature_scaling
        return feature_scaling

def load_adv_p_dataset(ds_name, UCR_version):
    """
    Returns:
        tuple: The dataset.
        x_train, y_train, x_test, y_test = load_ucr_dataset(ds_name, UCR_version)
    """
    return load_adv_p(ds_name)

def load_ucr_dataset(ds_name, UCR_version):
    """
    Args:
        UCR_version (int): The UCR version

    Returns:
        tuple: The dataset.
        x_train, y_train, x_test, y_test = load_ucr_dataset(ds_name, UCR_version)
    """
    from data.ucr_archive import load_dataset
    return load_dataset(ds_name, UCR_version)

def get_ucr_list_datasets(list_ds_name, UCR_version):
    """
    Args:
        list_ds_name (list): A list containing the dataset names
        UCR_version (int): The UCR version

    Returns:
        dict: A dict containing all datasets.
        x_train, y_train, x_test, y_test = dict[ds_name]
    """
    from ucr_archive import load_dataset
    all_ds = {}
    for ds_name in list_ds_name:
        all_ds[ds_name] = load_dataset(ds_name, UCR_version)
    return all_ds

def get_ucr_all(UCR_version):
    """
    Args:
        UCR_version (int): The UCR version

    Returns:
        dict: A dict containing all datasets.
        x_train, y_train, x_test, y_test = dict[ds_name]
    """
    if UCR_version==2015:
        return get_ucr_list_datasets(UCR_ARCHIVE_2015_DATASETS, UCR_version)
    if UCR_version==2018:
        return get_ucr_list_datasets(UCR_ARCHIVE_2018_DATASETS, UCR_version)

def get_ucr_first_10(UCR_version):
    """
    Args:
        UCR_version (int): The UCR version

    Returns:
        dict: The first 10 datasets of the UCR archive.
        x_train, y_train, x_test, y_test = dict[ds_name]
    """
    if UCR_version==2015:
        return get_ucr_list_datasets(sorted(UCR_ARCHIVE_2015_DATASETS[:10]), UCR_version)
    if UCR_version==2018:
        return get_ucr_list_datasets(sorted(UCR_ARCHIVE_2018_DATASETS[:10]), UCR_version)

def get_ucr_last_10(UCR_version):
    """
    Args:
        UCR_version (int): The UCR version

    Returns:
        dict: The last 10 datasets of the UCR archive.
        x_train, y_train, x_test, y_test = dict[ds_name]
    """
    if UCR_version==2015:
        return get_ucr_list_datasets(sorted(UCR_ARCHIVE_2015_DATASETS[-10:]), UCR_version)
    if UCR_version==2018:
        return get_ucr_list_datasets(sorted(UCR_ARCHIVE_2018_DATASETS[-10:]), UCR_version)
