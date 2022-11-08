# Imports
from data.ucr_archive   import load_dataset as load_ucr
from data.ucr_archive   import UCR_ARCHIVE_2015_DATASETS, UCR_ARCHIVE_2018_DATASETS, UCR_VERSIONS
from data.uea_archive   import load_dataset as load_uea
from data.uea_archive   import UEA_ARCHIVE_2018_DATASETS
from data.digitsRTD     import load_dataset as load_digits_dataset
from data.adv_p_dataset import load_dataset as load_adv_p


PREPROCESSINGS_NAMES = ["z_norm", "feature_scaling"]

def load_ds_from__archive(archive_name, ds_name, ds_version):
    if archive_name == "UCR":
        return load_ucr_dataset(ds_name, int(ds_version))
    elif archive_name == "adv_p":
        return load_adv_p_dataset(ds_name, ds_version)

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

def load_adv_p_dataset(ds_name, DS_version):
    """
    Returns:
        tuple: The dataset.
        x_train, y_train, x_test, y_test = load_ucr_dataset(ds_name, UCR_version)
    """
    from data.adv_p_dataset import load_dataset as load_adv_p
    return load_adv_p(ds_name, DS_version)

def load_ucr_dataset(ds_name, UCR_version):
    """
    Args:
        UCR_version (int): The UCR version

    Returns:
        tuple: The dataset.
        x_train, y_train, x_test, y_test = load_ucr_dataset(ds_name, UCR_version)
    """
    return load_ucr(ds_name, UCR_version)

def load_digits_dataset():
    """
    Returns:
        tuple: The dataset.
        x_train, y_train, x_test, y_test = load_ucr_dataset(ds_name, UCR_version)
    """
    return load_digits_dataset()

def load_adv_p_dataset(ds_name, UCR_version):
    """
    Returns:
        tuple: The dataset.
        x_train, y_train, x_test, y_test = load_ucr_dataset(ds_name, UCR_version)
    """
    return load_adv_p(ds_name)

def load_uea_dataset(ds_name):
    """
    Args:
        ds_name (str): The name of the dataset

    Returns:
        tuple: The dataset.
        x_train, y_train, x_test, y_test = load_uea(ds_name)
    """
    return load_uea(ds_name)

def get_ucr_list_datasets(list_ds_name, UCR_version):
    """
    Args:
        list_ds_name (list): A list containing the dataset names
        UCR_version (int): The UCR version

    Returns:
        dict: A dict containing all datasets.
        x_train, y_train, x_test, y_test = dict[ds_name]
    """
    all_ds = {}
    for ds_name in list_ds_name:
        all_ds[ds_name] = load_ucr(ds_name, UCR_version)
    return all_ds

def get_uea_list_datasets(list_ds_name):
    """
    Args:
        list_ds_name (list): A list containing the dataset names

    Returns:
        dict: A dict containing all datasets.
        x_train, y_train, x_test, y_test = dict[ds_name]
    """
    all_ds = {}
    for ds_name in list_ds_name:
        all_ds[ds_name] = load_uea(ds_name)
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

def get_uea_all():
    """
    Returns:
        dict: A dict containing all datasets.
        x_train, y_train, x_test, y_test = dict[ds_name]
    """
    return get_uea_list_datasets(UEA_ARCHIVE_2018_DATASETS)

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
