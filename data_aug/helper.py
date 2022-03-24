from enum import Enum
import math
import random

from utils.utils import split_list_in_groups

from tensorflow.keras.utils import Sequence

from data_aug.augmentation import *

AUG_METHODS = {
    "jitter":   jitter,
    "scaling":  scaling,
    "rotation": rotation,
    "permutation":  permutation,
    "magwarp":      magnitude_warp,
    "timewarp":     time_warp,
    "windowslice":  window_slice,
    "windowwarp":   window_warp,
    "spawner":  spawner,
    "rgw":      random_guided_warp,
    "rgws":     random_guided_warp_shape,
    "wdba":     wdba,
    "dgw":      discriminative_guided_warp,
    "dgws":     discriminative_guided_warp_shape,
}

def get_aug_by_name(name):
    """
    Returns an augmentation method give it's name.

    Args:
        name (str): The name of the method.

    Raises:
        ValueError: If the

    Returns:
        [type]: [description]
    """
    if name not in AUG_METHODS.keys():
        raise ValueError(
            "The name specified '%s' is not a valid augmentation method.\n\
            Valid methods are: [%s]" % (name, str(AUG_METHODS.keys())))
    return AUG_METHODS[name]

class MultiAugMethod(Enum):
    MULTI  = 1 # Several 
    MIXED  = 2 # Several 


class SequenceDataAugmentation(Sequence):
    def __init__(self, x_train, y_train, batch_size, 
                only_aug_data=False, # If we use only augmented data
                aug_methods=None,    # One or several augmentation methods
                shuffle=True,        # If we shuffle tha data for each epoch
                aug_each_epoch=True, # If we generate new augmented data at each epoch
                multi_aug_method=MultiAugMethod.MULTI
        ):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.aug_each_epoch = aug_each_epoch
        self.multi_aug_method = multi_aug_method
        self.only_aug_data = only_aug_data

        if only_aug_data and aug_methods is None:
            raise ValueError("Can't use only augmented data but no augmentation method!")

        self.aug_methods = aug_methods
        if self.aug_methods is not None:
            # Check if several data aug
            if len(self.aug_methods) > 1:
                self.is_multi_aug = True
            else:
                self.is_multi_aug = False
            # Load aug methods
            self.aug_methods = [get_aug_by_name(aug_name) for aug_name in self.aug_methods]
        
        self.shuffle = shuffle
        self.__augment_data()

    def __augment_data(self):

        if self.aug_methods is None:
            # No augmentation is needed
            self.x_train_aug = self.x_train
            self.y_train_aug = self.y_train
            return

        if not self.is_multi_aug:
            # Using only one aug method
            aug_method = self.aug_methods[0]
            aug_data = aug_method(self.x_train, self.y_train)
            if self.only_aug_data:
                self.x_train_aug = aug_data
                self.y_train_aug = self.y_train.copy()
            else:
                self.x_train_aug = np.concatenate((self.x_train.copy(), aug_data), axis=0)
                self.y_train_aug = np.concatenate((self.y_train.copy(), self.y_train), axis=0)

        elif self.multi_aug_method == MultiAugMethod.MIXED:
            # Augment data: Mixing randomly each aug method

            # - split indices in n groups, with n number of data augs
            aug_data = np.zeros(self.x_train.shape)
            list_idx = list(range(len(aug_data)))
            random.shuffle(list_idx)    # Shuffle list
            groups_idx = split_list_in_groups(list_idx, len(self.aug_methods))
            # - Generate data aug for each group
            for group_idx, aug_method in zip(groups_idx, self.aug_methods):
                aug_data[group_idx] = aug_method(self.x_train[group_idx], self.y_train[group_idx])

            if self.only_aug_data:
                self.x_train_aug = aug_data
                self.y_train_aug = self.y_train.copy()
            else:
                self.x_train_aug = np.concatenate((self.x_train.copy(), aug_data), axis=0)
                self.y_train_aug = np.concatenate((self.y_train.copy(), self.y_train), axis=0)

        elif self.multi_aug_method == MultiAugMethod.MULTI:
            # Append each data aug after another
            if self.only_aug_data:
                self.x_train_aug = np.array()
                self.y_train_aug = np.array()
            else:
                self.x_train_aug = self.x_train.copy()
                self.y_train_aug = self.y_train.copy()
            # - Generate data aug for each method
            for aug_method in self.aug_methods:
                aug_data = aug_method(self.x_train, self.y_train)
                self.x_train_aug = np.concatenate((self.x_train_aug, aug_data), axis=0)
                self.y_train_aug = np.concatenate((self.y_train_aug, self.y_train), axis=0)
        else:
            raise ValueError("Not implemented!")
        
    def __len__(self):
        return math.ceil(len(self.x_train_aug) / self.batch_size)

    def __getitem__(self, idx):
        # Get batch by index
        batch_x = self.x_train_aug[idx*self.batch_size : (idx + 1)*self.batch_size]
        batch_y = self.y_train_aug[idx*self.batch_size : (idx + 1)*self.batch_size]
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.aug_methods is None:
            return
        # Augment data
        if self.aug_each_epoch:
            self.__augment_data()

        # Shuffle data
        if self.shuffle:
            p = np.random.permutation(len(self.x_train_aug))
            self.x_train_aug = self.x_train_aug[p]
            self.y_train_aug = self.y_train_aug[p]
