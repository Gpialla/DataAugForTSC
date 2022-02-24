import math

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

class SequenceDataAugmentation(Sequence):
    def __init__(self, x_train, y_train, batch_size, aug_method=None, shuffle=True):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size

        self.aug_method = aug_method
        print(self.aug_method)
        if self.aug_method is not None:
            # Load aug method
            self.aug_method = get_aug_by_name(self.aug_method)
        self.shuffle = shuffle

        self.__augment_data()

    def __augment_data(self):

        if self.aug_method is None:
            # No augmentation is needed
            self.x_train_aug = self.x_train
            self.y_train_aug = self.y_train
            return

        # Augment data
        self.x_train_aug = np.concatenate((self.x_train, self.aug_method(self.x_train, self.y_train)), axis=0)
        self.y_train_aug = np.concatenate((self.y_train, self.y_train), axis=0)

        if self.shuffle:
            # Shuffle data
            p = np.random.permutation(len(self.x_train_aug))
            self.x_train_aug = self.x_train_aug[p]
            self.y_train_aug = self.y_train_aug[p]

    def __len__(self):
        return math.ceil(len(self.x_train_aug) / self.batch_size)

    def __getitem__(self, idx):
        # Get batch by index
        batch_x = self.x_train_aug[idx*self.batch_size : (idx + 1)*self.batch_size]
        batch_y = self.y_train_aug[idx*self.batch_size : (idx + 1)*self.batch_size]
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.aug_method is None:
            return
        # Augment data
        self.__augment_data()


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
