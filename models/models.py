# Imports 
import math
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Model

class Classifier_INCEPTION:
    def __init__(self, input_shape, nb_classes, nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41):

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = 32

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = layers.Concatenate(axis=2)(conv_list)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = layers.BatchNormalization()(shortcut_y)

        x = layers.Add()([shortcut_y, out_tensor])
        x = layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = layers.GlobalAveragePooling1D()(x)

        output_layer = layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        return model


def inception(input_shape, nb_class):
    clsf = Classifier_INCEPTION(input_shape, nb_class)
    model = clsf.build_model(input_shape, nb_class)
    model.summary()
    return model
    

def mlp4(input_shape, nb_class):
    # Z. Wang, W. Yan, T. Oates, "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline," Int. Joint Conf. Neural Networks, 2017, pp. 1578-1585
    
    ip = layers.Input(shape=input_shape)
    fc = layers.Flatten()(ip)
    
    fc = layers.Dropout(0.1)(fc)
            
    fc = layers.Dense(500, activation='relu')(fc)
    fc = layers.Dropout(0.2)(fc)
    
    fc = layers.Dense(500, activation='relu')(fc)
    fc = layers.Dropout(0.2)(fc)
    
    fc = layers.Dense(500, activation='relu')(fc)
    fc = layers.Dropout(0.3)(fc)
    
    out = layers.Dense(nb_class, activation='softmax')(fc)
    
    model = Model([ip], [out])
    model.summary()
    return model

def cnn_lenet(input_shape, nb_class):
    # Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
    
    ip = layers.Input(shape=input_shape)
    
    conv = ip
    
    nb_cnn = int(round(math.log(input_shape[0], 2))-3)
    print("pooling layers: %d"%nb_cnn)
    
    for i in range(nb_cnn):
        conv = layers.Conv1D(6+10*i, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = layers.MaxPooling1D(pool_size=2)(conv)
        
    flat = layers.Flatten()(conv)
    
    fc = layers.Dense(120, activation='relu')(flat)
    fc = layers.Dropout(0.5)(fc)
    
    fc = layers.Dense(84, activation='relu')(fc)
    fc = layers.Dropout(0.5)(fc)
    
    out = layers.Dense(nb_class, activation='softmax')(fc)
    
    model = Model([ip], [out])
    model.summary()
    return model


def cnn_vgg(input_shape, nb_class):
    # K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.
    
    ip = layers.Input(shape=input_shape)
    
    conv = ip
    
    nb_cnn = int(round(math.log(input_shape[0], 2))-3)
    print("pooling layers: %d"%nb_cnn)
    
    for i in range(nb_cnn):
        num_filters = min(64*2**i, 512)
        conv = layers.Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = layers.Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        if i > 1:
            conv = layers.Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = layers.MaxPooling1D(pool_size=2)(conv)
        
    flat = layers.Flatten()(conv)
    
    fc = layers.Dense(4096, activation='relu')(flat)
    fc = layers.Dropout(0.5)(fc)
    
    fc = layers.Dense(4096, activation='relu')(fc)
    fc = layers.Dropout(0.5)(fc)
    
    out = layers.Dense(nb_class, activation='softmax')(fc)
    
    model = Model([ip], [out])
    model.summary()
    return model

def lstm1v0(input_shape, nb_class):
    # Original proposal:
    # S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.
        
    ip = layers.Input(shape=input_shape)

    l2 = layers.LSTM(512)(ip)
    out = layers.Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model

def lstm1(input_shape, nb_class):
    # Original proposal:
    # S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.
    
    # Hyperparameter choices: 
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017
    
    ip = layers.Input(shape=input_shape)

    l2 = layers.LSTM(100)(ip)
    out = layers.Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def lstm2(input_shape, nb_class):
    ip = layers.Input(shape=input_shape)

    l1 = layers.LSTM(100, return_sequences=True)(ip)
    l2 = layers.LSTM(100)(l1)
    out = layers.Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def blstm1(input_shape, nb_class):
    # Original proposal:
    # M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Transactions on Signal Processing, vol. 45, no. 11, pp. 2673–2681, 1997.
    
    # Hyperparameter choices: 
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017
    ip = layers.Input(shape=input_shape)

    l2 = layers.Bidirectional(layers.LSTM(100))(ip)
    out = layers.Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model

def blstm2(input_shape, nb_class):
    ip = layers.Input(shape=input_shape)

    l1 = layers.Bidirectional(layers.LSTM(100, return_sequences=True))(ip)
    l2 = layers.Bidirectional(layers.LSTM(100))(l1)
    out = layers.Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model

def lstm_fcn(input_shape, nb_class):
    # F. Karim, S. Majumdar, H. Darabi, and S. Chen, “LSTM Fully Convolutional Networks for Time Series Classification,” IEEE Access, vol. 6, pp. 1662–1669, 2018.

    ip = layers.Input(shape=input_shape)
    
    # lstm part is a 1 time step multivariate as described in Karim et al. Seems strange, but works I guess.
    lstm = layers.Permute((2, 1))(ip)

    lstm = layers.LSTM(128)(lstm)
    lstm = layers.Dropout(0.8)(lstm)

    conv = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)

    flat = layers.GlobalAveragePooling1D()(conv)

    flat = layers.Concatenate([lstm, flat])

    out = layers.Dense(nb_class, activation='softmax')(flat)

    model = Model([ip], [out])

    model.summary()

    return model


def cnn_resnet(input_shape, nb_class):
    # I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, P-A Muller, "Data augmentation using synthetic data for time series classification with deep residual networks," International Workshop on Advanced Analytics and Learning on Temporal Data ECML/PKDD, 2018

    ip = layers.Input(shape=input_shape)
    residual = ip
    conv = ip
    
    for i, nb_nodes in enumerate([64, 128, 128]):
        conv = layers.Conv1D(nb_nodes, 8, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)

        conv = layers.Conv1D(nb_nodes, 5, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)

        conv = layers.Conv1D(nb_nodes, 3, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)

        if i < 2:
            # expands dimensions according to Fawaz et al.
            residual = layers.Conv1D(nb_nodes, 1, padding='same', kernel_initializer="glorot_uniform")(residual)
        residual = layers.BatchNormalization()(residual)
        conv = layers.Add([residual, conv])
        conv = layers.Activation('relu')(conv)
        
        residual = conv
    
    flat = layers.GlobalAveragePooling1D()(conv)

    out = layers.Dense(nb_class, activation='softmax')(flat)

    model = Model([ip], [out])

    model.summary()

    return model
