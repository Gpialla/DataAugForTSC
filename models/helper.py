from models.models import *

MODEL_LIST = {
    "vgg":      cnn_vgg, 
    "lstm":     lstm1v0, 
    "lstm1":    lstm1, 
    "lstm2":    lstm2, 
    "blstm1":   blstm1, 
    "blstm2":   blstm2, 
    "lstmfcn":  lstm_fcn, 
    "resnet":   cnn_resnet, 
    "mlp":      mlp4, 
    "lenet":    cnn_lenet, 
    "inception": inception,
    "fcn":      fcn
}


def get_model_by_name(name):
    """
    Returns a Keras model, according to the specified name

    Args:
        name (str): The name of the model. Must be in MODEL_LIST.

    Raises:
        ValueError: If the name is not valid. Should be in MODEL_LIST.

    Returns:
        keras.Model: A Keras model.
    """
    if not name in MODEL_LIST.keys():
        raise ValueError(
            "The model name provided '%s', is not implemented.\n\
            Implemented models are: %s."%(name, str(MODEL_LIST))
        )
    return MODEL_LIST[name]
