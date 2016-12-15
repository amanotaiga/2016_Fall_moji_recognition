import struct
import numpy as np
from PIL import Image, ImageEnhance
import numpy as np
import h5py
from sklearn import datasets, metrics
from keras.utils import np_utils
import tensorflow
from tensorflow.python.ops import control_flow_ops 
tensorflow.python.control_flow_ops = control_flow_ops
import scipy.misc
from keras import backend as K
from revised_load_data import get_testing_data

def data():
    writers = 1
    nb_classes = 1
    #count the numbers of label and rename from 0 to label-1
    y_train = np.repeat(np.arange(nb_classes), writers)
    x_train = get_testing_data()
    if K.image_dim_ordering() == 'th':
        X_train = x_train.reshape(x_train.shape[0], 1, 32, 32)
        input_shape = (1, 32, 32)
    else:
        X_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
        input_shape = (32, 32, 1)
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train,nb_classes,input_shape
