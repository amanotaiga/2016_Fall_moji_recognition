import struct
import numpy as np
from PIL import Image, ImageEnhance
import numpy as np
import h5py
from keras.utils import np_utils
from sklearn import datasets, metrics, cross_validation
from sklearn.utils import shuffle
from keras.models import Sequential
import tensorflow
from tensorflow.python.ops import control_flow_ops 
tensorflow.python.control_flow_ops = control_flow_ops
import scipy.misc
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from load_dataETL1C_1 import get_ETL_data

def data():
    test_size=0.2
    img_print = False
    for i in range(1, 2):
        chars, labs = get_ETL_data(i,img_print)
        if (i == 1):
            #get data X and data Y
            characters = chars
            labels = labs
        else:
            characters = np.concatenate((characters, chars), axis=0)
            labels = np.concatenate((labels, labs), axis=0)
    #count the numbers of label and rename from 0 to label-1
    unique_labels = list(set(labels))
    labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
    new_labels = np.array([labels_dict[l] for l in labels], dtype=np.int32)
    characters_shuffle, new_labels_shuffle = shuffle(
        characters, new_labels, random_state=0)
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(characters_shuffle,
                                                                         new_labels_shuffle,
                                                                         test_size=0.2,
                                                                         random_state=15)
    #X_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    #X_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))
    #input_shape = (1, 64, 64)
    #X_train = x_train.reshape(x_train.shape[0],64,64,1)
    #X_test = x_test.reshape(x_test.shape[0],64,64,1)
    if K.image_dim_ordering() == 'th':
        X_train = x_train.reshape(x_train.shape[0], 1, 64, 64)
        X_test = x_test.reshape(x_test.shape[0], 1, 64, 64)
        input_shape = (1, 64, 64)
    else:
        X_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
        X_test = x_test.reshape(x_test.shape[0], 64, 64, 1)
        input_shape = (64, 64, 1)
    # convert class vectors to binary class matrices
    nb_classes = len(unique_labels)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test,nb_classes,input_shape