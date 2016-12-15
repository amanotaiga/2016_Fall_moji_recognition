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
from load_data_kata import get_ETL_data

def data():
    test_size=0.2
    writers = 1383
    nb_classes = 48
    img_print = False
    for i in range(1,6):
        chars = get_ETL_data(i,img_print)
        if (i == 1):
            #get data X and data Y
            characters = chars
        else:
            characters = np.concatenate((characters, chars), axis=0)
    #count the numbers of label and rename from 0 to label-1
    y_train = np.repeat(np.arange(nb_classes), writers)
    x_train, x_test, y_train, y_test = train_test_split(characters, y_train, test_size=test_size)
    if K.image_dim_ordering() == 'th':
        X_train = x_train.reshape(x_train.shape[0], 1, 32, 32)
        X_test = x_test.reshape(x_test.shape[0], 1, 32, 32)
        input_shape = (1, 32, 32)
    else:
        X_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
        X_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
        input_shape = (32, 32, 1)
    # convert class vectors to binary class matrices
    #nb_classes = len(unique_labels)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test,nb_classes,input_shape
