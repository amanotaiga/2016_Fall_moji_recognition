import struct
import numpy as np
import h5py
from keras.utils import np_utils
import tensorflow
import scipy.misc
from keras import backend as K
from sklearn.model_selection import train_test_split
#from load_hira_all import read_hiragana
from load_ETL78 import read_hiragana
#from load_kata_ETL1 import read_katakana
def data():
    Test_size=0.2
    img_print = False
    nb_classes = 46
    writers = 360
    img_rows,img_cols = 32 , 32
    chars  = read_hiragana()
    #chars = read_katakana()
    characters = chars
    #count the numbers of label and rename from 0 to label-1
    y_train = np.repeat(np.arange(nb_classes), writers)
    x_train, x_test, y_train, y_test = train_test_split(characters, y_train, test_size=Test_size)
    if K.image_dim_ordering() == 'th':
        X_train = x_train.reshape(x_train.shape[0], 1,img_rows, img_cols)
        X_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        X_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test,nb_classes,input_shape
