import struct
import numpy as np
import h5py
from keras.utils import np_utils
import tensorflow
import scipy.misc
from keras import backend as K
from sklearn.model_selection import train_test_split
from preprocessing_beauty import get_testing_data

def data():
    Test_size=0.2
    N_beauty = 30
    beauty = 18 
    word = 20
    img_rows,img_cols = 32 , 32
    chars  = get_testing_data()
    characters = chars
    nb_classes = 2
    #count the numbers of label and rename from 0 to label-1
    y_train = np.append([0]*N_beauty*word,[1]*beauty*word)
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
