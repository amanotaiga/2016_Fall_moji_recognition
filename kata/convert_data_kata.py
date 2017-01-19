import numpy as np
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from load_data_kata import get_ETL_data

def data():
    test_size=0.2
    writers = 1383
    nb_classes = 48
    img_rows,img_cols = 32 , 32
    img_print = False
    for i in range(1,6):
        chars = get_ETL_data(i,img_print)
        if (i == 1):
            #get data X 
            characters = chars
        else:
            characters = np.concatenate((characters, chars), axis=0)
    #count the numbers of label and rename from 0 to label-1
    y_train = np.repeat(np.arange(nb_classes), writers)
    x_train, x_test, y_train, y_test = train_test_split(characters, y_train, test_size=test_size)
    if K.image_dim_ordering() == 'th':
        X_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
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
