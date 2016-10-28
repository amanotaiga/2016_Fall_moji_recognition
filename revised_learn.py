# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
#https://github.com/yukoba/CnnJapaneseCharacter/blob/master/src/learn.py
import numpy as np
import scipy.misc
from keras import backend as K
from keras import initializations
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.python.ops import control_flow_ops 
tensorflow.python.control_flow_ops = control_flow_ops

nb_classes = 8
writer = 1445
# input image dimensions
img_rows, img_cols = 32, 32
# img_rows, img_cols = 127, 128

ary = np.load("ETL1C_data.npz")['arr_0'].reshape([-1, 63, 64]).astype(np.float32) / 15
X_train = np.zeros([nb_classes * writer, img_rows, img_cols], dtype=np.float32)
for i in range(nb_classes * writer):
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
    # X_train[i] = ary[i]
Y_train = np.repeat(np.arange(nb_classes), writer)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

