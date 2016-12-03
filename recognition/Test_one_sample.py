# -*- coding: utf-8 -*-
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
from sklearn.metrics import classification_report,confusion_matrix
from operator import itemgetter
from keras.models import model_from_json
import os

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


np.set_printoptions(threshold=np.nan)

nb_classes = 1
writer = 1
# input image dimensions
img_rows, img_cols = 32, 32
# img_rows, img_cols = 127, 128

ary = np.load("Test.npz")['arr_0'].reshape([-1, 63, 64]).astype(np.float32) / 15
X_train = np.zeros([nb_classes * writer, img_rows, img_cols], dtype=np.float32)

for i in range(nb_classes * writer):
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
	
Y_train = np.repeat(np.arange(nb_classes), writer)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_train = Y_train.reshape((-1, 1))
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
loaded_model.summary()
# evaluate loaded model on test data
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

cbks = []

#history = loaded_model.fit(X_train, Y_train, batch_size=1, nb_epoch=1, verbose=1, callbacks=cbks, validation_data=(X_val, Y_val))
   
score = loaded_model.evaluate(X_train, Y_train, verbose=1)
print('Test accuracy:', score[1])

y_pred = loaded_model.predict_classes(X_train)
print(y_pred)
