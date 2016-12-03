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
from keras import initializations
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from convert_data_kata import data


X_train, Y_train, X_test, Y_test,nb_classes,input_shape = data()
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25)

n_output = Y_train.shape[1]


datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)
model = Sequential()


def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.1, name=name)


def m6_1():
    model.add(Convolution2D(32, 3, 3, init=my_init, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, init=my_init))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

def m6_2():
    model.add(Convolution2D(32, 3, 3, init=my_init, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, init=my_init))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

m6_1()

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16), samples_per_epoch=X_train.shape[0],
                    nb_epoch=13, validation_data=(X_val, Y_val))	


print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('acc_kata_M1.png')

plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('loss_kata_M1.png')

y_pred = model.predict_classes(X_test)
#print(y_pred)

p=model.predict_proba(X_test) # to predict probability

print(classification_report(np.argmax(Y_test,axis=1), y_pred))
confusion = confusion_matrix(np.argmax(Y_test,axis=1), y_pred)
print(confusion)

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

sorted_row_idx = np.argsort(confusion, axis=1)[:,confusion.shape[1]-3::]
print(sorted_row_idx)

map_character = ['ア','イ','ウ','エ','オ','カ','キ','ク','ケ','コ','サ','シ','ス','セ','ソ',
'タ','チ','ツ','テ','ト','ナ','二','ヌ','ネ','ノ','ハ','ヒ','フ','ヘ','ホ',
'マ','ミ','ム','メ','モ','ヤ','ユ','ヨ','ラ','リ','ル','レ','ロ','ワ','ヲ','ン',
' d',' o',' l',' r']

for j in range(48):
    print ()
    for i in range(3):
         key_map = sorted_row_idx[j,i]
         print(map_character[key_map]),

model_json = model.to_json()
with open("model_kata.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_kata.h5")
print("Saved model to disk")
