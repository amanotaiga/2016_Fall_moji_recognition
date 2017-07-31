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
#import os

import struct
import numpy as np
import h5py
from keras.utils import np_utils
from sklearn import datasets, metrics, cross_validation
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
from convert_beauty import data


X_train, Y_train, X_test, Y_test,nb_classes,input_shape = data()
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

n_output = Y_train.shape[1]


datagen = ImageDataGenerator(samplewise_center = False,rotation_range=15, zoom_range=0.30)

datagen.fit(X_train)
model = Sequential()

def m6_1():
    model.add(Convolution2D(32, 3, 3,  input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

m6_1()

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16), samples_per_epoch=X_train.shape[0],
                    nb_epoch=600, validation_data=(X_val, Y_val))	



y_pred = model.predict_classes(X_test)
print(y_pred)

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('/mnt/d/Desktop/beauty_acc.png')

plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('/mnt/d/Desktop/beauty_loss.png')

"""
#show confusion matrix and classification report
print(classification_report(np.argmax(Y_test,axis=1), y_pred))
confusion = confusion_matrix(np.argmax(Y_test,axis=1), y_pred)
print(confusion)
"""
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""
#show top 3 items to be classified
sorted_row_idx = np.argsort(confusion, axis=1)[:,confusion.shape[1]-3::]
print(sorted_row_idx)

map_character = ['あ','い','う','え','お','か','き','く','け','こ','さ','し','す','せ','そ',
'た','ち','つ','て','と','な','に','ぬ','ね','の','は','ひ','ふ','へ','ほ',
'ま','み','む','め','も','や','ゆ','よ','ら','り','る','れ','ろ','わ','を','ん',
' s',' s',' s',' s']

for j in range(48):
    print ()
    for i in range(3):
         key_map = sorted_row_idx[j,i]
         print(map_character[key_map]),
"""
#save model and weights
model_json = model.to_json()
with open("model_beauty_2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_beauty_2.h5")
print("Saved model to disk")
