import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
from sklearn.metrics import classification_report,confusion_matrix
from operator import itemgetter

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

nb_classes = 48
writer = 200
# input image dimensions
img_rows, img_cols = 32, 32
# img_rows, img_cols = 127, 128

#hiragana.npz
ary = np.load("ETL7C_hira.npz")['arr_0'].reshape([-1, 63, 64]).astype(np.float32) / 15
X_train = np.zeros([nb_classes * writer, img_rows, img_cols], dtype=np.float32)
for i in range(nb_classes * writer):
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
    # X_train[i] = ary[i]
Y_train = np.repeat(np.arange(nb_classes), writer)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
Y_val  = np_utils.to_categorical(Y_val, nb_classes)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

model = Sequential()


def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.1, name=name)


# Best val_loss: 0.0205 - val_acc: 0.9978 (just tried only once)
# 30 minutes on Amazon EC2 g2.2xlarge (NVIDIA GRID K520)
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


def classic_neural():
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

m6_1()
# classic_neural()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16), samples_per_epoch=X_train.shape[0],
                     nb_epoch=30, validation_data=(X_test, Y_test))

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('acc.png')

plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('loss.png')

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


#num = -3  
#top = np.argpartition(confusion, num, axis=1)[:, :num]
#print(top)
#confusion[np.arange(confusion.shape[0])[:, None], top]
#print(confusion)

#nl = []
#for i in confusion:
#    nl.append(sorted(i,reverse=True))
#print(np.matrix(nl))
