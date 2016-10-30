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


sz_record = 2052

def read_record_ETL1C(f):
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iL = iF.convert('P')
    enhancer = ImageEnhance.Brightness(iL)
    iT = enhancer.enhance(40)
    size_add = 12
    iT = iT.resize((64 + size_add, 63+ size_add))
    iT = iT.crop((size_add / 2,
                  size_add / 2,
                  64 + size_add / 2,
                  63 + size_add / 2))
    return r + (iT,)


def get_ETL_data(filenum):
        new_img = Image.new('1', (64, 64))
        X = []
        Y = []
        filename = 'ETL1/ETL1C_{:02d}'.format(filenum)
        with open(filename, 'rb') as f:
            f.seek(0 * 1445 * sz_record)
            #8 different characters 
            for j in range(8):
                # 1445 writers
                for i in range(1445):
                    r = read_record_ETL1C(f)
                    new_img.paste(r[-1], (0,0))
                    iE = Image.eval(new_img, lambda x: not x)
                    #if(i%100==0):
                       #fn = 'ETL1C_ds{:02d}  {:02d} {:03d}.png'.format(filenum,j,i)
                       #iE.save('ETL1C_01_new_data/'+fn, 'PNG')
                    shapes = 64, 64
					#put the image into an array
                    outData = np.asarray(iE.getdata()).reshape(shapes[0], shapes[1])
                    X.append(outData)	
					#append the label of the image
                    Y.append(r[3])
        X, Y = np.asarray(X, dtype=np.int32), np.asarray(Y, dtype=np.int32)
		#return array X and array Y
        return (X,Y)


def data():
    test_size=0.2
    for i in range(1, 2):
        chars, labs = get_ETL_data(i)
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

	
X_train, Y_train, X_test, Y_test,nb_classes,input_shape = data()
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


def classic_neural():
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


m6_1()

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16), samples_per_epoch=X_train.shape[0],
                    nb_epoch=40, validation_data=(X_test, Y_test))	

