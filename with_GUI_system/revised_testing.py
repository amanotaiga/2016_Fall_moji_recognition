# -*- coding: utf-8 -*-
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import matplotlib 
matplotlib.use('Agg')
from sklearn.metrics import classification_report,confusion_matrix
from operator import itemgetter
from keras.models import model_from_json
import os

import numpy as np
import scipy.misc
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import tensorflow
from tensorflow.python.ops import control_flow_ops 
from revised_convert_data import data

def testing_result(category):
    tensorflow.python.control_flow_ops = control_flow_ops
    np.set_printoptions(threshold=np.nan)
    X_train, Y_train,nb_classes,input_shape = data()
    n_output = Y_train.shape[1]
    model = Sequential()
    # load json and create model
    if (category==0):
        json_file = open('model_kata_S.json', 'r')
    elif(category==1):
        json_file = open('model_hira.json','r')	
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json) 
    # load weights into new model
    if(category==0):
        loaded_model.load_weights("model_kata_S.h5")
    elif(category==1):
        loaded_model.load_weights("model_hira.h5") 
    print("Loaded model from disk") 
    loaded_model.summary()
    # evaluate loaded model on test data
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    y_pred = loaded_model.predict_classes(X_train)
    print(y_pred)
    if(category==0):
        map_character = ['ア','イ','ウ','エ','オ','カ','キ','ク','ケ','コ','サ','シ','ス','セ','ソ',
        'タ','チ','ツ','テ','ト','ナ','二','ヌ','ネ','ノ','ハ','ヒ','フ','ヘ','ホ',
        'マ','ミ','ム','メ','モ','ヤ','ユ','ヨ','ラ','リ','ル','レ','ロ','ワ','ヲ','ン',
        ' d',' o',' l',' r']
    elif(category==1):
        map_character = ['あ','い','う','え','お','か','き','く','け','こ','さ','し','す','せ','そ',
        'た','ち','つ','て','と','な','に','ぬ','ね','の','は','ひ','ふ','へ','ほ',
        'ま','み','む','め','も','や','ゆ','よ','ら','り','る','れ','ろ','わ','を','ん',
        ' s',' s',' s',' s']
    for i in range(1):
        key_map = y_pred[i]
        print(map_character[key_map]),
		
if __name__ == "__main__":
    testing_result(category)
