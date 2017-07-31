import struct
import numpy as np
from PIL import Image,ImageEnhance

	
def read_hiragana():
    X = []
    for j in range(46):
        moji = 0
        for i in range(360):
            iF = Image.open("/mnt/d/Desktop/ETL78/{:01d}_{:03d}.png".format(j+1,i+1))
            #iF = iF.resize((128,127+1))
            iF.thumbnail((32,32))
            shapes = 32, 32
		    #put the image into an array
            outData = np.asarray(iF.getdata()).reshape(shapes[0], shapes[1])
            X.append(outData)   
    X = np.asarray(X, dtype=np.int32)
    #return array X 
    return X	
	
read_hiragana()