import struct
import numpy as np
from PIL import Image, ImageEnhance
import numpy as np
import h5py
import tensorflow
import scipy.misc


def read_testing(num):
    iF = Image.open("Test_{:01d}.png".format(num))
    iL = iF.convert('P')
    enhancer = ImageEnhance.Brightness(iL)
    iE = enhancer.enhance(40)
    size_add = 12
    iE = iE.resize((64 + size_add, 63 + size_add))
    #make img to be centered
    iE = iE.crop((size_add / 2,
                  size_add / 2,
                  64 + size_add / 2,
                  63 + size_add / 2))
    return iE


def get_testing_data():
        X = []
        for i in range(1):
            new_img = Image.new('1', (64, 64))
            r = read_testing(1)
            new_img.paste(r,(0,0))
            iE = Image.eval(new_img, lambda x: x)
            #resize image to 32,32
            iE.thumbnail((32,32))
            shapes = 32, 32
            #put the image into an array
            outData = np.asarray(iE.getdata()).reshape(shapes[0], shapes[1])
            X.append(outData)	
            #append the label of the image
        X = np.asarray(X, dtype=np.int32)
        #return array X
        return X
