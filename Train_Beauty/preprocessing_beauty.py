import struct
import numpy as np
from PIL import Image, ImageEnhance
import numpy as np
import h5py
import tensorflow
import scipy.misc


def read_testing(id,count):
    iF = Image.open("img/total/img_{:01d}_{:02d}.png".format(id,count+1))
    #iF = Image.open("Test_{:01d}.png".format(id+1,count))
    iL = iF.convert('P')
    enhancer = ImageEnhance.Brightness(iL)
    iE = enhancer.enhance(30)
    size_add = 15
    iE = iE.resize((64 + size_add, 64 + size_add))
    #make img to be centered
    iE = iE.crop((size_add / 2,
                  size_add / 2,
                  64 + size_add / 2,
                  64 + size_add / 2))
    return iE


def get_testing_data():
        X = []
	for i in range(48):
            for j in range(20):
                new_img = Image.new('1', (64, 64))
                r = read_testing(i,j)
                new_img.paste(r,(0,0))
                iE = Image.eval(new_img, lambda x: x)
                #resize image to 32,32
                iE.thumbnail((32,32))
                shapes = 32, 32
                iE.save('checking.png','PNG')
                #put the image into an array
                outData = np.asarray(iE.getdata()).reshape(shapes[0], shapes[1])
                X.append(outData)	
                #append the label of the image
        X = np.asarray(X, dtype=np.int32)
        #return array X
        return X

