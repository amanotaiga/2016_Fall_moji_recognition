import struct
import numpy as np
from PIL import Image, ImageEnhance
import numpy as np
import h5py
import tensorflow
import scipy.misc

def load_origin_img():
    count = 0
    for i in range(48):
        img = Image.open("img/sheet_reorder/{:01d}.jpg".format(i+1))
        size = 24
        row,col = 4,6
        count_n = 0
        for j in range(size/col):
            for k in range(size/row):
                img_crop = img.crop((128*k, 128*j, 128+128*k, 128+128*j))
                fn = 'img/total/img_{:01d}_{:02d}.png'.format(i,count_n+1)
                img_crop.save(fn,'PNG')
                count = count+1   
		count_n = count_n+1


load_origin_img()
