import struct
import numpy as np
from PIL import Image, ImageEnhance
import numpy as np
import h5py
import tensorflow
import scipy.misc


def read_record_ETL7C(f):
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iL = iF.convert('P')
    enhancer = ImageEnhance.Brightness(iL)
    iT = enhancer.enhance(20)
    size_add = 12
    origin_x = 64
    origin_y = 63
    iT = iT.resize((origin_x + size_add, origin_y+ size_add))
    iT = iT.crop((size_add / 2,
                  size_add / 2,
                  origin_x + size_add / 2,
                  origin_y + size_add / 2))
    return r + (iT,)


def get_ETL_data(filenum,img_print):
        new_img = Image.new('1', (64, 64))
        X = []
        Y = []
        filename = 'ETL7/ETL7LC_{:01d}'.format(filenum)
        with open(filename, 'rb') as f:
            set_range = 48
            writers = 200
            for j in range(set_range):
                for i in range(writers):
                    r = read_record_ETL7C(f)
                    new_img.paste(r[-1], (0, 0))
                    iE = Image.eval(new_img, lambda x: x)
                    iE.thumbnail((32,32))
                    if(img_print==True):
                        if(i%100==0):
                           fn = 'ETL7C_ds{:02d}_{:02d} {:03d}.png'.format(filenum,j,i)
                           iE.save('/mnt/d/Desktop/ETL7C_data/'+fn, 'PNG')
                    shapes = 32, 32
		    #put the image into an array
                    outData = np.asarray(iE.getdata()).reshape(shapes[0], shapes[1])
                    X.append(outData)
        X = np.asarray(X, dtype=np.int32)
	#return array X
        return X
