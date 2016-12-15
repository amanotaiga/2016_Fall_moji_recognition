import struct
import numpy as np
from PIL import Image, ImageEnhance
import numpy as np
import h5py
import tensorflow
import scipy.misc

def read_record_ETL6C(f):
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iL = iF.convert('P')
    enhancer = ImageEnhance.Brightness(iL)
    iT = enhancer.enhance(20)
    outer = 12
    iT = iT.resize((64 + outer, 63+ outer))
    iT = iT.crop((outer/2,
                  outer/2,
                  64 + outer/2,
                  63 + outer/2))
    return r + (iT,)


def get_ETL_data(filenum,img_print):
        new_img = Image.new('1', (64, 64))
        X = []
        filename = 'ETL6/ETL6C_{:02d}'.format(filenum)
        with open(filename, 'rb') as f:
            set_range = 10
            #8 different characters
            if(filenum==5):
                set_range = 8
            for j in range(set_range):
                for i in range(1383):
                    r = read_record_ETL6C(f)
                    new_img.paste(r[-1], (0, 0))
                    iE = Image.eval(new_img, lambda x: x)
                    #resize image to 32,32
                    iE.thumbnail((32,32))
                    if(img_print==True):
                        if(i%100==0):
                           fn = 'ETL6C_ds{:02d}_{:02d} {:03d}.png'.format(filenum,j,i)
                           iE.save('/mnt/d/Desktop/ETL6C_data/'+fn, 'PNG')
                    shapes = 32, 32
		            #put the image into an array
                    outData = np.asarray(iE.getdata()).reshape(shapes[0], shapes[1])
                    X.append(outData)	
        #append the label of the image
        X = np.asarray(X, dtype=np.int32)
		#return array X
        return X
