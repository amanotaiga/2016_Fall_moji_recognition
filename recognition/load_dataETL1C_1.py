import struct
import numpy as np
from PIL import Image, ImageEnhance
import numpy as np
import h5py
import tensorflow
import scipy.misc


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


def get_ETL_data(filenum,img_print):
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
                    if(img_print==True):
                        if(i%100==0):
                           fn = 'ETL1C_ds{:02d}  {:02d} {:03d}.png'.format(filenum,j,i)
                           iE.save('ETL1C_01_new_data/'+fn, 'PNG')
                    shapes = 64, 64
					#put the image into an array
                    outData = np.asarray(iE.getdata()).reshape(shapes[0], shapes[1])
                    X.append(outData)	
					#append the label of the image
                    Y.append(r[3])
        X, Y = np.asarray(X, dtype=np.int32), np.asarray(Y, dtype=np.int32)
		#return array X and array Y
        return (X,Y)
