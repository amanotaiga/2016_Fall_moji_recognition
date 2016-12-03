import struct
from PIL import Image, ImageEnhance
import numpy as np
 
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
	
new_img = Image.new('P', (64*32, 63*30))

def store_array():
    #nb_classes,writers,height,weight
    nb_classes = 2
    ary = np.zeros([2, 1, 63, 64], dtype=np.uint8)
    for j in range(nb_classes):
        moji = 0
        for i in range(1):
            iE = read_testing(j+1)
            # store img into array
            ary[j,moji] = np.array(iE)
            moji += 1
            fn = 'ETL6C{:02d}_{:03d}.png'.format(j,i)
            # make img to be more visible
            iE = Image.eval(iE, lambda x: 255-x*16)
            iE.save('/mnt/d/Desktop/ETL6C_data/'+fn, 'PNG')	
    #compress array into npz file
    np.savez_compressed("Test.npz", ary)

store_array()	

