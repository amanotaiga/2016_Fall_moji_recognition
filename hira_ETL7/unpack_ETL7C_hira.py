import struct
from PIL import Image, ImageEnhance
import numpy as np
 
def read_record_ETL7C(f):
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iL = iF.convert('P')
    size_add = 12
    iL = iL.resize((64 + size_add, 63+ size_add))
    iL = iL.crop((size_add / 2,
                  size_add / 2,
                  64 + size_add / 2,
                  63 + size_add / 2))
    return r + (iL,)

new_img = Image.new('P', (64*32, 63*30))

def store_array():
    ary = np.zeros([48, 200, 63, 64], dtype=np.uint8)
    for f_num in range(1,2):
       filename = 'ETL7/ETL7LC_{:01d}'.format(f_num)
       with open(filename, 'r') as f:
          f.seek(0 * 2052)
          for j in range(48):
              moji = 0
              for i in range(200):
                 r = read_record_ETL7C(f)
                 ary[(f_num-1)*48+j,moji] = np.array(r[-1])
                 moji += 1
                 #if(i%100==0):
                     #new_img.paste(r[-1], (64*(i%32), 64*(i/32)))
                     #iE = Image.eval(r[-1], lambda x: 255-x*16)
                     #fn = 'ETL7LC_ds{:02d}_{:02d}_{:03d}.png'.format(f_num,j,i)
                     #iE.save('/mnt/d/Desktop/ETL7C_01_data/'+fn, 'PNG')	
    np.savez_compressed("ETL7C_hira.npz", ary)


store_array()		   
