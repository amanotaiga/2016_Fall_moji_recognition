#based on https://github.com/yukoba/CnnJapaneseCharacter/blob/master/src/read_hiragana_file.py

import struct
from PIL import Image, ImageEnhance
import numpy as np
 
def read_record_ETL7C(f):
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iL = iF.convert('P')
    return r + (iL,)

new_img = Image.new('P', (64*32, 63*30))

#writers:200

def store_array():
    for f_num in range(1,2):
       filename = 'ETL7/ETL7LC_{:01d}'.format(f_num)
       with open(filename, 'r') as f:
          f.seek(0 * 2052)
          for i in range(9600):
              r = read_record_ETL7C(f)
              if(i%1==0):
                  new_img.paste(r[-1], (64*(i%32), 64*(i/32)))
                  iE = Image.eval(r[-1], lambda x: 255-x*16)
                  fn = 'ETL7C_{:02d}_{:03d}.png'.format(f_num,i)
                  iE.save('/mnt/d/Desktop/ETL7C_data/'+fn, 'PNG')	

store_array()		   
