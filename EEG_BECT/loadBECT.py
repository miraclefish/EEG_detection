from EDFreader import EDFreader
import numpy as np
import os

path = 'E:/data/NewEDFData'
filelist = os.listdir(path)
for i, filename in enumerate(filelist):
    print(i+1, ':', filename)
    filepath = os.path.join(path, filename)

    edfReader = EDFreader(filepath)
    edfReader.save_txt()
