from BECTdetection import BECTdetect
import os

path = './NewcsvData'
filelist = os.listdir(path)
for i, filename in enumerate(filelist):
    filepath = os.path.join(path, filename)
    bect = BECTdetect(filepath = filepath, S_size=61, threshold=2, print_log=False)
    pass