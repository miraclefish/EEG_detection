from BECTdetection import BECTdetect
import os
import pandas as pd

path = './NewcsvData'
filelist = os.listdir(path)
indicators = []
for i, filename in enumerate(filelist):
    filepath = os.path.join(path, filename)
    bect = BECTdetect(filepath = filepath, S_size=61, threshold=2, print_log=False)
    indicators.append(bect.indicator)
    pass

result = pd.DataFrame({'Name':filelist, 'indicator':indicators})
result.to_csv('测试结果.csv')