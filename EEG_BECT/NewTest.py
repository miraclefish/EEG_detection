import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from BECTdetection import BECTdetect

path = './csvData'
filelist = os.listdir(path)
indicators = []
# for i, filename in enumerate(filelist):
#     filepath = os.path.join(path, filename)
#     bect = BECTdetect(filepath = filepath, S_size=61, threshold=2, print_log=False)
#     indicators.append(bect.indicator)
#     pass
threshold = 1.5
filepath = os.path.join(path, filelist[13])
bect = BECTdetect(filepath = filepath, S_size=61, threshold=threshold, print_log=False)

plt.hist(bect.peak_score, bins=25)
plt.axvline(np.std(bect.peak_score)*threshold)
plt.show()

pass

for i in range(int(len(bect.data)/10000)):
    fig = bect.plot_result([i*10000, (i+1)*10000])
    # fig.savefig('./figsave1/{}.png'.format(i+1))
    pass

# result = pd.DataFrame({'Name':filelist, 'indicator':indicators})
# result.to_csv('测试结果.csv')
