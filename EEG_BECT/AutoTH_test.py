import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BECTdetection import BECTdetect

labeled_data = pd.read_excel("./新标注数据.xlsx", header=0, index_col=0)
nameList = labeled_data["文件名"].values
label = labeled_data["放电指数（按长度数）"].values

results = []
peaks = []
bins = np.linspace(-9,9,19)
bins[0] = -20
bins[-1] = 20
for file, gt in zip(nameList, label):
    filepath = "./NewcsvData/" + file[:-4] + ".txt"
    bect = BECTdetect(filepath=filepath, print_log=True)
    score = bect.Adaptive_Decomposition(Spike_width=61, template_mode='gamma')
    peak_ind, peak_score = bect.get_peak_score(score)
    hist, bin_edges = np.histogram(peak_score, bins=bins)
    peaks.append(peak_score)
    # Lo_SWI, Hi_SWI = bect.Auto_Analysis(Spike_width=61, template_mode="gamma")
    # results.append([Lo_SWI, Hi_SWI])
    pass
# results = np.concatenate(results, axis=1)
pass