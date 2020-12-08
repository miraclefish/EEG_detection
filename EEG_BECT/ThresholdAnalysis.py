import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BECTdetection import BECTdetect

labeled_data = pd.read_excel("./新标注数据.xlsx", header=0, index_col=0)
nameList = labeled_data["文件名"].values
label = labeled_data["放电指数（按长度数）"].values

thresholds = np.linspace(0.8, 3.2, 25)
for file, gt in zip(nameList, label):
    metrics = []
    best_th = 0
    best_metric = 1
    filepath = "./NewcsvData/" + file[:-4] + ".txt"
    bect = BECTdetect(filepath=filepath)
    for th in thresholds:
        SWI = bect.Analysis(Spike_width=61, threshold=th, template_mode="gamma")
        metrics.append(SWI)
    dist = list(np.abs(gt - metrics))
    ind = dist.index(np.min(dist))

    plt.figure(figsize=[15,5])
    

    plt.subplot(1, 3, 1)
    plt.scatter(thresholds, metrics)
    plt.plot(thresholds[ind], metrics[ind], 'ro')
    plt.axhline(gt, c='r')

    plt.subplot(1, 3, 2)
    n, bins, patches = plt.hist(bect.spike_score, bins=25, rwidth=0.9, density=True)
    mu = np.mean(bect.spike_score)
    sigma = np.std(bect.spike_score)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    plt.plot(bins, y, '--')
    plt.axvline(sigma*thresholds[ind], c='r', lineWidth=2)
    

    plt.subplot(1, 3, 3)
    cdf = [len(np.where(bect.spike_score <= bins[i])[0])/len(bect.spike_score) for i in range(len(bins))]
    plt.plot(bins, cdf)
    plt.axvline(sigma*thresholds[ind], c='r', lineWidth=2)

    prob_th = len(np.where(bect.spike_score <= sigma*thresholds[ind])[0])/len(bect.spike_score)
    plt.suptitle("<{}> best_th = {:.1f} || gt = {:.2f}% || prob_th = {:.2f}".format(file[:-4], thresholds[ind], gt*100, prob_th))
    plt.show()
    # plt.savefig('./th_fig/' + file[:-4] + '.png')
    pass

pass