import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BECTdetection import BECTdetect

labeled_data = pd.read_excel("./新标注数据12-18.xlsx", header=0, index_col=0)
nameList = labeled_data["文件名"].values
label = labeled_data["放电指数（按长度数）"].values

thresholds = np.linspace(0.5, 3.2, 135)
Min = []
Max = []
plt.figure(figsize=[15,5])
for file, gt in zip(nameList, label):
    print("*********************************")
    metrics = []
    best_th = 0
    best_metric = 1
    filepath = "./NewcsvData/" + file[:-4] + ".txt"
    bect = BECTdetect(filepath=filepath, print_log=True)
    for th in thresholds:
        SWI = bect.Custom_Analysis(Spike_width=61, threshold=th, template_mode="gamma")
        metrics.append(SWI)
    ind = np.argmin(np.abs(gt - metrics))

    Min.append(min(bect.spike_score))
    Max.append(max(bect.spike_score))
    print("Min Spike Score: ", Min[-1])
    print("Max Spike Score: ", Max[-1])
    
    plt.subplot(1, 3, 1)
    th = thresholds*np.std(bect.spike_score)+np.mean(bect.spike_score)
    Low_ind = len(np.where(metrics>=gt-0.05)[0])-1
    Hig_ind = max(len(np.where(metrics>=gt+0.05)[0])-1,0)
    print("Error allowed threshold interval [{:.2f},{:.2f}], length with {:.2f}".format(th[Hig_ind], th[Low_ind], th[Low_ind]-th[Hig_ind]))
    plt.scatter(th, metrics)
    plt.plot(th[ind], metrics[ind], 'ro')
    plt.axhline(gt, c='r')
    plt.axvline(th[ind], c='r')
    plt.axvline(th[Low_ind], c='g')
    plt.axvline(th[Hig_ind], c='g')

    plt.subplot(1, 3, 2)
    n, bins, patches = plt.hist(bect.spike_score, bins=25, rwidth=0.9, density=True)
    mu = np.mean(bect.spike_score)
    sigma = np.std(bect.spike_score)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    plt.plot(bins, y, '--')
    plt.axvline(th[ind], c='r', lineWidth=2)
    

    plt.subplot(1, 3, 3)
    N = len(bect.spike_score)
    ranks = np.linspace(min(bect.spike_score), max(bect.spike_score), 1000)
    cdf = np.array([len(np.where(bect.spike_score<=x)[0])/N for x in ranks])
    # cdf = [len(np.where(bect.spike_score <= bins[i])[0])/len(bect.spike_score) for i in range(len(bins))]
    l_ind = len(np.where(ranks<=th[Low_ind])[0])-1
    h_ind = len(np.where(ranks<=th[Hig_ind])[0])-1
    print("Error allowed cdf interval [{:.2f},{:.2f}], length with {:.2f}".format(cdf[h_ind], cdf[l_ind], cdf[l_ind]-cdf[h_ind]))
    plt.plot(ranks, cdf)
    plt.axvline(th[ind], c='r', lineWidth=2)
    plt.axhline(cdf[l_ind], c='g', lineWidth=2)
    plt.axhline(cdf[h_ind], c='g', lineWidth=2)

    prob_th = len(np.where(bect.spike_score <= th[ind])[0])/len(bect.spike_score)
    plt.suptitle("<{}> best_th = {:.2f} || gt = {:.2f}% || prob_th = {:.2f}".format(file[:-4], thresholds[ind], gt*100, prob_th))
    plt.show()
    # plt.savefig('./th_fig/' + file[:-4] + '.png')
    plt.clf()
    pass

pass