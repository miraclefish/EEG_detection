import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BECTdetection import BECTdetect

labeled_data = pd.read_excel("./新标注数据12-18.xlsx", header=0, index_col=0)
nameList = labeled_data["文件名"].values
label = labeled_data["放电指数（按长度数）"].values

thresholds = np.linspace(0.5, 3.2, 135)
GT_dict= {"threshold":[], "th":[], "th_Low":[], "th_Hig":[], "cdf":[], "cdf_Low":[], "cdf_Hig":[]}

# Min = []
# Max = []
for file, gt in zip(nameList, label):
    print("*********************************")
    metrics = []
    best_th = 0
    best_metric = 1
    filepath = "./NewcsvData/" + file[:-4] + ".txt"
    bect = BECTdetect(filepath=filepath, print_log=True)
    # SWI = bect.Custom_Analysis(Spike_width=61, threshold=thresholds[70], template_mode="gamma")
    # Min.append(min(bect.spike_score))
    # Max.append(max(bect.spike_score))

    for th in thresholds:
        SWI, _ = bect.Custom_Analysis(Spike_width=61, threshold=th, template_mode="gamma")
        metrics.append(SWI)
    ind = np.argmin(np.abs(gt - metrics))

    # 保存原始信号
    np.savetxt('./AutoTH/OrigData/'+ file[:-4] + ".txt", np.round(bect.bandPassData, 4), fmt='%.4f')

    # 保存滤后信号
    # np.savetxt('./AutoTH/FiltedData/'+ file[:-4] + ".txt", np.round(bect.score, 4), fmt='%.4f')

    # 保存极值分布特征
    bins = np.linspace(-8,8,51)
    bins[0] = -10
    bins[-1] = 10
    feature = np.zeros((len(bins)-1, 4))
    hist, bins = np.histogram(bect.spike_score, bins=bins, density=True)
    feature[:,0] = hist
    feature[:,1] = bins[:-1]
    feature[:,2] = bins[1:]
    # np.savetxt('./AutoTH/HistFeature/'+ file[:-4] + ".txt", feature, fmt='%.4f')

    # 保存真值相关的所有信息
    Low_ind = len(np.where(metrics>=gt-0.05)[0])-1
    Hig_ind = max(len(np.where(metrics>=gt+0.05)[0])-1,0)
    th = thresholds*np.std(bect.spike_score)+np.mean(bect.spike_score)
    print("Error allowed threshold interval [{:.2f},{:.2f}], length with {:.2f}".format(th[Hig_ind], th[Low_ind], th[Low_ind]-th[Hig_ind]))

    threshold_gt = thresholds[ind]
    th_gt = th[ind]
    th_Low = th[Low_ind]
    th_Hig = th[Hig_ind]
    chosen_mask = np.array(list(map(lambda x: 1 if (th_gt>x[0] and th_gt<x[1]) else 0, [bins_pair for bins_pair in feature[:,1:]])))
    feature[:,3] = chosen_mask
    # np.savetxt('./AutoTH/HistFeature/'+ file[:-4] + ".txt", feature, fmt='%.4f')
    
    N = len(bect.spike_score)
    ranks = np.linspace(min(bect.spike_score), max(bect.spike_score), 1000)
    cdf = np.array([len(np.where(bect.spike_score<=x)[0])/N for x in ranks])
    l_ind = len(np.where(ranks<=th[Low_ind])[0])-1
    h_ind = len(np.where(ranks<=th[Hig_ind])[0])-1
    # print("Error allowed cdf interval [{:.2f},{:.2f}], length with {:.2f}".format(cdf[h_ind], cdf[l_ind], cdf[l_ind]-cdf[h_ind]))
    
    cdf_gt = len(np.where(bect.spike_score <= th[ind])[0])/len(bect.spike_score)
    cdf_Low_gt = cdf[l_ind]
    cdf_Hig_gt = cdf[h_ind]
    
    GT_dict["threshold"].append(np.round(threshold_gt, 4))
    GT_dict["th"].append(np.round(th_gt, 4))
    GT_dict["th_Low"].append(np.round(th_Low, 4))
    GT_dict["th_Hig"].append(np.round(th_Hig, 4))
    GT_dict["cdf"].append(np.round(cdf_gt, 4))
    GT_dict["cdf_Low"].append(np.round(cdf_Low_gt, 4))
    GT_dict["cdf_Hig"].append(np.round(cdf_Hig_gt, 4))

    _, mask_label = bect.Custom_Analysis(Spike_width=61, threshold=threshold_gt, template_mode="gamma")
    np.savetxt('./AutoTH/MaskLabel/'+ file[:-4] + ".txt", mask_label, fmt='%.2f')

    pass

# GT_info = pd.DataFrame(data=GT_dict, index=[name[:-4] for name in nameList])
# GT_info.to_csv("./AutoTH/" + "GT_info.csv", sep="\t")
pass