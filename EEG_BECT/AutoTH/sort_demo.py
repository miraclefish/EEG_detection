import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt


def band_pass_filter(LowHz, HigHz, data):
    data = np.squeeze(data)
    hf = HigHz * 2.0 / 1000
    lf = LowHz * 2.0 / 1000
    N = 2
    b, a = butter(N, [lf, hf], "bandpass")
    filted_data = filtfilt(b, a, data)
    filted_data = filted_data.reshape(-1, 1)
    return filted_data


if __name__ == "__main__":
    Datapath = './Origdata/train'
    fileList = os.listdir(Datapath)
    for file in fileList:
        path = os.path.join(Datapath, file)
        data = np.loadtxt(path, skiprows=1)
        bandPassData = band_pass_filter(LowHz=0.5, HigHz=40, data=data)
        np.savetxt(os.path.join('./KKData/train', file), bandPassData, fmt='%.4f')
        pass
    pass
