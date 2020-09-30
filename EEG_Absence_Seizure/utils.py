from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import scipy.signal as signal


def load_data_with_index(sample_length, strides):
    SZ_times = np.load("SZ_event_times.npy").item()
    filenames = os.listdir("SZ_csv/")
    dd_orginal_data = []
    n_splits = {}
    k = 0
    # filenames = filenames[:2]
    for filename in filenames:
        fileid = filename[:-4]
        path = "SZ_csv/" + filename
        Avdf = pd.read_csv(path, encoding="latin-1", engine="python")
        splits, n_split = split_data(Avdf, split_length=sample_length, steps=strides)
        k = k + n_split
        n_splits[fileid] = n_split
        dd_orginal_data = dd_orginal_data + splits
        print(fileid + " have " + str(n_split) + " splits", "total splits = " + str(k))

    splits_index = {}
    k_start = 0
    k_end = 0
    for fileid, n in n_splits.items():
        k_end = k_end + n
        splits_index[fileid] = (k_start, k_end)
        k_start = k_start + n

    sz_index = {}
    for fileid, times in SZ_times.items():
        box = []
        for time in times:
            box.append((int(time[0] / strides), int(time[1] / strides) + 1))
        sz_index[fileid] = box
    np.save("splits_index.npy", splits_index)
    np.save("sz_index.npy", sz_index)
    return dd_orginal_data, splits_index, sz_index


def split_data(Avdf, split_length=5000, steps=500):
    Data = Avdf.drop(columns="Y")
    Data = Data.diff()
    Data = Data.fillna(0)
    Y = Avdf["Y"]
    N, C = Avdf.shape
    splits = []
    n_splits = int((N - (split_length - steps)) / steps)
    for i in range(n_splits):
        data = Data.values[i * steps : i * steps + split_length, :]
        y = Y.values[i * steps : i * steps + split_length]
        y = y[:, np.newaxis]
        splits.append((data, y))
    return splits, n_splits


def seperate_NP(orig_data):
    T, c = orig_data[0][0].shape
    N_sample = []
    P_sample = []
    for data in orig_data:
        X, y = data
        if y[int(T / 2)] == 0:
            N_sample.append(data)
        elif y[int(T / 2)] == 1:
            P_sample.append(data)
    print("N_sample:", len(N_sample), "; P_sample", len(P_sample))
    return N_sample, P_sample


def segment_X(X, split_length, strides):
    X = X[:, 3]
    difX = np.zeros(X.shape)
    difX[1:] = X[1:] - X[:-1]
    difX[0] = difX[1]

    L = len(X)
    T = int((L - split_length) / strides) + 1
    splitX = np.zeros((T, split_length))
    for i in range(T):
        splitX[i, :] = X[i * strides : i * strides + split_length]
    return splitX


def segment_Y(y, split_length, strides):
    L = len(y)
    T = int((L - split_length) / strides) + 1
    splity = np.zeros(T)
    for i in range(T):
        splity[i] = y[i * strides + int(split_length / 2)]
    return splity


def auto_cor(X):
    meanX = np.mean(X, axis=1)
    stdX = np.std(X, axis=1)
    T, N = np.shape(X)
    cor_X = np.zeros((T, N - 1))
    for i in range(1, N):
        data1 = X[:, : N - i]
        data2 = X[:, i:]
        A = np.multiply((data1.T - meanX), (data2.T - meanX))
        B = np.sum(A.T, axis=1)
        cor_X[:, i - 1] = B / (stdX)
    cor_X = cor_X / N
    return cor_X


def AC_represent(orig_data, split_length, strides):
    AC_data = []
    for data in orig_data:
        X_orig, y_orig = data
        X_seg = segment_X(X_orig, split_length, strides)
        y_seg = segment_Y(y_orig, split_length, strides)
        X_AC = auto_cor(X_seg)
        AC_data.append((X_AC, y_seg))
    return AC_data


def balanceNP(orig_data):
    N_sample, P_sample = seperate_NP(orig_data)
    P_num = len(P_sample)
    N_num = int(P_num / 100) * 100 + 300
    ran_index = np.arange(len(N_sample))
    np.random.shuffle(ran_index)
    index = ran_index[:N_num]
    N_data = []
    P_data = P_sample
    for i in index:
        N_data.append(N_sample[i])
    return N_data, P_data


def prepare_XY(AC_data, shuffle=True):
    X = []
    Y = []
    num = len(AC_data)
    T, c = AC_data[0][0].shape
    for data in AC_data:
        X.append(data[0])
        Y.append(data[1])
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    X = X.reshape(num, T, c)
    Y = Y.reshape(num, T, 1)
    if shuffle:
        index = np.arange(num)
        np.random.shuffle(index)
        X = X[index, :, :]
        Y = Y[index, :, :]
    return X, Y


# ********************************************************************************* #
# Read data Parts
def read_data(root):
    raw = mne.io.read_raw_edf(root, preload=True)
    Event_with_time = gpythonet_event_with_time(raw)
    SZ_time = get_SZ_time(Event_with_time)
    Avdf = original_data(raw)
    N, C = Avdf.shape
    y = np.zeros((N, 1))
    y = insert_ones(y, SZ_time)
    Avdf["Y"] = y

    return Avdf, Event_with_time, SZ_time


def recover_signal(DF):
    df = {}
    Av = (DF.EEGA1Ref + DF.EEGA2Ref) / 2
    df["Fp1Av"] = DF.EEGFp1Ref - Av
    df["Fp2Av"] = DF.EEGFp2Ref - Av
    df["F3Av"] = DF.EEGF3Ref - Av
    df["F4Av"] = DF.EEGF4Ref - Av
    df["C3Av"] = DF.EEGC3Ref - Av
    df["C4Av"] = DF.EEGC4Ref - Av
    df["P3Av"] = DF.EEGP3Ref - Av
    df["P4Av"] = DF.EEGP4Ref - Av
    df["O1Av"] = DF.EEGO1Ref - Av
    df["O2Av"] = DF.EEGO2Ref - Av
    df["F7Av"] = DF.EEGF7Ref - Av
    df["F8Av"] = DF.EEGF8Ref - Av
    df["T3Av"] = DF.EEGT3Ref - Av
    df["T4Av"] = DF.EEGT4Ref - Av
    df["T5Av"] = DF.EEGT5Ref - Av
    df["T6Av"] = DF.EEGT6Ref - Av
    df["FzAv"] = DF.EEGFzRef - Av
    df["CzAv"] = DF.EEGCzRef - Av
    df["PzAv"] = DF.EEGPzRef - Av
    Avdf = -pd.DataFrame(df)
    return Avdf


def get_event_with_time(raw):
    Event = mne.io.find_edf_events(raw)

    Event_list = []
    for i in range(Event.shape[0]):
        if Event[i][2] == "P_COMMENT" or Event[i][2][:3] == "HVT":
            continue
        Event_list.append(Event[i])
    Event_list = np.array(Event_list)

    Event_with_time = [("Start", 0)]
    for i in range(Event_list.shape[0] - 1):
        if (
            Event_list[i][2][0] == "+"
            and Event_list[i + 1][2][:2] in ["SZ", "sz"]
            and Event_with_time[-1][0] != "SZ"
        ):
            Event_with_time.append(("SZ", int(float(Event_list[i][2]) * 1000)))
        elif (
            Event_list[i][2][0] == "+"
            and Event_list[i + 1][2] in ["END", "end"]
            and Event_with_time[-1][0] != "END"
        ):
            Event_with_time.append(("END", int(float(Event_list[i][2]) * 1000)))
    Event_with_time.remove(("Start", 0))
    return Event_with_time


def get_SZ_time(Event_with_time):
    SZ_time = []
    for i in range(len(Event_with_time) - 1):
        if Event_with_time[i][0] == "SZ" and Event_with_time[i + 1][0] == "END":
            start_time, end_time = Event_with_time[i][1], Event_with_time[i + 1][1]
            SZ_time.append((start_time, end_time))
    return SZ_time


def original_data(raw):
    ch_names = raw.info["ch_names"]
    ch_names = [ch_names[i].replace(" ", "") for i in range(len(ch_names))]
    ch_names = [ch_names[i].replace("-", "") for i in range(len(ch_names))]

    dic = {}
    data = raw.get_data()
    col, N = data.shape
    N = N - N % 1000
    data = data[:, :N] * 1000000

    for i in range(col):
        dic.update({ch_names[i]: data[i, :]})
    DF = pd.DataFrame(dic)
    Avdf = recover_signal(DF)
    columns = [
        "Fp1Av",
        "Fp2Av",
        "F3Av",
        "F4Av",
        "C3Av",
        "C4Av",
        "P3Av",
        "P4Av",
        "O1Av",
        "O2Av",
        "F7Av",
        "F8Av",
        "T3Av",
        "T4Av",
        "T5Av",
        "T6Av",
        "FzAv",
        "CzAv",
        "PzAv",
    ]
    Avdf = Avdf[columns]
    return Avdf


def insert_ones(y, SZ_time):
    for time in SZ_time:
        y[time[0] : time[1]] = 1
    return y


"""


def seperate_sz_unsz(X, Y, y_sz_label, n_choose_X0):
    index_0 = np.nonzero(y_sz_label==0)[0]
    index_1 = np.nonzero(y_sz_label==1)[0]
    X_label0 = X[index_0,:,:]
    X_label1 = X[index_1,:,:]
    y_0 = Y[index_0]
    y_1 = Y[index_1]
    index = np.arange(X_label0.shape[0])
    np.random.shuffle(index)
    index = index[:n_choose_X0]
    X_0 = X_label0[index]
    y_0 = y_0[index]
    return X_0, y_0, X_label1, y_1

def slide_auto_cor(data_with_y, slide_length=200, noverlap=120):
    data = data_with_y[0][:,3]
    y = data_with_y[1]
    N = data.shape[0]
    n = int((N-slide_length)/(slide_length-noverlap)+1)

    cor = []
    Y = np.zeros((n, 1))

    for i in range(n):
        x = data[i*80:i*80+200]
        cor.append(auto_cor(x))
        if y[i*80+100] == 1:
            Y[i] = 1
    cor = np.concatenate(cor)
    cor = cor.reshape(n, -1)
    return cor, Y

def array_auto_cor(X, y, slide_length=200, noverlap=120):
    X = X[:,:,3]
    X = X.squeeze()
    y = y.squeeze()
    m, T = X.shape
    step = slide_length-noverlap
    n = int((T-slide_length)/(slide_length-noverlap)+1)
    X_cor = []
    Y = []
    for j in range(m):
        cor = []
        yy = np.zeros((n, 1))
        for i in range(n):
            x = X[j,i*step:i*step+slide_length]
            cor.append(auto_cor(x))
            if y[j,i*step+int(slide_length/2)] == 1:
                yy[i] = 1
        cor = np.concatenate(cor)
        cor = cor.reshape(n, -1)
        X_cor.append(cor)
        Y.append(yy)
        if j%100 == 0:
            print('m = ', j)
    X_cor, Y = np.concatenate(X_cor), np.concatenate(Y)
    X_cor, Y = X_cor.reshape(m, n, -1), Y.reshape(m, -1)
    return X_cor,Y

def auto_cor(data):
    cor_list = []
    n = len(data)
    for k in range(1, n):
        mean = np.mean(data)
        std = np.std(data)
        data1 = data[:n - k]
        data2 = data[k:]
        cor = np.dot((data1 - mean).T, (data2 - mean)) / std
        cor_list.append(cor/n)
    cor = np.array(cor_list)
    return cor

def get_fft(acc):
    N = int(len(acc)/2)
    Y = np.fft.fft(acc)
    Y = Y[:N]
    absY = np.abs(Y)
    return absY

def graph_spectrogram(data):
    nfft = 200 # Length of each window segment
    fs = 20 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

def filt_low(inputs, N, Wn):
    b, a = signal.butter(N, 2*Wn/1000, 'low')
    outputs = signal.filtfilt(b, a, inputs)
    return outputs

def Error_correction(orig_signal, ini_stage_signal):
    ee = orig_signal - ini_stage_signal
    N = len(ee)           ##信号长度
    top_index = []        ##存储极值点位置
    box = []              ##存储所有不为零的box片段的位置
    for i in range(1,N-1):##找到差值信号的全部极值点位置
        if (ee[i] - ee[i-1]) * (ee[i+1] - ee[i]) < 0:
            top_index.append(i)

    for i in range(len(top_index)):       ##对于每一个极值点，向前和向后搜索，确认这一段非零尖波信号的boxi
        top = top_index[i]
        boxi = []                         ##存储这一段信号的位置索引
        n = top
        while ee[n] != 0 or ee[n-1] != 0: ##向左搜索，直到两个值连续为零
            boxi.append(n)
            n = n-1
            if n < 0:
                break                     ##已到最左端，退出
        n = top + 1
        while ee[n] != 0 or ee[n+1] != 0: ##向右搜索，直到两个值连续为零
            boxi.append(n)
            n = n+1
            if n >= N:
                break                     ####已到最右端，退出
        boxi = np.array(boxi)             ##对于符合要求的尖波，两个极值点所确认的区域相同，所
        boxi = np.sort(boxi)              ##以两次搜索得到的索引有序排列后会重复，需要进行剔除
        box.append(boxi)                  ##将第i个非零片段索引存入box

    box2 = []
    if not box == []:
        box2.append(box[0])
        for i in range(len(box)-1):       ##剔除box中的重复片段索引
            if len(box[i+1]) != len(box[i]):
                box2.append(box[i+1])
            elif not (box[i+1] == box[i]).all():
                box2.append(box[i+1])
    box = box2

#     for i in range(len(box)):
#         boxi = box[i]
#         for j in range(len(boxi)):
#             plt.axvline(boxi[j], c='r')
#     plt.figure(figsize=[15,2])
#     plt.plot(ee, c = 'b')

    for i in range(len(box)):           ##对于差值信号的每个片段，检验其是否符合尖波特征，不符合归零
        boxi = box[i]
        cut = ee[boxi]
        if (cut >= 0).all() or (cut <= 0 ).all():
            ee[boxi] = 0
#     plt.figure(figsize=[15,2])
#     plt.plot(ee, c = 'g')
#     plt.show()
    return ee

"""

