import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import sklearn.preprocessing as prep
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])
    return np.array(accum)

def fft(time_data):
    return np.log10(np.absolute(np.fft.rfft(time_data, axis=1)[:,1:48]))

def freq_corr(fft_data):
    scaled = prep.scale(fft_data, axis=0)
    corr_matrix = np.corrcoef(scaled)
    eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
    eigenvalues.sort()
    corr_coefficients = upper_right_triangle(corr_matrix) # cusstom func
    return np.concatenate((corr_coefficients, eigenvalues))

def time_corr(time_data):
    resampled = signal.resample(time_data, 400, axis=1) if time_data.shape[-1] > 400 else data
    scaled = prep.scale(resampled, axis=0)
    corr_matrix = np.corrcoef(scaled)
    eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
    corr_coefficients = upper_right_triangle(corr_matrix) # custom func
    return np.concatenate((corr_coefficients, eigenvalues))

def transform(data):
    fft_out = fft(data)
    freq_corr_out = freq_corr(fft_out)
    time_corr_out = time_corr(data)
    return np.concatenate((fft_out.ravel(), freq_corr_out, time_corr_out))

def construct_label(label_name, m):
    swich = {"正常": np.zeros(m),
            "异常": np.zeros(m) + 1,
            "干扰": np.zeros(m) + 2}
    return swich[label_name]

def load_data(rootpath):
    eegdata = {}
    list = os.listdir(rootpath)
    for i, filename in enumerate(list):
        label_name = filename[-6:-4]
        datadict = sio.loadmat(rootpath + "\\" + filename)
        dataarray = datadict['dataarray']
        m = dataarray.shape[0]
        label = construct_label(label_name, m)
        eegdata[filename[:-4]] = (dataarray, label)
    return eegdata

def feature_extract(eegdata):
    X = []
    y = []
    for label_name, (data, label) in eegdata.items():
        m = data.shape[0]
        for i in range(m):
            feature = transform(data[i])
            X.append(feature[:,np.newaxis])
            y.append(label[i])
    X = np.concatenate(X, axis=-1)
    y = np.array(y)
    return X, y

def plot_2D(X_orig, y, label):
    estimator = PCA(n_components=2)
    X_2D = estimator.fit_transform(X_orig)
    pca_2 = X_2D
    plt.figure()
    n = len(label)
    if n == 1:
        index = [i for i in range(len(y)) if y[i] == label]
    elif n == 2:
        index = [i for  i in range(len(y)) if y[i] == label[0] or y[i] == label[1]]
    elif n == 3:
        index = [i for  i in range(len(y))]
    pca = pca_2[index]
    Y = y[index]
    ax, ay = pca[:,0], pca[:,1]
    C = []
    for l in Y:
        if l == 0:
            C.append('g')
        elif l == 1:
            C.append('r')
        elif l == 2:
            C.append('b')
    plt.scatter(ax, ay, c = C, alpha = 0.5)  
    return

def plot_3D(X_orig, y, label):
    estimator = PCA(n_components=3)
    X_3D = estimator.fit_transform(X_orig)
    pca_3 = X_3D
    fig = plt.figure()
    aa = fig.add_subplot(111, projection = '3d')
    n = len(label)
    if n == 1:
        index = [i for i in range(len(y)) if y[i] == label]
    elif n == 2:
        index = [i for i in range(len(y)) if y[i] == label[0] or y[i] == label[1]]
    elif n == 3:
        index = [i for i in range(len(y))]
    pca = pca_3[index]
    Y = y[index]
    ax, ay, az = pca[:, 0], pca[:, 1], pca[:, 2]
    C = []
    for l in Y:
        if l == 0:
            C.append('g')
        elif l == 1:
            C.append('r')
        elif l == 2:
            C.append('b')
    aa.scatter(ax, ay, az, c=C, alpha=0.5)