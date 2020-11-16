import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt



def band_pass_filter(LowHz, HigHz, data):
    data = np.squeeze(data)
    hf = HigHz * 2.0 / 1000
    lf = LowHz * 2.0 / 1000
    N = 4
    b, a = butter(N, [lf, hf], "bandpass")
    filted_data = filtfilt(b, a, data)
    return filted_data

def g_(W, h, alpha):
    arr = np.zeros((2*W+1, 1)).squeeze()
    arr[:W+1] = h*(1-np.exp(-alpha*np.arange(0, W+1)))
    arr[W+1:] = h*(1-np.exp(-alpha*np.arange(W-1, -1, -1)))
    return arr

def window_slide(x, window, stride):
    n = int(len(x)-(window-1))
    out = np.zeros((n, window))
    for i in range(window-1):
        out[:,i] = x[i:-(window-1-i)].T
    output = out[::stride,:]
    return output

#腐蚀
def Erosion(f,g):
    M = len(g)
    N = len(f)
    fe = np.zeros((N,1)).squeeze()
    slide_f = window_slide(x=f, window=M, stride=1)
    fe[int(M/2):-int(M/2)] = np.min(slide_f - g, axis=1)
    return fe

#膨胀
def Dilation(f,g):
    M = len(g)
    N = len(f)
    fd = np.zeros((N,1)).squeeze()
    slide_f = window_slide(x=f, window=M, stride=1)
    fd[int(M/2):-int(M/2)] = np.max(slide_f + g, axis=1)
    return fd

#开运算
def Opening(f,g):
    fo=Dilation(Erosion(f,g),g)
    return fo

#闭运算
def Closing(f,g):
    fc=Erosion(Dilation(f,g),g)
    return fc

#PKD
def PKD(f,g1,g2):
    fk=Opening(f,g1)-Closing(Opening(f,g1),g2)
    return fk

#PTD
def PTD(f,g3,g4):
    ft=Closing(f,g3)-Opening(Closing(f,g3),g4)
    return ft

h = 3000000
alpha = 0.5
g1=g_(20,h,alpha)
g2=g_(40,h,alpha)
g3=g_(40,h,alpha)
g4=g_(14,h,alpha)

a = np.loadtxt("./NewcsvData/T5-zsm-001.txt", skiprows=1)
f = band_pass_filter(LowHz=0.5, HigHz=40, data=a)

x = f
i = 2
n = 5
plt.figure(figsize=[20,10])

plt.subplot(n, 1, 1)
plt.plot(x[i*5000:(i+1)*5000])
plt.title('x')

x_er = Erosion(x, g1)
plt.subplot(n, 1, 2)
plt.plot(x_er[i*5000:(i+1)*5000])
plt.title('x_er')

x_dl = Dilation(x_er, g1)
plt.subplot(n, 1, 3)
plt.plot(x_dl[i*5000:(i+1)*5000])
plt.title('x_dl')

x_dldl = Dilation(x_dl, g1)
plt.subplot(n, 1, 4)
plt.plot(x_dldl[i*5000:(i+1)*5000])
plt.title('x_dldl')

x_erer = Erosion(x_dldl, g1)
plt.subplot(n, 1, 5)
plt.plot(x_erer[i*5000:(i+1)*5000])
plt.title('x_erer')

plt.tight_layout()
plt.show()