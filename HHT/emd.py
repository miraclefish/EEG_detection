import math
import numpy as np 
import pylab as pl
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import fftpack  
import scipy.signal as signal
from scipy import interpolate

#判定当前的时间序列是否是单调序列
def ismonotonic(x):
    max_peaks=signal.argrelextrema(x,np.greater)[0]
    min_peaks=signal.argrelextrema(x,np.less)[0]
    all_num=len(max_peaks)+len(min_peaks)
    if all_num>0:
        return False
    else:
        return True
        
#寻找当前时间序列的极值点
def findpeaks(x):
    return signal.argrelextrema(x,np.greater)[0]

#判断当前的序列是否为 IMF 序列
def isImf(x):
    N=np.size(x)
    pass_zero=np.sum(x[0:N-2]*x[1:N-1]<0)#过零点的个数
    peaks_num=np.size(findpeaks(x))+np.size(findpeaks(-x))#极值点的个数
    if abs(pass_zero-peaks_num)>1:
        return False
    else:
        return True
#获取当前样条曲线
def getspline(x):
    N=np.size(x)
    peaks=findpeaks(x)
#     print '当前极值点个数：',len(peaks)
    peaks=np.concatenate(([0],peaks))
    peaks=np.concatenate((peaks,[N-1]))
    if(len(peaks)<=3):
#         if(len(peaks)<2):
#             peaks=np.concatenate(([0],peaks))
#             peaks=np.concatenate((peaks,[N-1]))
#             t=interpolate.splrep(peaks,y=x[peaks], w=None, xb=None, xe=None,k=len(peaks)-1)
#             return interpolate.splev(np.arange(N),t)
        t=interpolate.splrep(peaks,y=x[peaks], w=None, xb=None, xe=None,k=len(peaks)-1)
        return interpolate.splev(np.arange(N),t)
    t=interpolate.splrep(peaks,y=x[peaks])
    return interpolate.splev(np.arange(N),t)
#     f=interp1d(np.concatenate(([0,1],peaks,[N+1])),np.concatenate(([0,1],x[peaks],[0])),kind='cubic')
#     f=interp1d(peaks,x[peaks],kind='cubic')
#     return f(np.linspace(1,N,N))
    
#经验模态分解方法
def emd(x):
    imf=[]
    i = 0
    while not ismonotonic(x):
        x1=x
        sd=np.inf
        i += 1
        while sd>0.1 or (not isImf(x1)):
            s1=getspline(x1)
            s2=-1*getspline(-1*x1)
            x2=x1-(s1+s2)/2
            sd=np.sum((x1-x2)**2)/np.sum(x1**2)
            x1=x2
        
        imf.append(x1)
        x=x-x1
        if i == 24:
            break

    imf.append(x)
    return imf

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def extend_arctan(hx, imf):
    w = np.arctan(hx/(imf+10e-8))
    dw = np.zeros(len(w))
    dw[0] = 0
    for i in range(len(w)-1):
        if w[i+1] < w[i]:
            dw[i+1] = np.pi + w[i+1] - w[i]
        else:
            dw[i+1] = w[i+1] - w[i]
    extend_w = [ w+np.sum(dw[:i+1]) for i in range(len(dw))]
    extend_w = np.array(extend_w)
    return dw, extend_w

def hilbert_transform(imfs, dt):
    H = []
    F = []
    for imf in imfs:
        hx = fftpack.hilbert(imf)
        amplitude = np.sqrt(imf**2 + hx**2)[1:]
        dw, extend_w = extend_arctan(hx, imf)
        frequence = np.round(dw/dt/(2*np.pi)).astype(int)
        H.append(amplitude)
        F.append(frequence[1:])
    hh = hh_spectrum(H, F)
    return hh

def hh_spectrum(H, F):
    T = len(H[0])
    maxf = int(max([np.max(f) for f in F]))
    hh = np.zeros((maxf, T))
    for f, h in zip(F, H):
        for i in range(len(f)):
            if hh[f[i]-1, i] == 0:
                hh[f[i]-1, i] = h[i] + hh[f[i]-1, i]
            else:
                hh[f[i]-1, i] = h[i]
    return hh

fs = 1000
ts = 1/fs
t=np.arange(0,0.3,ts)
z = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*100*t)
Imf = emd(z)
for i in range(len(Imf)):
    plt.subplot(len(Imf),1,i+1)
    plt.plot(Imf(i))
plt.show()


# sampling_rate=1000
# dt = 1.0/sampling_rate

# imfs = emd(x1)
# n = len(imfs)
# nrows = math.ceil((n+1)/2)

# hh = hilbert_transform(imfs, dt)

# plt.imshow(hh.T, cmap=plt.cm.hot_r)
# plt.show()


# plt.figure(figsize=[20,15])
# plt.subplot(nrows, 2, 1)
# plt.plot(x1)
# plt.title("X")

# for i in range(len(imfs)):
#     plt.subplot(nrows, 2, i+2)
#     plt.plot(imfs[i])
#     plt.title("Imf"+str(i+1))

# plt.show()
# pass