import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt

class BECTdetect(object):
    
    def __init__(self, filepath, print_log=False):
        
        self.path = filepath
        self.filename = self.path[-14:-4]
        self.print_log = print_log
        
        self.data = self.load_data()
        self.bandPassData = self._band_pass_filter(LowHz=0.5, HigHz=40, data=self.data)
        self.maskPassData = self._band_pass_filter(LowHz=0.5, HigHz=8, data=self.data)

        self.score = None
        self.spike_ind = None
        self.spike_score = None
        self.band_pair = None
    
    def load_data(self):

        if self.print_log:
            print("File loading: \""+self.path+"\".")

        raw_data = pd.read_csv(self.path, sep='\t')
        self.s_channel = raw_data.columns[0]
        data = raw_data.values

        if self.print_log:
            print("The length of the data is {:.3f}s.".format(data.shape[0]/1000))

        return data
    
    def Analysis(self, Spike_width, threshold, template_mode):

        self.score = self.Adaptive_Decomposition(Spike_width, template_mode)

        self.spike_ind, self.spike_score = self.bect_discrimination(score=self.score, threshold=threshold)

        band_ind = self.band_ind_expand(spike_ind=self.spike_ind)

        self.band_pair = self.find_slow_wave(band_ind=band_ind)

        SWI = self.get_SWI(band_pair=self.band_pair)

        return SWI


    def Adaptive_Decomposition(self, Spike_width, template_mode):
        
        template = np.zeros(Spike_width)
        if template_mode == "beta":
            template = self._beta_template(Spike_width)
        elif template_mode == "gamma":
            template = self._gamma_template(Spike_width)
        elif template_mode == "triang":
            template = self._triang_template(Spike_width)
        else:
            raise("请确认一种小波模板类型, 可选['beta', 'gamma', 'triang']")

        # 保证滑窗滤波后信号长度与原信号长度相同，进行Padding操作
        pad_width = ((int((Spike_width-1)/2),int((Spike_width-1)/2)), (0,0))
        x_pad = np.pad(self.bandPassData, pad_width=pad_width, mode='constant', constant_values=0)

        # 对 data 滑窗的过程矩阵并行化，详情请参考函数 self._window_slide()
        # data_windowed 的每一行是一个 data 的滑窗提取，步长为 1
        data_windowed = self._window_slide(x_pad, Spike_width)
        wave = self._Adaptive_wave(Original_wave=template, data=data_windowed)

        score = np.sum(data_windowed*wave, axis=1)/data_windowed.shape[1]**2
        return score

    def bect_discrimination(self, score, threshold):
        dscore = score[1:]-score[:-1]
        peak_ind = np.where(dscore[:-1]*dscore[1:]<0)[0]+1

        peak_score = score[peak_ind]
        peak_score = np.sign(peak_score)*np.log(np.abs(peak_score)+1)

        spike_ind = np.where(peak_score - np.mean(peak_score) > np.std(peak_score)*threshold)[0]
        spike_ind = peak_ind[spike_ind]
        return spike_ind, peak_score

    def band_ind_expand(self, spike_ind):
        d_data = self.bandPassData[1:] - self.bandPassData[:-1]
        peak_ind = np.where(d_data[:-1]*d_data[1:]<0)[0]+1
        l = len(peak_ind)
        band_ind = []
        for ind in spike_ind:
            loc = int(np.sum(peak_ind<ind))-1
            if loc-1 >=0 and loc+1<=l-1:
                band_ind.append(peak_ind[loc-1])
                band_ind.append(peak_ind[loc+1])
        return band_ind

    def find_slow_wave(self, band_ind):

        d_mask_data = self.maskPassData[1:] - self.maskPassData[:-1]
        peak_ind = np.where(d_mask_data[:-1]*d_mask_data[1:]<0)[0]+1

        band_pair = np.array(band_ind).reshape(-1,2)
        for i, ind_pair in enumerate(band_pair):
            loc = int(np.sum(peak_ind<ind_pair[1]))-1
            if loc+3 < len(peak_ind):
                
                # 慢波宽度校准
                candidate_wave_length = peak_ind[loc+3] - peak_ind[loc+1]
                low_bound = (ind_pair[1]-ind_pair[0])*1
                high_bound = (ind_pair[1]-ind_pair[0])*7
                length_flag = candidate_wave_length > low_bound and candidate_wave_length < high_bound

                # 慢波高度校准
                candidate_slow_wave = self.maskPassData[peak_ind[loc+1]:peak_ind[loc+3]]
                spike_wave = self.maskPassData[ind_pair[0]:ind_pair[1]]
                candidate_wave_high = np.max(candidate_slow_wave) - np.min(candidate_slow_wave)
                low_bound = (np.max(spike_wave)-np.min(spike_wave))*0.33
                high_flag = candidate_wave_high > low_bound

                if high_flag and length_flag:
                    band_pair[i,1] = peak_ind[loc+3]

        band_pair = band_pair.reshape(-1,1).squeeze()
        return band_pair

    def get_SWI(self, band_pair):
        mask = np.zeros(len(self.data))
        band_pair = band_pair.reshape(-1,2)
        for ind_pair in band_pair:
            mask[ind_pair[0]:ind_pair[1]] = 1
        spike_time = np.sum(mask)
        SWI = spike_time/len(self.data)
        return SWI

    def _Adaptive_wave(self, Original_wave, data):
        # 小波的尺度根据原始信号的形状做自适应调整
        min_data = np.min(data, axis=1)
        max_data = np.max(data, axis=1)
        Original_wave = Original_wave.reshape(1, -1)
        wave = np.tile(Original_wave, [data.shape[0], 1])
        out = (wave.T*(max_data-min_data)+min_data).T
        return out

    def _beta_template(self, Spike_width):
        alpha=2
        beta=3.2
        x = np.linspace(0,1,Spike_width)
        template = stats.beta(alpha, beta).pdf(x)
        template = (template-min(template))/(max(template)-min(template))
        return template
    
    def _gamma_template(self, Spike_width):
        x = np.linspace(0,4,Spike_width)
        template = stats.gamma.pdf(x, a=2.2)
        template = template**2
        template = (template-min(template))/(max(template)-min(template))
        return template

    def _triang_template(self, Spike_width):
        template = np.zeros(Spike_width)
        mid_ind = int(Spike_width/3)
        template[:mid_ind] = np.linspace(0, 1, mid_ind, endpoint=False)
        template[mid_ind:] = np.linspace(1, 0, Spike_width-mid_ind)
        return template

    def _window_slide(self, x, Spike_width):
        stride = 1
        n = int((len(x)-(Spike_width-stride))/stride)
        out = np.zeros((n, Spike_width))
        for i in range(Spike_width-1):
            out[:,i] = x[i:-(Spike_width-i-1)].squeeze()
        out[:,-1] = x[Spike_width-1:].squeeze()
        out = (out.T - np.mean(out, axis=1)).T
        return out

    def _band_pass_filter(self, LowHz, HigHz, data):
        data = np.squeeze(data)
        hf = HigHz * 2.0 / 1000
        lf = LowHz * 2.0 / 1000
        N = 2
        b, a = butter(N, [lf, hf], "bandpass")
        filted_data = filtfilt(b, a, data)
        filted_data = filted_data.reshape(-1, 1)
        return filted_data

    def plot_result(self, slice_ind=None):
        if slice_ind == None:
            size = [0, len(self.data)-1]
        else:
            if slice_ind[1] > len(self.data)-1:
                slice_ind[1] = len(self.data)-1
            size = [slice_ind[0], slice_ind[1]]
        
        n = 4
        i = 0
        plt.figure(figsize=[15,4])
        plt.clf()

        i += 1
        plt.subplot(n,1,i)
        plt.plot(np.arange(size[0],size[1]), self.bandPassData[size[0]:size[1]], linewidth="1")
        plt.title(self.filename+" Signal of "+self.s_channel+" channel")

        i += 1
        plt.subplot(n,1,i)
        plt.plot(np.arange(size[0],size[1]),self.score[size[0]:size[1]])
        plt.title("Spike Detection with Threshold")

        for ind in self.spike_ind:
            if ind>=size[0] and ind<=size[1]:
                plt.axvline(ind, c="g")

        i += 1
        plt.subplot(n,1,i)
        plt.plot(np.arange(size[0],size[1]), self.maskPassData[size[0]:size[1]], linewidth="1")
        plt.title("Slow-wave Discriminent")

        for ind in self.band_pair.reshape(-1,2):
            if ind[0]>=size[0] and ind[1]<=size[1]:
                plt.plot(np.arange(ind[0], ind[1]), self.maskPassData[ind[0]:ind[1]],c="r")
        
        for ind in self.spike_ind:
            if ind>=size[0] and ind<=size[1]:
                plt.axvline(ind, c="g")

        i += 1
        plt.subplot(n,1,i)
        plt.plot(np.arange(size[0],size[1]), self.bandPassData[size[0]:size[1]], linewidth="1")
        plt.title("Detection Result")

        for ind in self.band_pair.reshape(-1,2):
            if ind[0]>=size[0] and ind[1]<=size[1]:
                plt.plot(np.arange(ind[0], ind[1]), self.bandPassData[ind[0]:ind[1]],c="r")
    
        plt.tight_layout()
        fig = plt.gcf()
        plt.show()
        return fig