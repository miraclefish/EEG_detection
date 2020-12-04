import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt

class BECTdetect(object):
    
    def __init__(self, filepath, S_size, threshold, template_mode="pdf", NAME="S-BECT", print_log=True):
        self.NAME = NAME
        self.threshold = threshold
        self.print_log = print_log
        
        self.path = filepath
        self.filename = self.path[-14:-4]
        self.data, self.raw_data = self._read_txt()
        
        self.window = S_size
        self.template_mode = template_mode
        self.pdf_distribution = "gamma"
        if self.template_mode == "pdf":
            self._template = self._get_pdf_template()
        
        self.bandPassData = self._band_pass_filter(LowHz=0.5, HigHz=40, data=self.data)
        self.maskPassData = self._band_pass_filter(LowHz=0.5, HigHz=8, data=self.data)
        self.output1, self.diff_score = self._adaptive_filter(self.bandPassData)
        self.pred_ind, self.peak_score = self._bect_detection(self.output1)
        self.band_ind = self._band_ind_detection(self.bandPassData)
        self._find_slow_wave()

        self.indicator = self.get_metric()
        
        self._print_output()
        
            
    def _read_txt(self):
        raw_data = pd.read_csv(self.path, sep='\t')
        # raw_data = raw_data[0:86500]
        self.s_channel = raw_data.columns[0]
        data = raw_data.values
        if self.print_log:
            print("File loading: \""+self.filepath+"\".")
            print("The length of the data is "+str(data.shape[0])+".")
        return data, raw_data
    
    def _print_output(self):
        print(" ")
        print("**********{:s}***********".format(self.filename))
        print("Signal Length-->{:d} s;".format(int(self.data.shape[0]/1000)))
        print("Pred_S in channel \"{:s}\"-->{:d}; Metric-->{:.2f}%.".format(self.s_channel, int(len(self.pred_ind)), self.indicator*100))
        return None
    
    def _get_pdf_template(self):
        
        # Use beta distribution
        if self.pdf_distribution == "beta":
            alpha=2
            beta=3.2
            x = np.linspace(0,1,self.window)
            pdf = stats.beta(alpha, beta).pdf(x)
        
        # Use gamma distribution
        elif self.pdf_distribution == "gamma":
            x = np.linspace(0,4,self.window)
            pdf = stats.gamma.pdf(x, a=2.2)
            pdf = pdf**2
        return pdf
    def _triang_spike(self, x):
        # 模板的尺度根据原始信号的形状做调整
        minx = np.min(x, axis=1)
        maxx = np.max(x, axis=1)

        out = np.zeros(x.shape)
        mid_ind = int(self.windows*1/3)
        out[:, :mid_ind] = np.linspace(minx, maxx, mid_ind, axis=1)
        out[:, mid_ind:] = np.linspace(maxx, minx, length-mid_ind, axis=1)
        return out
    def _pdf_spike(self, x):
        # 模板的尺度根据原始信号的形状做调整
        minx = np.min(x, axis=1)
        maxx = np.max(x, axis=1)
        
        pdf = self._template
        pdf = (pdf-min(pdf))/(max(pdf)-min(pdf))
        pdf = np.tile(pdf, [x.shape[0], 1])
        out = (pdf.T*(maxx-minx)+minx).T
        return out

    def _window_slide(self, x):
        stride = 1
        n = int((len(x)-(self.window-stride))/stride)
        out = np.zeros((n, self.window))
        for i in range(self.window-1):
            out[:,i] = x[i:-(self.window-1-i)].T
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

    def _adaptive_filter(self, data):

        # 设定默认的信号分割长度为1000
        detection_length = 4000

        # 等长分割信号
        N = int(np.ceil(len(data)/detection_length))

        # 先分割前 N-1 段
        datas = np.array_split(data[:-(len(data)-(N-1)*detection_length)], N-1)
        # 尾部长度不够 detection_length 的 1 段
        rest = len(data)-(N-1)*detection_length
        
        # 如果 rest 的长度不够一个滑窗 self.window 的长度，就丢掉 rest
        if rest >= self.window:
            datas = datas + [data[-(len(data)-(N-1)*detection_length):]]
        
        # 定义滤后信号的存储空间
        filted_data = np.zeros(data.shape)

        # 定义检测出尖波位置的存储空间
        pred_ind = []
        
        i = 0
        for x in datas:
            # 保证滑窗滤波后信号长度与原信号长度相同，进行Padding操作
            pad_width = ((int((self.window-1)/2),int((self.window-1)/2)), (0,0))
            x_pad = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

            # 对 x 滑窗的过程矩阵并行化，详情请参考函数 self._window_slide()
            # xx 的每一行是一个 x 的滑窗提取，步长为 1
            xx = self._window_slide(x_pad)

            # 对每个滑窗生成一个匹配模板，参考 xx 进行自适应变形，也进行了矩阵并行化。
            if self.template_mode == "pdf":
                template = self._pdf_spike(xx)
            elif self.tmplate_mode == "triang":
                template = self._triang_spike(xx)

            # 将 xx 与 template 逐元素相乘，每行求和得到原始信号与模板的内积
            # 求均值是为了让输出 score 变小一点
            score = np.sum(xx*template, axis=1)/xx.shape[1]**2

            # # 滑窗计算后两端补零到原始长度
            # expanded_score = np.hstack([np.zeros(int(self.window/2)), score, np.zeros(self.window-int(self.window/2)-1)])

            # 存储输出结果为滤后信号 filted_data
            if i<len(datas)-1:
                filted_data[i*x.shape[0]:(i+1)*x.shape[0]] = score.reshape(-1,1)
            else:
                filted_data[-(len(datas)-i)*x.shape[0]:] = score.reshape(-1,1)

            i += 1
        
        diff_score = np.zeros((len(filted_data),1))
        diff_score[1:] = filted_data[1:] - filted_data[:-1]

        return filted_data, diff_score

    def _find_slow_wave(self):
        d_mask_data = self.maskPassData[1:] - self.maskPassData[:-1]
        peak_ind = np.where(d_mask_data[:-1]*d_mask_data[1:]<0)[0]+1
        band_pair = np.array(self.band_ind).reshape(-1,2)
        for i, ind_pair in enumerate(band_pair):
            loc = int(np.sum(peak_ind<ind_pair[1]))-1
            if loc+3 < len(peak_ind):

                candidate_wave_length = peak_ind[loc+3] - peak_ind[loc+1]
                low_bound = (ind_pair[1]-ind_pair[0])*1
                high_bound = (ind_pair[1]-ind_pair[0])*7
                length_flag = candidate_wave_length > low_bound and candidate_wave_length < high_bound

                candidate_slow_wave = self.maskPassData[peak_ind[loc+1]:peak_ind[loc+3]]
                spike_wave = self.maskPassData[ind_pair[0]:ind_pair[1]]
                candidate_wave_high = np.max(candidate_slow_wave) - np.min(candidate_slow_wave)
                low_bound = (np.max(spike_wave)-np.min(spike_wave))*0.33
                high_flag = candidate_wave_high > low_bound
                if high_flag and length_flag:
                    band_pair[i,1] = peak_ind[loc+3]
        self.band_ind = band_pair.reshape(-1,1).squeeze()
        return None

    def _band_ind_detection(self, diff_score):
        dd_score = diff_score[1:] - diff_score[:-1]
        peak_ind = np.where(dd_score[:-1]*dd_score[1:]<0)[0]+1
        l = len(peak_ind)
        band_ind = []
        for ind in self.pred_ind:
            loc = int(np.sum(peak_ind<ind))-1
            if loc-1 >=0 and loc+1<=l-1:
                band_ind.append(peak_ind[loc-1])
                band_ind.append(peak_ind[loc+1])
        return band_ind
    
    def _bect_detection(self, score):
        dscore = score[1:]-score[:-1]
        peak_ind = np.where(dscore[:-1]*dscore[1:]<0)[0]+1

        peak_score = score[peak_ind]
        peak_score = np.sign(peak_score)*np.log(np.abs(peak_score)+1)

        large_ind = np.where(peak_score - np.mean(peak_score) > np.std(peak_score)*self.threshold)[0]
        ind = peak_ind[large_ind]
        pred_ind = ind
        return pred_ind, peak_score
        
    # def get_metric(self):

    #     window = 1500

    #     length = self.data.shape[0]
    #     mask = np.zeros(length)
    #     mask[self.pred_ind] = 1
    #     N = int(length/window)
    #     rest = length%window
    #     mask = mask[:-rest]
    #     rest = mask[-rest:]
    #     split = mask.reshape(N, window)
    #     out = np.zeros(N+1)
    #     out[:N] = split.sum(axis=1)
    #     out[-1] = sum(rest)
    #     indicator = 1-(len(np.where(out==0)[0])/len(out))

    #     return indicator

    # def get_metric(self):
    #     split_length = self.pred_ind[1:]-self.pred_ind[:-1]
    #     long_split_length = split_length[np.where(split_length>=1500)[0]]
    #     long_split_second = np.floor(long_split_length/1000)
    #     indicator = 1-(np.sum(long_split_second)*1000/len(self.data))
    #     return indicator
    
    def get_metric(self):
        mask = np.zeros((len(self.output1),1))
        band_pair = self.band_ind.reshape(-1,2)
        for ind_pair in band_pair:
            mask[ind_pair[0]:ind_pair[1]] = 1
        spike_time = np.sum(mask)
        metric = spike_time/len(self.output1)
        return metric


    def plot_result(self, slice_ind=None):
        if slice_ind == None:
            size = [0, len(self.data)-1]
        else:
            if slice_ind[1] > len(self.data)-1:
                slice_ind[1] = len(self.data)-1
            size = [slice_ind[0], slice_ind[1]]
        
        n = 4
        i = 0
        self.p = plt.figure(figsize=[15,4])
        plt.clf()

        i += 1
        plt.subplot(n,1,i)
        # plt.plot(self.raw_data[self.s_channel][size[0]:size[1]], linewidth="1")
        plt.plot(np.arange(size[0],size[1]), self.bandPassData[size[0]:size[1]], linewidth="1")
        plt.title(self.filename+" Signal of "+self.s_channel+" channel")

        i += 1
        plt.subplot(n,1,i)
        plt.plot(np.arange(size[0],size[1]),self.output1[size[0]:size[1]])
        plt.title("Spike Detection with Threshold")

        for ind in self.pred_ind:
            if ind>=size[0] and ind<=size[1]:
                plt.axvline(ind, c="g")

        i += 1
        plt.subplot(n,1,i)
        plt.plot(np.arange(size[0],size[1]), self.maskPassData[size[0]:size[1]], linewidth="1")
        plt.title("Slow-wave Discriminent")

        for ind in np.array(self.band_ind).reshape(-1,2):
            if ind[0]>=size[0] and ind[1]<=size[1]:
                plt.plot(np.arange(ind[0], ind[1]), self.maskPassData[ind[0]:ind[1]],c="r")
        
        for ind in self.pred_ind:
            if ind>=size[0] and ind<=size[1]:
                plt.axvline(ind, c="g")

        i += 1
        plt.subplot(n,1,i)
        plt.plot(np.arange(size[0],size[1]), self.bandPassData[size[0]:size[1]], linewidth="1")
        plt.title("Detection Result")

        for ind in np.array(self.band_ind).reshape(-1,2):
            if ind[0]>=size[0] and ind[1]<=size[1]:
                plt.plot(np.arange(ind[0], ind[1]), self.bandPassData[ind[0]:ind[1]],c="r")
    
        plt.tight_layout()
        fig = plt.gcf()
        plt.show()
        return fig
