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

        self.score = None
        self.spike_ind = None
        self.spike_score = None
        self.maskPassData = None
        self.band_pair = None
    
    def load_data(self):

        if self.print_log:
            print("File loading: \""+self.path+"\".")

        raw_data = pd.read_csv(self.path, sep='\t')
        self.s_channel = raw_data.columns[0]
        data = raw_data.values

        if self.print_log:
            print("The length of the data is {:.2f}s.".format(data.shape[0]/1000))

        bandPassData = self._band_pass_filter(LowHz=0.5, HigHz=40, data=data)
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
        x_pad = np.pad(self.data, pad_width=pad_width, mode='constant', constant_values=0)

        # 对 data 滑窗的过程矩阵并行化，详情请参考函数 self._window_slide()
        # data_windowed 的每一行是一个 data 的滑窗提取，步长为 1
        data_windowed = self._window_slide(self.data, Spike_width)
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
        d_data = self.data[1:] - self.data[:-1]
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

        self.maskPassData = self._band_pass_filter(LowHz=0.5, HigHz=8, data=self.data)
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

    def _get_pdf_template(self):
        
        # Use beta distribution
        if self.pdf_distribution == "beta":
            alpha=2
            beta=3.2
            x = np.linspace(0,1,Spike_width)
            pdf = stats.beta(alpha, beta).pdf(x)
        
        # Use gamma distribution
        elif self.pdf_distribution == "gamma":
            x = np.linspace(0,4,Spike_width)
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

    def _window_slide(self, x, Spike_width):
        stride = 1
        n = int((len(x)-(Spike_width-stride))/stride)
        out = np.zeros((n, Spike_width))
        for i in range(Spike_width-1):
            out[:,i] = x[i:-(Spike_width-1-i)].T
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
        
        # 如果 rest 的长度不够一个滑窗 Spike_width 的长度，就丢掉 rest
        if rest >= Spike_width:
            datas = datas + [data[-(len(data)-(N-1)*detection_length):]]
        
        # 定义滤后信号的存储空间
        filted_data = np.zeros(data.shape)

        # 定义检测出尖波位置的存储空间
        pred_ind = []
        
        i = 0
        for x in datas:
            # 保证滑窗滤波后信号长度与原信号长度相同，进行Padding操作
            pad_width = ((int((Spike_width-1)/2),int((Spike_width-1)/2)), (0,0))
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
            # expanded_score = np.hstack([np.zeros(int(Spike_width/2)), score, np.zeros(Spike_width-int(Spike_width/2)-1)])

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

    #     Spike_width = 1500

    #     length = self.data.shape[0]
    #     mask = np.zeros(length)
    #     mask[self.pred_ind] = 1
    #     N = int(length/Spike_width)
    #     rest = length%Spike_width
    #     mask = mask[:-rest]
    #     rest = mask[-rest:]
    #     split = mask.reshape(N, Spike_width)
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
