import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.signal import butter

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
        
        self.pred_ind, self.output = self._bect_detection()
        self.indicator = self.get_indicator()
        
        self._print_output()
            
    def _read_txt(self):
        raw_data = pd.read_csv(self.path, sep='\t')
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
        print("Pred_S in channel \"{:s}\"-->{:d}; Indicator-->{:.2f}%.".format(self.s_channel, int(len(self.pred_ind)), self.indicator*100))
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
    
    def _bect_detection(self):

        # 设定默认的信号分割长度为1000
        detection_length = 1000

        # 等长分割信号
        data = self.data
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
            # 对 x 滑窗的过程矩阵并行化，详情请参考函数 self._window_slide()
            # xx 的每一行是一个 x 的滑窗提取，步长为 1
            xx = self._window_slide(x)

            # 对每个滑窗生成一个匹配模板，参考 xx 进行自适应变形，也进行了矩阵并行化。
            if self.template_mode == "pdf":
                template = self._pdf_spike(xx)
            elif self.tmplate_mode == "triang":
                template = self._triang_spike(xx)

            # 将 xx 与 template 逐元素相乘，每行求和得到原始信号与模板的内积
            # 求均值是为了让输出 score 变小一点
            score = np.mean(xx*template, axis=1)

            # 滑窗计算后两端补零到原始长度
            expanded_score = np.hstack([np.zeros(int(self.window/2)), score, np.zeros(self.window-int(self.window/2)-1)])

            # 存储输出结果为滤后信号 filted_data
            if i<len(datas)-1:
                filted_data[i*x.shape[0]:(i+1)*x.shape[0]] = expanded_score.reshape(-1,1)
            else:
                filted_data[-(len(datas)-i)*x.shape[0]:] = expanded_score.reshape(-1,1)

            
            outline_ind, flag = self._find_S_points(score)
            # print(i,"-->",flag)
            if len(outline_ind) > 0:
                outline_ind += i * detection_length
                pred_ind = pred_ind + list(outline_ind)
            i += 1
        pred_ind = np.array(pred_ind)

        return pred_ind, filted_data
    
    def _window_slide(self, x):
        stride = 1
        n = int((len(x)-(self.window-stride))/stride)
        out = np.zeros((n, self.window))
        for i in range(self.window-1):
            out[:,i] = x[i:-(self.window-1-i)].T
        out = (out.T - np.mean(out, axis=1)).T
        return out
    
    def _find_S_points(self, score):
        dscore = score[1:]-score[:-1]
        
        peak_ind = np.where(dscore[:-1]*dscore[1:]<0)[0]
        peak_score = score[peak_ind]
        flag = 0
        
        if np.std(peak_score) < 200:
            outline_ind = []
        else:
            flag_score = (score - min(score))/(max(score)-min(score))-0.5
            # peak_ind_0 = [ind for i, ind in enumerate(peak_ind) if i%2==0]
            # peak_ind_1 = [ind for i, ind in enumerate(peak_ind) if i%2==1]
            # if len(peak_ind_0) > len(peak_ind_1):
            #     peak_ind_0 = peak_ind_0[:-1]
            # elif len(peak_ind_1) > len(peak_ind_0):
            #     peak_ind_1 = peak_ind_1[:-1]
            # assert(len(peak_ind_0)==len(peak_ind_1))
            # delta = np.abs(flag_score[peak_ind_1] - flag_score[peak_ind_0])
            delta = np.abs(flag_score[peak_ind[1:]] - flag_score[peak_ind[:-1]])
            delta = delta[np.where(delta < 0.8)[0]]
            flag = np.sum(delta) 

            if flag > 3.0:
                outline_ind = []
            else:
                n = self.threshold
                mean = np.mean(peak_score)
                std = np.std(peak_score)
                ind = np.where(abs(peak_score-mean)>n*std)[0]
                outline_ind = peak_ind[ind]
                ind = np.where(score[outline_ind]>0)[0]
                outline_ind = outline_ind[ind]
                outline_ind += int(self.window/2)
        return outline_ind, flag
        
    def get_indicator(self):

        window = 1500

        length = self.data.shape[0]
        mask = np.zeros(length)
        mask[self.pred_ind] = 1
        N = int(length/window)
        rest = length%window
        mask = mask[:-rest]
        rest = mask[-rest:]
        split = mask.reshape(N, window)
        out = np.zeros(N+1)
        out[:N] = split.sum(axis=1)
        out[-1] = sum(rest)
        indicator = 1-(len(np.where(out==0)[0])/len(out))

        return indicator

    # def get_indicator(self):
    #     split_length = self.pred_ind[1:]-self.pred_ind[:-1]
    #     long_split_length = split_length[np.where(split_length>=2000)[0]]
    #     long_split_second = np.floor(long_split_length/1000)
    #     indicator = 1-(np.sum(long_split_second)*1000/len(self.s_data))
    #     return indicator
    
    def plot_result(self, slice_ind=None):
        if slice_ind == None:
            size = [0, len(self.data)-1]
        else:
            if slice_ind[1] > len(self.data)-1:
                slice_ind[1] = len(self.data)-1
            size = [slice_ind[0], slice_ind[1]]
        
        plt.figure(figsize=[15,3])
        i = 0
        i += 1
        plt.subplot(2,1,i)
        plt.plot(self.raw_data[self.s_channel][size[0]:size[1]], linewidth="1")
        plt.title("Signal of "+self.s_channel+" channel")
        
        i += 1
        plt.subplot(2,1,i)
        plt.plot(np.arange(size[0],size[1]),self.output[size[0]:size[1]])
        plt.title("Detection")
        for ind in self.pred_ind:
            if ind>=size[0] and ind<=size[1]:
                plt.axvline(ind, c="g")
        plt.tight_layout()
        plt.show()
        return None