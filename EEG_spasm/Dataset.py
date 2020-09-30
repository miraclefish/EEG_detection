import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import scale
from scipy.fftpack import fft, ifft
from collections import Counter
import time


class Dataset:
    """A class for you to load data from .csv files and pre-process EEG data.
    
    Parameters:
    -----------
    NAME: char
        The type of the label need to be load.
    split_length: int
        The length of the EEG sample extracted from the original long EEG data,
        default setted to 5000.
    path: long char
        The root path of the .csv files.
    filelist: list[char]
        A .csv file list of all .csv files labeled as NAME. 
    filename: char
        The filename includes the patient's name abbreviation and the file 
        number, or sometimes the name of the particular channel.
    all_data: numpy.array --> shape(N, split_length, channels)
        Includes all data seperated from the original EEG data with splited length.
    channel_name: List[char]
        Storage the channel names of the EEG signal, including the name "label".
    data: dictionary["event_data": array, "no_event_data": array]
        Storage the filted data sorted as two parts, event data and no event data.
        The data in which the event duration is not totally included (both start 
        time and end time) was filted.
    events_num_list: List[0 or 1]
        Storage the flag of each EEG sample from 'data' which has been filted. 
    """

    def __init__(self, NAME, split_length=5000):
        self.NAME = NAME
        self.split_length = split_length
        self.path = os.getcwd() + "\\csv_data\\" + NAME
        self.filelist = os.listdir(self.path)
        self.all_data = self.read_all_data()
        self.data, self.events_num_list = self.filt_data()
        self.channel_name = self._load_channel_name()

    def read_all_data(self):
        all_data = []
        for filename in self.filelist:
            data = self._read_csv(self.path + "\\" + filename)
            all_data.append(data)
        all_data = np.concatenate(all_data, axis=0)
        return all_data

    def _load_channel_name(self):
        filepath = self.path + "\\" + self.filelist[0]
        raw_data = pd.read_csv(filepath)
        channel_name = raw_data.columns[:-1]
        return channel_name

    def _read_csv(self, filepath):
        raw_data = pd.read_csv(filepath, header=0)
        raw_data = raw_data.values
        data = self._split_data(raw_data)
        return data

    def _split_data(self, raw_data):
        """split the data with split_length and stride setted before."""
        stride = int(self.split_length / 2)
        n = int(raw_data.shape[0] / stride - 1)
        splited_data = np.zeros((n, self.split_length, raw_data.shape[1]))
        for i in range(n):
            splited_data[i] = raw_data[i * stride : i * stride + self.split_length, :]
        assert splited_data.shape == (n, self.split_length, raw_data.shape[1])
        return splited_data

    def data_info(self):
        events_list = self.events_num_list
        count = Counter(events_list)
        for item, num in count.items():
            print(str(item) + " events sample number: " + str(num))
        return count

    def filt_data(self):
        """filt the data in which the event duration is not totally included (both start time and end time)"""
        event_data = []
        no_event_data = []
        events_numbers = []
        all_data = self.all_data
        _, T, C = all_data.shape
        labels = all_data[:, :, -1]
        diff_labels = labels[:, 1:] - labels[:, :-1]
        for i in range(all_data.shape[0]):
            diff_label_i = diff_labels[i, :]
            index = np.argwhere(diff_label_i != 0)
            label_flag = np.squeeze(diff_label_i[index])
            if len(index):
                if len(index) == 1:
                    continue
                if label_flag[0] == -1 or label_flag[-1] == 1:
                    continue
            events_no = int(np.sum(np.abs(diff_label_i)) / 2)
            events_numbers.append(events_no)
            if events_no > 0:
                event_data.append(all_data[i])
            elif events_no == 0:
                no_event_data.append(all_data[i])
        event_data = np.concatenate(event_data, axis=0).reshape(-1, T, C)
        no_event_data = np.concatenate(no_event_data, axis=0).reshape(-1, T, C)
        data = {"event_data": event_data, "no_event_data": no_event_data}

        return data, events_numbers

    def getData(self, shuffle=False, fft_process=False, norm=False, only_P=True, L=None):
        data_X, data_Y = self._preProcess(fft_process=fft_process, norm=norm, L=L)
        if shuffle:
            index = np.arange(data_Y.shape[0])
            np.random.shuffle(index)
            data_X = data_X[index]
            data_Y = data_Y[index]
        return data_X, data_Y

    def _preProcess(self, fft_process=False, norm=False, only_P=True, L=138):
        ev_data = self.data["event_data"]
        ev_data_0 = ev_data[:, :, :-1]
        ev_data_label = ev_data[:, :, -1]
        if only_P==True:
            data = ev_data_0
            label = ev_data_label
        else:
            no_ev_data = self.data["no_event_data"]
            no_ev_data_0 = no_ev_data[:, :, :-1]
            no_ev_data_label = no_ev_data[:, :, -1]
            data = np.concatenate([ev_data_0, no_ev_data_0], axis=0)
            label = np.concatenate([ev_data_label, no_ev_data_label], axis=0)

        if norm == True:
            mean_data = np.mean(data, axis=1, keepdims=1)
            std_data = np.std(data, axis=1, keepdims=1)
            data = (data - mean_data)/std_data
            # data = np.transpose(data, axes=[0,2,1])
            # data = data.reshape(-1, self.split_length)
            # data = scale(data, axis=1)
            # data = data.reshape(-1, 23, 5000)
            # data = np.transpose(data, axes=[0,2,1])

        if fft_process == True:
            data_1 = self._seperate_data(data, length=500)
            data = self._fft_transation(data_1)
            L = data.shape[1]
            label = self._align_label(label, L, output_dim=1)
        if L != None:
            label = self._align_label(label, L, output_dim=1)
        print("There original data are sorted as shape" + str(data.shape))
        print("The label of data was sorted as shape " + str(label.shape))
        return data, label

    def _seperate_data(self, data, length=500):
        N, L, c = data.shape
        stride = int(length / 2)
        n = int((L - length) / stride) + 1
        X = np.zeros((N, n, length, c))
        for i in range(n):
            X[:, i, :, :] = data[:, i * stride : i * stride + length, :]
        return X

    def _fft_transation(self, data):
        """Feature with 1454 dimensions."""
        shape = data.shape
        N = shape[2]
        windows = self._choose_windows(name="Hamming", N=N)
        windows = windows.reshape(windows.shape[0], 1)
        windows = np.broadcast_to(windows, shape)
        data_windowed = data * windows
        data_windowed = np.transpose(data_windowed, axes=[0, 1, 3, 2])

        x = np.linspace(0, 0.25, N)
        yy = fft(data_windowed, axis=-1)
        yf = abs(yy)
        yf1 = yf / (len(x) / 2)
        yf2_EEG = yf1[:, :, :19, 4:70]
        yf2_EMG = yf1[:, :, 19:, 20:70]
        yf2_EEG = yf2_EEG.reshape(shape[0], shape[1], -1)
        yf2_EMG = yf2_EMG.reshape(shape[0], shape[1], -1)
        yf2 = np.concatenate([yf2_EEG, yf2_EMG], axis=-1)

        fft_data = yf2
        return fft_data

    def _choose_windows(self, name="Hamming", N=250):
        # Rect/Hanning/Hamming
        if name == "Hamming":
            window = np.array(
                [0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]
            )
        elif name == "Hanning":
            window = np.array(
                [0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]
            )
        elif name == "Rect":
            window = np.ones(N)
        return window

    def _align_label(self, label, L, output_dim=None):
        N, t = label.shape
        stride = int(t / (L + 1))
        length = 2 * stride
        label_0 = np.zeros((N, L, length))
        for i in range(L):
            label_0[:, i, :] = label[:, i * stride : i * stride + length]
        if output_dim == None:
            output_dim = 1
        if length % output_dim != 0:
            raise Exception(
                "You need to choose an output_dim which can decile seperated length = "
                + str(length)
            )
        l = int(length / output_dim)
        Y = np.zeros((N, L, output_dim))
        for j in range(output_dim):
            Y[:, :, j] = label_0[:, :, j * l + int(l / 2)]
        Y = Y.squeeze()
        return Y

    def plot_EEG_data(self, mode="demo", data=None, ratio=1.0, placement=5):
        """If you want to show your own test data, you need to input your data as shape (None, 24), where the
        first 23 channels includes the original EEG signal and the last channel restores the event labels.
        """
        if mode == "demo":
            data = self.data["event_data"][0]
        if mode == "test_show":
            if data.all() == None:
                raise Exception("You need to input your EEG data to this function!")
            assert data.shape[1] == 24 and len(data.shape) == 2

        ys = data[:, :-1]
        label = data[:, -1]

        plt.figure(figsize=[20, 15])

        x = np.linspace(0, 5, 5000)
        i = 0
        for ind in range(ys.shape[1]):
            y = ys[:, -(ind + 1)]
            color = "b"
            linew = 0.5
            y = y * ratio + i * placement
            ys[:, -(ind + 1)] = y
            plt.plot(x, y, c=color, Linewidth=linew)
            i += 1

        label_0_up = label
        label_0_down = np.zeros(label.shape)
        label_0_down[label == 0] = 1.0
        label_up = label_0_up * np.max(ys) + placement/10
        label_up[label == 0] = np.min(ys) - placement/10
        label_down = label_0_down * np.min(ys) - placement/10

        plt.plot(x, label_up, color="white", alpha=0.1)
        plt.plot(x, label_down, color="white", alpha=0.1)
        plt.fill_between(
            x,
            label_up,
            label_down,
            where=label_up > label_down,
            color="pink",
            alpha=0.5,
        )

        plt.yticks(np.arange(24) * placement, self.channel_name[::-1], rotation=45)
        plt.tick_params(labelsize=14)

        title = "<" + mode + "> EEG data slice show"
        plt.title(title, fontdict={"size": 16})
        plt.show()
        return None

# dataset = Dataset("spasm")
# dataX, dataY = dataset.getData(norm=False, fft_process=False, only_P=False)
# # data = np.concatenate([dataX[1], dataY[1][:,np.newaxis]], axis=1)
# # dataset.plot_EEG_data(mode="test_show", data=data)
# pass