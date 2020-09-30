import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class VisualData(object):

    def __init__(self, NAME, num):
        self.NAME = NAME
        self.num = num
        self.path = os.getcwd() + "\\csv_data\\" + self.NAME
        self.filelist = os.listdir(self.path)
        self.filepath = self.path + "\\" + self.filelist[self.num-1]
        self.data, self.label, self.columns = self._read_csv()
        if self.NAME == "S-BECT":
            self.s_channel = self.filepath[-14:-12]

    def _read_csv(self):
        raw_data = pd.read_csv(self.filepath)
        data = raw_data.drop(["label"], axis=1)
        label = raw_data["label"].values
        columns = raw_data.columns.values[:-1]
        return data, label, columns

    def plot_data(self, time_slice=None, ratio=1, placement=120):
        """Show the EEG data in a simple way, you just need to choose a interval
        with time_scile=[start_time, end_time].
        """
        if time_slice == None:
            raise Exception("Choose a time interval!")

        N, n = self.data.shape
        start_time, end_time = max(time_slice[0], 0), min(time_slice[1], N - 1)

        plt.figure(figsize=[20, 15])

        i = 0
        ymax = 0
        ymin = 0
        for col in list(self.columns[::-1]):
            y = self.data[col][start_time:end_time]
            yplaced = y.values * ratio + i * placement
            if ymin >= yplaced.max():
                ymin = yplaced.max()
            if ymax <= yplaced.max():
                ymax = yplaced.max()
            
            color = "b"
            linew = 0.5
            if self.NAME == "S-BECT" and col[:2] == self.s_channel:
                color = "black"
                linew = 1
            plt.plot(y.index.values, yplaced, c=color, Linewidth=linew)
            i += 1

        if self.NAME in ["spasm", "tonic"]:
            x = np.arange(start_time, end_time)
            label = self.label[start_time:end_time]

            label_0_up = label
            label_0_down = np.zeros(label.shape)
            label_0_down[label == 0] = 1.0
            label_up = label_0_up * ymax + placement/10
            label_up[label == 0] = ymin - placement/10
            label_down = label_0_down * ymin - placement/10

            plt.plot(x, label_up, color="white", alpha=0.1)
            plt.plot(x, label_down, color="white", alpha=0.1)
            plt.fill_between(x, label_up, label_down, where=label_up > label_down, color="pink", alpha=0.5)

        if self.NAME in ["S-BECT", "myoclonus"]:
            event_times = np.where(self.label==1)[0]
            for event_time in event_times:
                if event_time < end_time and event_time >= start_time:
                    plt.axvline(event_time, c="r")
                    plt.text(event_time + 15, -80, s=self.NAME, color="r", fontdict={"size": 14})
                

        plt.yticks(np.arange(24) * placement, self.columns[::-1], rotation=45)
        plt.tick_params(labelsize=14)

        title = "<" + self.filelist[self.num-1] + "> EEG data slice show with <" + self.NAME + "> abnormal"
        plt.title(title, fontdict={"size": 16})
        plt.show()
        return None


# Indicate the type of data "NAME"
NAME = "S-BECT" # or "spasm", "tonic", "myoclonus", "S-BECT"

# Choose a file number
num = 5

Data = VisualData(NAME = NAME, num=num)
Data.plot_data([6000,7000])
pass