import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import mne  # version of mne=0.17.1

"""
You can use "conda install mne=0.17.1" or "pip install mne=0.17.1" to install the library "mne".
If you don't want to print the information in the process of reading .edf files, you need to manually 
modify the source code of the library "mne".

There are some tips:
1. File <mne/io/edf/edf.py>  line 188 needs to be commented out;
2. File <mne/io/edf/edf.py>  line 392 needs to be commented out;
3. File <mne/io/edf/edf.py>  line 436 needs to be commented out;
4. File <mne/io/edf/edf.py>  line 462,463 need to be commented out;
5. File <mne/io/edf/edf.py>  line 467,468 need to be commented out;
6. File <mne/io/edf/edf.py>  line 479,480,496 need to be commented out;
7. File <mne/io/edf/edf.py>  line 193 needs to be commented out;
8. File <mne/io/base.py>     line 663,664 need to be commented out;
9. File <mne/annotations.py> line 787,788 need to be commented out;
"""


class EDFreader(object):
    """A class for you to simplize the reading process of EDF files.
    If you have saved the EEG data as .csv files, maybe you don't need
    to use the EDFreader any more, unless you need to restore the .csv files.
    
    Parameters:
    -----------
    filepath: char
        The global path of the current file.
    filename: char
        The filename includes the patient's name abbreviation and the file 
        number, or sometimes the name of the particular channel.
    NAME: char
        The type of the label need to be load.
    raw: object
        Base object for Raw data, defined by the library "mne".
    event_times: list[float]
        It storages the time indices when the labeled events happend, 
        corresponding with the event_names (have the same length).
    event_names: list[char]
        It storages the labeled events name, corresponding with the event_times.
    Avdata: DataFrame
        It storages the sorted EEG data with Average Lead Method.
        The voltage unit of each channel is microvolts.
    """

    def __init__(self, file, NAME):
        self.filepath = os.getcwd() + "\\" + file
        self.filename = self.filepath[-11:]
        self.NAME = NAME
        self.raw = mne.io.read_raw_edf(self.filepath, preload=True)
        self.event_times, self.event_names = self.read_events()
        self.Avdata = self._read_original_data()

    def _recover_signal_EEG_AV(self, data):
        columns = [
            "Fp1-Av",
            "Fp2-Av",
            "F3-Av",
            "F4-Av",
            "C3-Av",
            "C4-Av",
            "P3-Av",
            "P4-Av",
            "O1-Av",
            "O2-Av",
            "F7-Av",
            "F8-Av",
            "T3-Av",
            "T4-Av",
            "T5-Av",
            "T6-Av",
            "Fz-Av",
            "Cz-Av",
            "Pz-Av",
            "A1",
            "A2",
        ]
        assert data.shape[0] == 21
        b, a = signal.butter(8, 0.24, "lowpass")
        filteddata = signal.filtfilt(b, a, data)
        Av_EEG = pd.DataFrame(filteddata.T, columns=columns)
        Av_EEG = -Av_EEG
        return Av_EEG

    def _read_original_data(self):
        """data with original shape (44, L), the corresponding channel name are
        
        ['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref', 'EEG C3-Ref',
        'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref', 'EEG O1-Ref', 'EEG O2-Ref',
        'EEG F7-Ref', 'EEG F8-Ref', 'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref',
        'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref', 'POL E',
        'POL PG1', 'POL PG2', 'EEG A1-Ref', 'EEG A2-Ref', 'POL T1', 'POL T2',
        'POL X1', 'POL X2', 'POL X3', 'POL X4', 'POL X5', 'POL X6', 'POL X7',
        'POL SpO2', 'POL EtCO2', 'POL DC03', 'POL DC04', 'POL DC05', 'POL DC06',
        'POL Pulse', 'POL CO2Wave', 'POL $A1', 'POL $A2', 'STI 014']
        
        We just consider 19 brain electrodes with 2 reference electrodes,
        and we don not consider the EMG.
        """
        data = self.raw.get_data()
        data = data * 1000000  # the original unit is Microvolt
        Av_data = pd.DataFrame()
        if self.NAME in ["S-BECT"]:
            Av_data = self._read_data_without_EMG(data)
        elif self.NAME in ["myoclonus", "spasm", "tonic"]:
            Av_data = self._read_data_with_EMG(data)

        return Av_data

    def _read_data_without_EMG(self, data):
        """Read EEG data without electromyogram (EMG)"""
        data = np.vstack((data[0:19, :], data[22:24, :]))
        Av_data_EEG = self._recover_signal_EEG_AV(data)
        Av_data_EEG.drop(["A1", "A2"], axis=1, inplace=True)
        return Av_data_EEG

    def _read_data_with_EMG(self, data):
        """Read EEG data with electromyogram (EMG)"""
        Av_data_EEG = self._read_data_without_EMG(data)

        data = data[26:30, :]
        b, a = signal.butter(8, [0.1, 0.24], "bandpass")
        filteddata = signal.filtfilt(b, a, data)

        columns = ["X1", "X2", "X3", "X4"]
        Ov_data_EMG = pd.DataFrame(filteddata.T, columns=columns)
        Av_data = pd.concat([Av_data_EEG, Ov_data_EMG], axis=1)
        # Av_data = Ov_data_EMG
        return Av_data

    def read_events(self):
        """ Read events and events_timeids from the edf file.
        Which kind of event will be read depends on the NAME,
        and different NAME corresponds different read_events_method()
        with different output event_timeinds.
        """
        raw_events = self.raw.find_edf_events()
        raw_events = raw_events[:, [0, 2]]
        if self.NAME == None:
            raise Exception("Choose a kind of event name!")
        elif self.NAME in ["S-BECT", "myoclonus"]:
            if self.NAME == "S-BECT":
                self.s_channel = self.filepath[-14:-12]
            event_timeinds, events_name_list = self._read_point_events(raw_events)
        elif self.NAME in ["spasm", "tonic"]:
            event_timeinds, events_name_list = self._read_interval_events(raw_events)
        # print(events_name_list)
        return event_timeinds, events_name_list

    def _read_point_events(self, raw_events):
        """ Read event method of 'S-BECT' and 'myoclonus'. """
        events_timeinds = []
        events_name_list = []
        for i in range(1, raw_events.shape[0]):
            if raw_events[i, 1][:7] == "Segment" or raw_events[i, 1] in ["A1+A2 OFF"]:
                continue
            if raw_events[i, 1][0] == "+":
                events_timeinds.append(float(raw_events[i, 1]))
                events_name_list.append(self.NAME)
            if raw_events[i - 1, 1][0] != "+" and raw_events[i, 1] == self.NAME:
                events_timeinds.append(float(raw_events[i, 0]))
                events_name_list.append(self.NAME)
        events_timeinds = np.array(events_timeinds) * 1000
        return events_timeinds, events_name_list

    def _read_interval_events(self, raw_events):
        """Read event method of 'spasm' and 'tonic'. """
        events_timeinds = []
        events_name_list = []
        for i in range(1, raw_events.shape[0]):
            if raw_events[i, 1][:7] == "Segment" or raw_events[i, 1] in ["A1+A2 OFF"]:
                continue
            if i >= 3:
                if raw_events[i, 1] == "end" and raw_events[i - 2, 1][0] != self.NAME:
                    events_timeinds.append(float(raw_events[i - 3, 0]))
                    events_timeinds.append(float(raw_events[i - 1, 0]))
                    events_name_list.append(self.NAME)
                    events_name_list.append("end")
        events_timeinds = np.array(events_timeinds) * 1000
        return events_timeinds, events_name_list

    def saved_as_csv(self, path):
        self._insert_label()
        if self.NAME in ["S-BECT"]:
            csv_name = self.filepath[-14:-3] + "csv"
        else:
            csv_name = self.filename[:8] + "csv"
        filename = path + '\\' + self.NAME
        root = os.getcwd()
        path = root + "\\" + filename
        if not os.path.exists(path):
            os.makedirs(path)
            print("<" + path + "> Build succeed")
        self.data_labeled.to_csv(path + "\\" + csv_name, header=True, index=False)
        print("<" + csv_name + "> has been saved in " + path)
        return None

    def _insert_label(self):
        L = self.Avdata.shape[0]
        label = np.zeros(L)
        if self.NAME in ["spasm", "tonic"]:
            for i in range(0, len(self.event_times) - 1, 2):
                label[int(self.event_times[i]) : int(self.event_times[i + 1])] = 1
        if self.NAME in ["S-BECT", "myoclonus"]:
            timeinds = [int(timeind) for timeind in self.event_times]
            # label[timeinds] = 1
        self.data_labeled = self.Avdata.copy()
        self.data_labeled["label"] = label
        return None

    def plot_data(self, time_slice=None, ratio=1, placement=120):
        """Show the EEG data in a simple way, you just need to choose a interval
        with time_scile=[start_time, end_time].
        """
        if time_slice == None:
            raise Exception("Choose a time interval!")

        N, n = self.Avdata.shape
        start_time, end_time = max(time_slice[0], 0), min(time_slice[1], N - 1)

        plt.figure(figsize=[20, 15])

        i = 0
        for col in list(self.Avdata.columns[::-1]):
            y = self.Avdata[col][start_time:end_time]
            color = "b"
            linew = 0.5
            if self.NAME == "S-BECT" and col[:2] == self.s_channel:
                color = "black"
                linew = 1
            plt.plot(y.index.values, y.values * ratio + i * placement, c=color, Linewidth=linew)
            i += 1

        inds = np.where(((self.event_times >= start_time) + 0 + (self.event_times <= end_time)) >= 2)[0]
        if len(inds):
            for event, name in zip(self.event_times[inds], np.array(self.event_names)[inds]):
                plt.axvline(event, c="r")
                plt.text(event + 15, -80, s=name, c="r", fontdict={"size": 14})
        plt.yticks(np.arange(n) * placement, self.Avdata.columns[::-1], rotation=45)
        plt.tick_params(labelsize=14)
        plt.xlim(start_time, end_time)
        title = ("<"+ self.filename[:7]+ "> EEG slice with time interval ("+ str(start_time)+ ","+ str(end_time)+ ")")
        plt.title(title, fontdict={"size": 16})
        plt.savefig("plot_data\\" + self.filename + ".png")
        plt.show()
        return None

