import numpy as np
from pandas import DataFrame, concat
from scipy import signal
from mne.io import read_raw_edf  # version of mne=0.17.1

"""
You can use "conda install mne==0.17.1" or "pip install mne==0.17.1" to install the library "mne".
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

    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = self.filepath[-14:-4]
        self.channel = self.filename[:2]
        self.raw = read_raw_edf(self.filepath, preload=True)
        self.OriginalData = self._read_original_data()

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
        original_data = self._data_sort(data)
        
        return original_data

    def _data_sort(self, data):
        
        columns = ["Fp1","Fp2","F3","F4","C3","C4",
                    "P3","P4","O1","O2","F7","F8",
                    "T3","T4","T5","T6","Fz","Cz",
                    "Pz"]

        eeg_signal = data[0:19, :]
        ref_signal = data[22:24, :]

        # EEG data filter
        eeg_filted = self._lowpass(HigHz=200, data=eeg_signal)
        ref_filted = self._lowpass(HigHz=200, data=ref_signal)

        # eeg_filted_Av = eeg_filted
        eeg_filted_Av = eeg_filted - np.mean(ref_filted, axis=0, keepdims=True)

        filted_data = -eeg_filted_Av
        sorted_data = DataFrame(filted_data.T, columns=columns)
        return sorted_data[self.channel]


    def _bandpass(self, lowHz, HigHz, data):
        lf = lowHz * 2.0 / 1000
        hf = HigHz * 2.0 / 1000
        N = 8
        b, a = signal.butter(N, [lf, hf], "bandpass")
        filted_data = signal.filtfilt(b, a, data)
        return filted_data

    def _lowpass(self, HigHz, data):
        hf = HigHz * 2.0 / 1000
        N = 8
        b, a = signal.butter(N, hf, "lowpass")
        filted_data = signal.filtfilt(b, a, data)
        return filted_data

    def save_txt(self, path):
        print('length: ', len(self.OriginalData))
        self.OriginalData.to_csv('./'+path+'/'+self.filename+'.txt', sep='\t', index=False)