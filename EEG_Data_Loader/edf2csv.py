import os
from EDFreader import EDFreader

# EDF_NAME = "spasm"
# EDF_NAME = "tonic"
# EDF_NAME = "myoclonus"
EDF_NAME = "S-BECT"

path = "test_data\\" + EDF_NAME
filenames = os.listdir(path)
n = 0
for filename in filenames:
    n += 1
    print("-----------file No." + str(n) + "------------")
    edf = EDFreader(path + "\\" + filename, NAME=EDF_NAME)
    event_times = edf.event_times
    event_names = edf.event_names
    if EDF_NAME in ["spasm", "tonic"]:
        for i in range(0, len(event_times) - 1, 2):
            print(event_names[i : i + 2], "--->", event_times[i : i + 2])
    if EDF_NAME in ["S-BECT", "myoclonus"]:
        for i in range(len(event_times)):
            print(event_names[i], "--->", int(event_times[i]))
    edf.saved_as_csv("test_csv_data")
    edf.plot_data([0, 5000])
pass
