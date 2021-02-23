import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class BECTDataset(Dataset):
    """BECT with SWI labeled data"""

    def __init__(self, DataPath, FeaturePath, LabelFile, type=None, withData=False):
        """
        Args:
            DataPath(string): Path to the txt file of the filted data;
            FeaturePath(string): Path to the txt file of the spike point distribution features;
            LabelFile(string): File of the ground truth information.
        """
        if type == None:
            self.DataPath = DataPath
            self.FeaturePath = FeaturePath
        else:
            self.DataPath = os.path.join(DataPath, type)
            self.FeaturePath = os.path.join(FeaturePath, type)
        self.withData = withData
        # self.correct_factor = -0.45
        self.LabelInfo = pd.read_csv(LabelFile, sep='\t', index_col=0)
        self.DataFileList = self._get_file_list(self.DataPath)
        
    def __len__(self):
        return len(self.DataFileList)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.DataPath, self.DataFileList[idx])
        feature_path = os.path.join(self.FeaturePath, self.DataFileList[idx])
        HistFeature = np.loadtxt(feature_path)
        label = self.LabelInfo['threshold'][self.DataFileList[idx][:-4]]
        sample = {"MaskLabel": torch.from_numpy(HistFeature)}
        # sample = {"label": torch.from_numpy(np.array(label))}
        # sample = {"Feature": torch.from_numpy(HistFeature[:,:-1]), "label": torch.from_numpy(np.array(label))}
        # sample = {"Feature": torch.from_numpy(HistFeature[:,:-1]), "label": torch.from_numpy(np.array(label)), "mask_label": torch.from_numpy(HistFeature[:,-1])}
        if self.withData == True:
            data = np.loadtxt(data_path)
            # data = np.sign(data)*np.log(np.abs(data)+1)
            data = (data - np.mean(data))/np.std(data)
            sample["Data"] = torch.from_numpy(data)

        return sample

    def _get_file_list(self, DataPath):
        return os.listdir(self.DataPath)

if __name__ == "__main__":
    dataset = BECTDataset(DataPath='./OrigData', FeaturePath='./MaskLabel', LabelFile='GT_info.csv', type="train", withData=True)
    a = dataset[0]
    pass