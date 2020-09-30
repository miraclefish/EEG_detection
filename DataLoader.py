import numpy as np
from Dataset import Dataset


class Dataloader:
    def __init__(self, batchsize, fft_process=False, norm=False):
        self.fft_process = fft_process
        self.norm = norm
        self.only_P = True
        self.data = self._load_data()
        self.Num = self.data[0].shape[0]
        self.test_ratio = 0.3
        self.batchsize = batchsize
        self.i = 0
        self.train_data, self.test = self._Data_partation()
        self.train, self.valid = self._Vali_from_train()
        

    def _load_data(self, shuffle=True):
        dataset = Dataset(NAME="spasm", split_length=5000)
        data_X, data_Y = dataset.getData(fft_process=self.fft_process, norm=self.norm, only_P=self.only_P)
        if shuffle:
            index = np.arange(data_Y.shape[0])
            np.random.shuffle(index)
            data_X = data_X[index]
            data_Y = data_Y[index]
        return data_X, data_Y

    def _Data_partation(self):
        dataX, dataY = self.data
        train_X = dataX[: -int(self.Num * self.test_ratio)]
        train_Y = dataY[: -int(self.Num * self.test_ratio)]
        test_X = dataX[-int(self.Num * self.test_ratio) :]
        test_Y = dataY[-int(self.Num * self.test_ratio) :]
        return (train_X, train_Y), (test_X, test_Y)

    def _Vali_from_train(self):
        data_X, data_Y = self.train_data
        N = data_Y.shape[0]
        vali_ratio = 0.1
        vali_num = int(self.Num * vali_ratio)
        n = np.random.choice(range(int(self.Num * (1 - vali_ratio - self.test_ratio))))
        train_index = list(np.arange(n)) + list(np.arange(n + vali_num, N))
        valid_index = list(np.arange(n, n + vali_num))
        train_X = data_X[train_index]
        train_Y = data_Y[train_index]
        valid_X = data_X[valid_index]
        valid_Y = data_Y[valid_index]
        return (train_X, train_Y), (valid_X, valid_Y)

    def refresh_tra_val(self):
        self.train, self.valid = self._Vali_from_train()
        return None

    def getTrainData(self):
        train_X, train_Y = self.train
        return train_X, train_Y

    def getTestData(self):
        test_X, test_Y = self.test
        return test_X, test_Y

    def getValiData(self):
        valid_X, valid_Y = self.valid
        return valid_X, valid_Y

    def nextBatch(self):
        train_X, train_Y = self.train
        N = train_Y.shape[0]
        if self.i < N / self.batchsize:
            if self.i + 1 >= N / self.batchsize:
                end = N
            else:
                end = (self.i + 1) * self.batchsize
            batch_X = train_X[self.i * self.batchsize : end]
            batch_Y = train_Y[self.i * self.batchsize : end]
            self.i = self.i + 1
        else:
            self.i = 0
            raise Exception("All batches has been used in this epoch!")
        return batch_X, batch_Y
