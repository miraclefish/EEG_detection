import numpy as np
import tensorflow as tf
from DataLoader import Dataloader


data = Dataloader(batchsize=32)
train_X, train_Y = data.getTrainData()
test_X, test_Y = data.getTestData()
valid_X, valid_Y = data.getValiData()
label_num = train_Y.shape[0]*train_Y.shape[1] + test_Y.shape[0]*test_Y.shape[1] + valid_Y.shape[0]*valid_Y.shape[1]
label_1_num = np.sum(train_Y) + np.sum(test_Y) + np.sum(valid_Y)
pass