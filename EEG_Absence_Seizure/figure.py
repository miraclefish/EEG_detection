from utils import *
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import Model, load_model, Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

def PRandFscore(pred_Y, Y, threshold=0.95):
    pred_Y = np.array([pred_Y.reshape(-1) > threshold]).astype(np.int).squeeze()
    Y = Y.reshape(-1).astype(np.int)
    # confuison_matrix = confusion_matrix(pred_Y, Y)
    N, P = np.bincount(Y)
    PN, PP = np.bincount(pred_Y)
    KK = sum(Y == pred_Y)
    TP = int((KK - PN + P) / 2)
    TN = int((KK - PP + N) / 2)
    FP = P - TP
    FN = N - TN
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    Fscore = 2*(precision*recall)/(precision+recall)
    return precision,recall,Fscore


model = load_model('log\\1573_47_399_spl=400_str=100_cnnf=64_gs=32\\model.h5')

# original_data, splits_index, sz_index = load_data_with_index(sample_length=5000, strides=500)
# N_sample, P_sample = seperate_NP(original_data)
# AC_N = AC_represent(N_sample, split_length=400, strides=100)
# AC_P = AC_represent(P_sample, split_length=400, strides=100)
# AC_data = AC_N+AC_P
# X, Y = prepare_XY(AC_data)
X, Y = np.load('X.npy'), np.load('Y.npy')
pred = model.predict(X)

gates = np.linspace(0.01,0.99,99)
pre = []
re = []
for k in gates:
    precision, recall, Fscore = PRandFscore(pred, Y, threshold=k)
    pre.append(precision)
    re.append(recall)
pre = np.array(pre)
re = np.array(re)
dis = np.sqrt((1-pre)**2+(1-re)**2)
index = np.argmin(dis)
s1 = '('+'%.2f'%pre[index]+','+'%.2f'%re[index]+')'
s2 = 'Threshold='+'%.2f'%gates[index]
plt.figure()
plt.scatter(pre, re, c='#1f4e5f', linewidths=2)
plt.plot(pre[index], re[index], c='#ff7473', marker='o', markersize=8)

plt.text(pre[index]-0.05, re[index]-0.05, s1, fontdict={'size': 10})
plt.text(pre[index]-0.05, re[index]-0.09, s2, fontdict={'size': 10})
plt.xlabel('Precision')
plt.ylabel('recall')

# filenames = os.listdir('log')
# for filename in filenames:
#     log = np.load('log\\'+filename+'\\log.npy').item()
#     print(filename)
#     print('tracc=','%.3f'%log['train_acc'],
#           'teacc=', '%.3f'%log['test_acc'],
#           'pre=', '%.3f'%log['precisionte'],
#           'recall=', '%.3f'%log['recallte'],
#           'F=', '%.3f'%log['Fscorete'])
#     print()




# l = 200
# original_data, splits_index, sz_index = load_data_with_index(sample_length=5000, strides=500)
# N_sample, P_sample = seperate_NP(original_data)
# AC_N = AC_represent(N_sample, split_length=l, strides=l)
# AC_P = AC_represent(P_sample, split_length=l, strides=l)
# N = AC_N[0][0].shape[1]
#
#
# plt.figure(figsize=[12,4])
# plt.subplot(121)
# plt.scatter(np.arange(N), AC_N[10][0][10,:], c='#1f4e5f')
# plt.axhline(0, c='#79a8a9')
# plt.title('Seizure Fragment')
# plt.xlabel('k (order of the ACF)')
# plt.ylabel('AC Function value')
# plt.subplot(122)
# plt.scatter(np.arange(N), AC_P[20][0][10,:], c='#1f4e5f')
# plt.title('Non-seizure Fragment')
# plt.xlabel('k (order of the ACF)')
# plt.ylabel('AC Function value')
# plt.axhline(0, c='#79a8a9')
# # plt.plot(AC_N[10][1])
# # plt.plot(AC_P[10][1])