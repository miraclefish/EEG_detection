from __future__ import division, print_function, absolute_import
from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal



from keras.models import Model, load_model, Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

def model(input_shape, cnn_filters, gru_units):
    X_input = Input(shape=input_shape)
    #     # Step 2: First GRU Layer (≈4 lines)
    X = Conv1D(filters=cnn_filters, kernel_size=16, strides=1, padding='same')(X_input)  # CONV1D
    X = BatchNormalization()(X)  # Batch normalization
    X = Activation('relu')(X)  # ReLu activation
    #     X = Dropout(0.9)(X)
    X = GRU(units=gru_units, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    #     X = Dropout(0.9)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units=gru_units, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    #     X = Dropout(0.9)(X)                                # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    #     X = Dropout(0.9)(X)                                 # dropout (use 0.8)
    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)
    ### END CODE HERE ###
    model = Model(inputs=X_input, outputs=X)
    return model

def seperate_TT_data(X,Y, ratio=0.6):
    num = X.shape[0]
    train_X = X[:int(num*ratio)]
    train_Y = Y[:int(num*ratio)]
    val_X = X[int(num*ratio):int(num*(ratio+0.2))]
    val_Y = Y[int(num*ratio):int(num*(ratio+0.2))]
    test_X = X[int(num*(ratio+0.2)):]
    test_Y = Y[int(num*(ratio+0.2)):]
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def PRandFscore(pred_Y, Y, threshold=0.95):
    pred_Y = np.array([pred_Y.reshape(-1) > threshold]).astype(np.int).squeeze()
    Y = Y.reshape(-1).astype(np.int)
    confuison_matrix = confusion_matrix(pred_Y, Y)
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
    return precision,recall,Fscore,confuison_matrix


def train(X, Y, split_length, strides, cnn_filters, gru_units):
    num, T, c = X.shape
    train_X, train_Y, val_X, val_Y, test_X, test_Y = seperate_TT_data(X, Y)
    models = model(input_shape=(T, c), cnn_filters=cnn_filters, gru_units=gru_units)
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    models.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = models.fit(train_X, train_Y, validation_data=(val_X, val_Y), batch_size=64, epochs=5, shuffle=True, verbose=2)
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    acc = history.history['acc']
    pred_train = models.predict(train_X)
    pred_test = models.predict(test_X)
    test_loss, test_acc = models.evaluate(test_X, test_Y)
    precisiontr, recalltr, Fscoretr, confuison_matrixtr = PRandFscore(pred_train, train_Y)
    precisionte, recallte, Fscorete, confuison_matrixte = PRandFscore(pred_test, test_Y)

    log = {'loss': loss, 'acc': acc, 'val_loss': val_loss, 'val_acc': val_acc, 'train_loss': loss[-1], 'train_acc': acc[-1],
           'test_loss': test_loss, 'test_acc': test_acc, 'precisiontr': precisiontr, 'recalltr':recalltr, 'Fscoretr': Fscoretr,
           'cmtr': confuison_matrixtr, 'precisionte': precisionte, 'recallte': recallte, 'Fscorete': Fscorete, 'cmte': confuison_matrixte}

    logpath = str(num)+'_'+str(T)+'_'+str(c)+'_spl='+str(split_length)+'_str='+str(strides)+'_cnnf='+str(cnn_filters)+'_gs='+str(gru_units)
    path = 'log\\'+logpath
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+'\\log.npy', log)
    models.save(path+'\\model.h5')
    print(logpath+':  Precosion='+str(precisionte)+', recall='+str(recallte)+', Fscore='+str(Fscorete))
    return None


orig_data, spindex, szindex = load_data_with_index(sample_length=5000, strides=500)
N_data, P_data = balanceNP(orig_data)
print('N_data num:', len(N_data), '; P_data num', len(P_data), '; Total num:', len(N_data)+len(P_data))

split_length = [200, 400, 600]
strides = [80, 100, 120]
cnn_filters = [16, 32, 64]
gru_units = [32, 64, 128]

for i in range(len(strides)):
    AC_N_data = AC_represent(N_data, split_length=split_length[i], strides=strides[i])
    AC_P_data = AC_represent(P_data, split_length=split_length[i], strides=strides[i])
    AC_data = AC_N_data + AC_P_data
    X, Y = prepare_XY(AC_data)
    for cnn in cnn_filters:
        for gru in gru_units:
            train(X, Y, split_length=split_length[i], strides=strides[i], cnn_filters=cnn, gru_units=gru)