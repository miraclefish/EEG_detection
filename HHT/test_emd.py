import numpy as np
import matplotlib.pyplot as plt
from emd import *

from DataLoader import Dataloader

# data = Dataloader(batchsize=32)
# train_X, _ = data.getTrainData()
X = np.load("X.npy")
Y = np.load("Y.npy")
x1 = X[0,:,3]

sampling_rate=1000

dt = 1.0/sampling_rate

imfs = emd(x1)
n = len(imfs)
nrows = math.ceil((n+1)/2)

plt.figure(figsize=[20,15])
plt.subplot(nrows, 2, 1)
plt.plot(x1)
plt.title("X")

for i in range(len(imfs)):
    plt.subplot(nrows, 2, i+2)
    plt.plot(imfs[i])
    plt.title("Imf"+str(i+1))

plt.show()

pass

hh = hilbert_transform(imfs, dt)

plt.imshow(hh.T, cmap=plt.cm.hot)
plt.show()

pass

