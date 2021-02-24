import numpy as np
import torch
import matplotlib.pyplot as plt

from SignalSegNet import SignalSegNet, Basicblock, Bottleneck
from Dataset import BECTDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
from Segtest import test


# 初始化设定

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("<<<<<<<<Device: ", device," >>>>>>>>>>>")

lr = 1e-3
batch_size = 16
n_epoch = 100
model_root = "./Segmodel"

dataset_train = BECTDataset(DataPath='./OrigData', FeaturePath='./MaskLabel', LabelFile='GT_info.csv', type="train", withData=True)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

# 定义网络
writer = SummaryWriter('./runs4', flush_secs=1)

net = SignalSegNet(Basicblock, [2,2,2,2,2])

# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)
loss = nn.CrossEntropyLoss()


net = net.to(device)
loss = loss.to(device)

for epoch in range(n_epoch):
    data_train_iter = iter(dataloader_train)

    i = 0
        
    while i < len(dataloader_train):

        data_train = data_train_iter.next()
        x_data = data_train['Data'].float()
        label = data_train['MaskLabel'].long()
        
        net.zero_grad()

        x_data = x_data.unsqueeze(dim=1)
        x_data = x_data.to(device)
        label = label.to(device)

        output = net(x=x_data)
        Loss = loss(output, label)
        Loss.backward()
        optimizer.step()

        i += 1
        print('epoch: %d, [iter: %d / all %d], loss : %f' \
              % (epoch, i, len(dataloader_train), Loss.cpu().data.numpy()))
    
    torch.save({'state_dict': net.state_dict()},'{0}/model_epoch_{1}.pth.tar'.format(model_root, epoch))

    train_loss = test("train", epoch)
    test_loss = test("test", epoch)
    writer.add_scalars('Loss', {'train_loss': train_loss, 'test_loss':test_loss}, epoch)
    writer.close()
    # writer.flush()

pass