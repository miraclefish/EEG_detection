import numpy as np
import torch
import matplotlib.pyplot as plt

from AutoTHNet import AutoTHNet
from Dataset import BECTDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
from test import test

# 初始化设定

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("<<<<<<<<Device: ", device," >>>>>>>>>>>")

lr = 1e-2
batch_size = 32
n_epoch = 100
model_root = "./model"

dataset_train = BECTDataset(DataPath='./FiltedData', FeaturePath='./HistFeature', LabelFile='GT_info.csv', type="train", withData=True)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

# 定义网络
writer = SummaryWriter('./runs', flush_secs=1)

histFeatureSize = dataset_train[0]['Feature'].shape[0]
dataSize = dataset_train[0]['Data'].shape[0]

net = AutoTHNet(maxPoolSize=histFeatureSize, avgPoolSize=dataSize)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss_function = nn.MSELoss()


net = net.to(device)
loss = loss.to(device)

for epoch in range(n_epoch):
    data_train_iter = iter(dataloader_train)

    i = 0
        
    while i < len(dataloader_train):

        data_train = data_train_iter.next()
        x_data = data_train['Data'].float()
        x_feature = data_train['Feature'].float()
        label = data_train['label'].float()
        label = label.unsqueeze(dim=1)
        
        net.zero_grad()

        x_data = x_data.to(device)
        x_feature = x_feature.to(device)
        label = label.to(device)

        output, chosen_mask, th = net(x_data=x_data, x_feature=x_feature)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

        i += 1
        print('epoch: %d, [iter: %d / all %d], loss : %f' \
              % (epoch, i, len(dataloader_train), loss.cpu().data.numpy()))
    
    torch.save({'state_dict': net.state_dict()},'{0}/model_epoch_{1}.pth.tar'.format(model_root, epoch))

    loss_train = test("train", epoch)
    loss_test = test("test", epoch)
    writer.add_scalars('Loss', {'train_loss': loss_train, 'test_loss':loss_test}, epoch)
    writer.close()

pass