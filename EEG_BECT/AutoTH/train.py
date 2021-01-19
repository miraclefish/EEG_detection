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

def plot_chosen_mask(epoch, matirx_trian, matrix_test, train_loss, test_loss):
    fig = plt.figure(figsize=[8,16])
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(matirx_trian)
    ax1.set_title("epoch:{0:d}\n(train loss:{1:.5f})".format(epoch, train_loss))

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(matrix_test)
    ax2.set_title("epoch:{0:d}\n(test loss:{1:.5f})".format(epoch, test_loss))
    return fig

# 初始化设定

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("<<<<<<<<Device: ", device," >>>>>>>>>>>")

lr = 1e-3
batch_size = 60
n_epoch = 100
model_root = "./model"

dataset_train = BECTDataset(DataPath='./FiltedData', FeaturePath='./HistFeature', LabelFile='GT_info.csv', type="train", withData=True)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

# 定义网络
writer = SummaryWriter('./runs', flush_secs=1)

histFeatureSize = dataset_train[0]['Feature'].shape[0]
dataSize = dataset_train[0]['Data'].shape[0]

net = AutoTHNet(maxPoolSize=histFeatureSize, avgPoolSize=dataSize)

# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.01)
loss_cdf = nn.MSELoss()
loss_chosen = nn.CrossEntropyLoss()


net = net.to(device)
loss_cdf = loss_cdf.to(device)
loss_chosen = loss_chosen.to(device)

for epoch in range(n_epoch):
    data_train_iter = iter(dataloader_train)

    i = 0
        
    while i < len(dataloader_train):

        data_train = data_train_iter.next()
        x_data = data_train['Data'].float()
        x_feature = data_train['Feature'].float()
        label = data_train['label'].float()
        mask_label = data_train['mask_label'].float()
        mask_label = torch.where(mask_label == 1)[1]
        label = label.unsqueeze(dim=1)
        
        net.zero_grad()

        x_data = x_data.to(device)
        x_feature = x_feature.to(device)
        label = label.to(device)
        mask_label = mask_label.to(device)

        output, chosen, chosen_mask, th = net(x_data=x_data, x_feature=x_feature)
        loss = loss_chosen(chosen, mask_label)
        # loss = loss_cdf(output, label) + loss_chosen(chosen, mask_label)
        # loss = loss_cdf(output, label) + torch.mean(torch.sum((chosen_mask - mask_label)**2, axis=1))
        # loss = loss_cdf(output, label) + torch.mean(torch.abs(torch.log(torch.sum(torch.mul(chosen_mask, chosen_mask), axis=1))))
        loss.backward()
        optimizer.step()

        i += 1
        print('epoch: %d, [iter: %d / all %d], loss : %f' \
              % (epoch, i, len(dataloader_train), loss.cpu().data.numpy()))
    
    torch.save({'state_dict': net.state_dict()},'{0}/model_epoch_{1}.pth.tar'.format(model_root, epoch))

    train_loss, matrix_train = test("train", epoch)
    test_loss, matrix_test = test("test", epoch)
    writer.add_scalars('Loss', {'train_loss': train_loss, 'test_loss':test_loss}, epoch)
    writer.close()
    writer.add_figure('chosen_mask', plot_chosen_mask(epoch, matrix_train, matrix_test, train_loss, test_loss), epoch)
    writer.close()
    # writer.flush()

pass