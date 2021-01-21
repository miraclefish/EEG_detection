import numpy as np
import torch
import matplotlib.pyplot as plt

from AutoTHNet2 import AutoTHNet2, Basicblock, Bottleneck
from Dataset import BECTDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
from test2 import test

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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("<<<<<<<<Device: ", device," >>>>>>>>>>>")

lr = 1e-3
batch_size = 80
n_epoch = 1000
model_root = "./model3"

dataset_train = BECTDataset(DataPath='./OrigData', FeaturePath='./HistFeature', LabelFile='GT_info.csv', type="train", withData=True)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

# 定义网络
writer = SummaryWriter('./runs3', flush_secs=1)

net = AutoTHNet2(Basicblock, [2,2,2,2,2])

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.99)
# loss = nn.CrossEntropyLoss()
loss_function = nn.MSELoss()


net = net.to(device)
loss_function = loss_function.to(device)

for epoch in range(n_epoch):
    data_train_iter = iter(dataloader_train)

    i = 0

    train_loss = 0
        
    while i < len(dataloader_train):

        data_train = data_train_iter.next()
        data = data_train['Data'].float()
        label = data_train['label'].float()
        
        net.zero_grad()

        data = data.to(device)
        label = label.to(device)
        # x_feature = x_feature.to(device)
        # mask_label = mask_label.to(device)

        data = data.unsqueeze(dim=1)
        output = net(x=data)
        loss = loss_function(output, label)
        # loss = loss_cdf(output, label) + loss_chosen(chosen, mask_label)
        # loss = loss_cdf(output, label) + torch.mean(torch.sum((chosen_mask - mask_label)**2, axis=1))
        # loss = loss_cdf(output, label) + torch.mean(torch.abs(torch.log(torch.sum(torch.mul(chosen_mask, chosen_mask), axis=1))))
        loss.backward()
        optimizer.step()
        scheduler.step()

        i += 1
        print('epoch: %d, [iter: %d / all %d], loss : %f, lr : %f' \
              % (epoch, i, len(dataloader_train), loss.cpu().data.numpy(), scheduler.get_lr()[0]))
        train_loss += loss.cpu().data.numpy()
    
    if epoch%5 == 0:
        torch.save({'state_dict': net.state_dict()},'{0}/model_epoch_{1}.pth.tar'.format(model_root, epoch))
        train_loss = train_loss/len(dataloader_train)
        print('epoch: %d, loss of the train dataset: %f' % (epoch, train_loss))
        # train_loss = test("train", epoch)
        test_loss = test("test", epoch)

        # train_loss, matrix_train = test("train", epoch)
        # test_loss, matrix_test = test("test", epoch)
        writer.add_scalars('Loss', {'train_loss': train_loss, 'test_loss':test_loss}, epoch)
        writer.close()
        # writer.add_figure('chosen_mask', plot_chosen_mask(epoch, matrix_train, matrix_test, train_loss, test_loss), epoch)
        # writer.close()
        # writer.flush()

pass