import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from SignalSegNet import SignalSegNet, Basicblock, Bottleneck
from torch.utils.data import DataLoader
from Dataset import BECTDataset

def inital_net(model_root, epoch=0):

    net = SignalSegNet(Basicblock, [2,2,2,2,2])
    checkpoint = torch.load(os.path.join(model_root, 'model_epoch_'+str(epoch)+'.pth.tar'))
    net.load_state_dict(checkpoint['state_dict'])
    net = net.eval()

    return net

def getdata(type, num):
    assert type in ['train', 'test']
    """load data"""
    if type == 'train':
        dataset = BECTDataset(DataPath='./OrigData', FeaturePath='./MaskLabel', LabelFile='GT_info.csv', type=type, withData=True)
    elif type == 'test':
        dataset = BECTDataset(DataPath='./OrigData', FeaturePath='./MaskLabel', LabelFile='GT_info.csv', type=type, withData=True)
    
    assert num < len(dataset)

    data = dataset[num]
    x = data['Data'].float()
    x = torch.unsqueeze(x, dim=0)
    x = torch.unsqueeze(x, dim=0)
    label = data['MaskLabel'].long()

    return x, label

def test_demo(net, data_type, num):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x, label = getdata(type=data_type, num=num)
    net = net.to(device)
    x = x.to(device)
    label = label.to(device)

    output = net(x=x)
    pred = torch.argmax(output, dim=1)
    pred = pred.squeeze(dim=0)

    fig, ax = plt.subplots(figsize=(15,5))

    ax.plot(label.cpu().data.numpy(), label='Label')
    ax.plot(pred.cpu().data.numpy(), label='Pred')
    # ax.set_title('Test_demo with MSE = {:.2f}'.format(mse), fontsize=18)
    ax.set_title('{}_demo'.format(data_type))
    ax.set_xlabel('Samples', fontsize=18)
    ax.set_ylabel('spike', fontsize=18)
    ax.legend()

    # print('epoch: %d, loss of the %s dataset: %f' % (epoch, dataset_name, mse))

    plt.show()

    return None

if __name__ == "__main__":
    net = inital_net(model_root='./Segmodel', epoch=0)
    test_demo(net=net, data_type='test', num=1)
    pass