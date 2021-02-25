import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from SignalSegNet import SignalSegNet, Basicblock, Bottleneck
from torch.utils.data import DataLoader
from Dataset import BECTDataset

def find_index_pair(label):
    d_label = label[1:] - label[:-1]
    index_S = np.where(d_label==1)[0]
    index_E = np.where(d_label==-1)[0]

    if index_E[0] < index_S[0]:
        index_S.insert(0, -1)
    elif index_S[-1] > index_E[-1]:
        index_E.append(len(d_label))
    
    assert len(index_S)==len(index_E)
    index_pair = [np.array([[start, end]]) for start, end in zip(index_S, index_E)]
    index_pair = np.concatenate(index_pair, axis=0)
    index_pair = index_pair + 1
    return index_pair

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
    data = x.cpu().data.numpy()
    label = label.cpu().data.numpy()

    net = net.to(device)
    x = x.to(device)

    output = net(x=x)
    output = torch.argmax(output, dim=1)
    output = output.squeeze(dim=0)

    pred = output.cpu().data.numpy()

    # 寻找输出为1的起始index
    label_index_pair = find_index_pair(label)
    pred_index_pair = find_index_pair(pred)

    fig, ax = plt.subplots(figsize=(15,5))
    ax.subplot(3,1,1)
    ax.plot(data)
    ax.title("Original data")

    ax.subplot(3,1,2)
    ax.plot(data)
    for ind in label_index_pair:
        plt.plot(np.arange(ind[0], ind[1]), data[ind[0]:ind[1]],c="r")
    ax.title("BECT label segment")

    ax.subplot(3,1,3)
    ax.plot(data)
    for ind in pred_index_pair:
        plt.plot(np.arange(ind[0], ind[1]), data[ind[0]:ind[1]],c="g")
    ax.title("BECT pred segment")

    # print('epoch: %d, loss of the %s dataset: %f' % (epoch, dataset_name, mse))

    plt.show()

    return None

if __name__ == "__main__":
    net = inital_net(model_root='./Segmodel', epoch=0)
    test_demo(net=net, data_type='test', num=1)
    pass