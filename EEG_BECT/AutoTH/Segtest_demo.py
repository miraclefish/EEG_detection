import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from SignalSegNet import SignalSegNet, Basicblock, Bottleneck
from torch.utils.data import DataLoader
from Dataset import BECTDataset

def test_demo(dataset_name, epoch):

    assert dataset_name in ['train', 'test']

    model_root = './Segmodel'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    """load data"""
    if dataset_name == 'train':
        dataset = BECTDataset(DataPath='./OrigData', FeaturePath='./MaskLabel', LabelFile='GT_info.csv', type="train", withData=True)
    elif dataset_name == 'test':
        dataset = BECTDataset(DataPath='./OrigData', FeaturePath='./MaskLabel', LabelFile='GT_info.csv', type="test", withData=True)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    net = SignalSegNet(Basicblock, [2,2,2,2,2])
    checkpoint = torch.load(os.path.join(model_root, 'model_epoch_'+str(epoch)+'.pth.tar'))
    net.load_state_dict(checkpoint['state_dict'])

    net = net.eval()

    net = net.to(device)

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    label_total = np.zeros(len(dataset))
    output_total = np.zeros(len(dataset))
    batch_size = dataloader.batch_size

    while i<len_dataloader:
        data = data_iter.next()
        
        x = data['Data'].float()
        x = torch.unsqueeze(x, dim=1)
        label = data['MaskLabel'].long()
        

        x = x.to(device)
        label = label.to(device)
        
        output = net(x=x)
        pred = torch.argmax(output, dim=1)

        swi_pred = torch.mean(pred.float(), dim=1)
        swi_label = torch.mean(label.float(), dim=1)

        if i<len_dataloader-1:
            label_total[i*batch_size:(i+1)*batch_size] = swi_label.cpu().data.numpy()
            output_total[i*batch_size:(i+1)*batch_size] = swi_pred.cpu().data.numpy()
        else:
            label_total[i*batch_size:] = swi_label.cpu().data.numpy()
            output_total[i*batch_size:] = swi_pred.cpu().data.numpy()
        
        i += 1

    fig, ax = plt.subplots(figsize=(8,8))

    ax.plot(label_total, label='Label')
    ax.plot(output_total, label='Pred')
    # ax.set_title('Test_demo with MSE = {:.2f}'.format(mse), fontsize=18)
    ax.set_title('{}_demo'.format(dataset_name))
    ax.set_xlabel('Samples', fontsize=18)
    ax.set_ylabel('swi', fontsize=18)
    ax.legend()

    # print('epoch: %d, loss of the %s dataset: %f' % (epoch, dataset_name, mse))

    plt.show()

    return None

if __name__ == "__main__":
    test_demo(dataset_name='test', epoch=0)
    pass