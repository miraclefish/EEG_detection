import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from AutoTHNet2 import AutoTHNet2, Basicblock
from torch.utils.data import DataLoader
from Dataset import BECTDataset

def test_demo(dataset_name, epoch):

    assert dataset_name in ['train', 'test']

    model_root = './model2'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 80

    """load data"""
    if dataset_name == 'train':
        dataset = BECTDataset(DataPath='./FiltedData', FeaturePath='./HistFeature', LabelFile='GT_info.csv', type="train", withData=True)
    elif dataset_name == 'test':
        dataset = BECTDataset(DataPath='./FiltedData', FeaturePath='./HistFeature', LabelFile='GT_info.csv', type="test", withData=True)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    net = AutoTHNet2(Basicblock, [2,2,2,2,2])
    checkpoint = torch.load(os.path.join(model_root, 'model_epoch_'+str(epoch)+'.pth.tar'))
    net.load_state_dict(checkpoint['state_dict'])

    net = net.eval()

    net = net.to(device)

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    label_total = np.zeros(len(dataset))
    output_total = np.zeros(len(dataset))

    while i<len_dataloader:
        data = data_iter.next()
        
        x = data['Data'].float()
        x = torch.unsqueeze(x, dim=1)
        label = data['label'].float()

        x = x.to(device)
        label = label.to(device)
        
        output = net(x=x)

        i += 1
    
    mse = np.mean((label.cpu().data.numpy()-output.cpu().data.numpy())**2)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(label.cpu().data.numpy(), label='Label')
    ax.plot(output.cpu().data.numpy(), label='Pred')
    # ax.set_title('Test_demo with MSE = {:.2f}'.format(mse), fontsize=18)
    ax.set_title('{}_demo'.format(dataset_name))
    ax.set_xlabel('Samples', fontsize=18)
    ax.set_ylabel('threshold', fontsize=18)
    ax.legend()

    print('epoch: %d, loss of the %s dataset: %f' % (epoch, dataset_name, mse))

    plt.show()

    return mse

if __name__ == "__main__":
    mse = test_demo(dataset_name='test', epoch=47)
    pass