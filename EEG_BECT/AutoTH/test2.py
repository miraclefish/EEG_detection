import os
import torch
import numpy as np
from AutoTHNet2 import AutoTHNet2, Basicblock, Bottleneck
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from Dataset import BECTDataset

def test(dataset_name, epoch):

    assert dataset_name in ['train', 'test']

    model_root = './model3'

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    """load data"""
    if dataset_name == 'train':
        dataset = BECTDataset(DataPath='./OrigData', FeaturePath='./HistFeature', LabelFile='GT_info.csv', type="train", withData=True)
    elif dataset_name == 'test':
        dataset = BECTDataset(DataPath='./OrigData', FeaturePath='./HistFeature', LabelFile='GT_info.csv', type="test", withData=True)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    "testing"

    net = AutoTHNet2(Basicblock, [2,2,2,2,2])
    checkpoint = torch.load(os.path.join(model_root, 'model_epoch_'+str(epoch)+'.pth.tar'))
    net.load_state_dict(checkpoint['state_dict'])

    net = net.eval()

    net = net.to(device)

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    loss_all = 0
    loss = nn.MSELoss()

    while i<len_dataloader:
        data = data_iter.next()
        
        x = data['Data'].float()
        label = data['label'].float()

        x = x.unsqueeze(dim=1)
        x = x.to(device)
        label = label.to(device)
        
        output = net(x=x)

        loss_all += loss(output, label)

        i += 1
    
    loss_all = loss_all/len_dataloader
    loss_all = loss_all.cpu().data.numpy()
    print('epoch: %d, loss of the %s dataset: %f' % (epoch, dataset_name, loss_all))
    # print('chosenMask:', chosen_mask.cpu().data.numpy())
    # matrix = output.cpu().data.numpy()

    return loss_all
