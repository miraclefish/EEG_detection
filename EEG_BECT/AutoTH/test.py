import os
import torch
import numpy as np
from AutoTHNet import AutoTHNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Dataset import BECTDataset

def test(dataset_name, epoch):

    assert dataset_name in ['train', 'test']

    model_root = './model'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 60

    """load data"""
    if dataset_name == 'train':
        dataset = BECTDataset(DataPath='./FiltedData', FeaturePath='./HistFeature', LabelFile='GT_info.csv', type="train", withData=True)
    elif dataset_name == 'test':
        dataset = BECTDataset(DataPath='./FiltedData', FeaturePath='./HistFeature', LabelFile='GT_info.csv', type="test", withData=True)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    "testing"
    histFeatureSize = dataset[0]['Feature'].shape[0]
    dataSize = dataset[0]['Data'].shape[0]

    net = AutoTHNet(maxPoolSize=histFeatureSize, avgPoolSize=dataSize)
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
        
        x_data = data['Data'].float()
        x_feature = data['Feature'].float()
        label = data['label'].float()
        label = label.unsqueeze(dim=1)

        x_data = x_data.to(device)
        x_feature = x_feature.to(device)
        label = label.to(device)
        
        output, chosen_mask, th = net(x_data=x_data, x_feature=x_feature)

        if i == len_dataloader-1:
            label_total[i*batch_size:] = label.cpu().data.numpy().squeeze()
            output_total[i*batch_size:] = output.cpu().data.numpy().squeeze()
        else:
            label_total[i*batch_size:(i+1)*batch_size] = label.cpu().data.numpy().squeeze()
            output_total[i*batch_size:(i+1)*batch_size] = output.cpu().data.numpy().squeeze()

        i += 1
    
    mse = np.mean((label_total-output_total)**2)
    print('epoch: %d, loss of the %s dataset: %f' % (epoch, dataset_name, mse))

    return mse
