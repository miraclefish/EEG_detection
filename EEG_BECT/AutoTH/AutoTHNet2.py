from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
import torch

def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):

    # 每个stage中扩展的倍数
    extension = 4

    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplane, midplane, stride)
        self.bn1 = nn.BatchNorm1d(midplane)
        self.conv2 = conv3x1(midplane, midplane)
        self.bn2 = nn.BatchNorm1d(midplane)
        self.conv3 = conv1x1(midplane, midplane*self.extension)
        self.bn3 = nn.BatchNorm1d(midplane*self.extension)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 残差数据
        residual = x

        # 卷积
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        # 是否直连（如果是Identity block就直连，如果是conv block就对残差边进行卷积）
        if self.downsample != None:
            residual = self.downsample(x)

        # 相加
        out = out + residual
        out = self.relu(out)

        return out

class Basicblock(nn.Module):

    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Basicblock, self).__init__()

        self.conv1 = conv3x1(inplane, midplane, stride)
        self.bn1 = nn.BatchNorm1d(midplane)
        self.conv2 = conv3x1(midplane, midplane)
        self.bn2 = nn.BatchNorm1d(midplane)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 残差数据
        residual = x

        # 卷积
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        # 是否直连（如果是Identity block就直连，如果是conv block就对残差边进行卷积）
        if self.downsample != None:
            residual = self.downsample(x)

        # 相加
        out = out + residual
        out = self.relu(out)

        return out


class AutoTHNet2(nn.Module):

    def __init__(self, block, layers):

        self.inplane = 64

        super(AutoTHNet2, self).__init__()

        self.block = block
        self.layers = layers

        self.conv1 = nn.Conv1d(1, self.inplane, kernel_size=7, stride=3, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, padding=1, stride=2)

        # 64, 128, 256, 512是指扩大四倍之前的维度
        self.stage1 = self.make_layer(self.block, 64, self.layers[0], stride=2)
        self.stage2 = self.make_layer(self.block, 128, self.layers[1], stride=3)
        self.stage3 = self.make_layer(self.block, 256, self.layers[2], stride=3)
        self.stage4 = self.make_layer(self.block, 512, self.layers[3], stride=3)
        self.stage5 = self.make_layer(self.block, 1024, self.layers[4], stride=3)

        self.conv_final = nn.Conv1d(1024, 1024, kernel_size=5, padding=2, stride=3)
        self.bn_final = nn.BatchNorm1d(1024)
        self.relu_final = nn.ReLU()
        self.avgpool = nn.AvgPool1d(31)
        self.fc = nn.Linear(1024, 1)
        self.relu_out = nn.ReLU()


    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)# block
    
        out = self.maxpool(out)

        # block
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)

        # out = self.conv_final(out)
        # out = self.bn_final(out)
        # out = self.relu_final(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.relu_out(out)
        out = torch.squeeze(out)

        return out
    
    def make_layer(self, block, midplane, block_num, stride=1):
        '''
            block: block模块
            midplane: 每个模块中间的通道维数，一般等于输出维度/4
            block_num: 重复次数
            stride: Conv Block的步长
        '''
        block_list = []

        # 先确定要不要downsamlpe模块
        downsample=None
        if stride!=1 or self.inplane!=midplane:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplane, midplane, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm1d(midplane)
            )
        
        # Conv Block
        conv_block = block(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = midplane

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.inplane, midplane, stride=1))
        
        return nn.Sequential(*block_list)
        

if __name__ == "__main__":
    net = AutoTHNet2(Basicblock, [2,2,2,2,2])
    input = torch.randn([3,1,30014])
    output = net(x=input)
    print("参数数量：\n", sum(p.numel() for p in net.parameters() if p.requires_grad))
    pass