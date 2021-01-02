from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
import torch

class BinarizedF(Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        a = torch.ones_like(input)
        b = torch.zeros_like(input)
        output = torch.where(input>0, a, b)
        return output
    
    @staticmethod
    def backward(self, output_grad):
        input, = self.saved_tensors
        input_abs = torch.abs(input - 0.5)
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        input_grad = torch.where(input_abs<=0.5, ones, zeros)
        return input_grad

class AutoTHNet(nn.Module):

    def __init__(self, maxPoolSize, avgPoolSize):
        super(AutoTHNet, self).__init__()
        self.pad1 = nn.ConstantPad1d(1, value=0)
        self.BN = BinarizedF()
        self.pool = nn.MaxPool1d(maxPoolSize, 1)
        self.minus_BN = BinarizedF()
        self.chosen_BN = BinarizedF()
        self.avgPool = nn.AvgPool1d(avgPoolSize, 1)

        self.interval_accurated = nn.Sequential()
        self.interval_accurated.add_module('linear1', nn.Linear(maxPoolSize, maxPoolSize))
        self.interval_accurated.add_module('sigmoid', nn.Sigmoid())

        self.interval_chosen = nn.Sequential()
        self.interval_chosen.add_module('linear2', nn.Linear(maxPoolSize, maxPoolSize))

    
    def forward(self, x_data, x_feature):
        
        peak_mask = self.peak_find(x_data)
        filted_x_data = torch.mul(peak_mask, x_data)
        filted_x_data = torch.unsqueeze(filted_x_data, dim=1)

        sigmoid = self.interval_accurated(x_feature[:,:,0])
        interval_accurated = sigmoid * (x_feature[:,:,2]-x_feature[:,:,1]) + x_feature[:,:,1]

        chosen_mask = self.interval_chosen(x_feature[:,:,0])
        chosen_mask = BinarizedF.apply(chosen_mask)
        chosen = torch.mul(chosen_mask, interval_accurated)
        chosen = torch.unsqueeze(chosen, dim=1)
        th = self.pool(chosen)

        cdf_BN = BinarizedF.apply(th - filted_x_data)
        output = self.avgPool(cdf_BN)
        output = output.squeeze(dim=1)

        return output, chosen_mask, th

    def peak_find(self, x_data):
        d_x_data = x_data[:,1:] - x_data[:,:-1]
        dd_x_data = -(d_x_data[:,1:] * d_x_data[:,:-1])
        dd_x_data = self.pad1(dd_x_data)
        bn = BinarizedF.apply(dd_x_data)
        return bn



if __name__ == "__main__":

    pass