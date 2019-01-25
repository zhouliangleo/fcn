#!/usr/bin/python
#coding=utf8
"""
# Author: wei
# Created Time : 2018-09-26 16:19:35

# File Name: fcn.py
# Description:

"""
from torchvision import models
import torch.nn as nn
import torch
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        return l1, l2, l3

class Deconv(nn.Module):
    def __init__(self, ch1, ch2, pad=True):
        super(Deconv, self).__init__()
        if pad:
            self.deconv = nn.ConvTranspose2d(in_channels=ch1, out_channels=ch2, kernel_size=2, stride=2, bias=False)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels=ch1, out_channels=ch2, kernel_size=2, padding=(1,0), stride=2, output_padding=(1,0), bias=False)
        self.conv_combine = nn.Conv2d(in_channels=ch2, out_channels=ch2, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=ch2)
        self.relu = nn.ReLU(True)
        self.conv_1 = nn.Conv2d(in_channels=ch2, out_channels=ch2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=ch2)
    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        out = self.conv_combine((x1 + x2))
        out = self.relu(self.bn1(out))
        out = self.bn2(self.conv_1(out))
        return out

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.resnet = ResNet()
        self.deconv1 = Deconv(ch1=1024, ch2=512, pad=True)
        self.deconv2 = Deconv(ch1=512, ch2=256, pad=True)

        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=2, stride=1, bias=False)

        self.conv_to_map = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1, stride=1, bias=False)
        self.fc = nn.Linear(960*560*3, 1024)
        self.relu4 = nn.ReLU(True)
        self.fc_bias = nn.Linear(1024, 1)

    def forward(self, x):
        l1, l2, l3 = self.resnet(x)

        l2_up = self.deconv1(l3, l2)
        l1_up = self.deconv2(l2_up, l1)
        de_l1 = self.deconv3(l1_up)
        last_map = self.conv(self.relu(self.bn(de_l1)))

        dense_map = self.conv_to_map(last_map)

        fc = last_map.view(last_map.size()[0], -1)
        bias = self.fc_bias(self.relu4(self.fc(fc)))
        return dense_map, bias

if __name__ == '__main__':
    net = FCN().cuda()
    from torch.autograd import Variable
    a = Variable(torch.randn([1,3,320,640]).cuda())
    dense_map, bias = net(a)
    print(dense_map.size(), bias)
