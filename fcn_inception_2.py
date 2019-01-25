#!/usr/bin/python
#coding=utf8
"""
# Author: wei
# Created Time : 2018-09-28 15:41:59

# File Name: fcn_inception.py
# Description:

"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Inception_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inception_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, bias=False)
        self.conv3_3 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                kernel_size=3, padding=1, bias=False)
        self.conv5_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, bias=False)
        self.conv5_5 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                kernel_size=5, padding=2, bias=False)
        self.conv7_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, bias=False)
        self.conv7_7 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        x_1 = F.relu(self.conv1(x), inplace=True)
        x_3 = F.relu(self.conv3_1(x), inplace=True)
        x_3 = F.relu(self.conv3_3(x_3), inplace=True)
        x_5 = F.relu(self.conv5_1(x), inplace=True)
        x_5 = F.relu(self.conv5_5(x_5), inplace=True)
        x_7 = F.relu(self.conv7_1(x), inplace=True)
        x_7 = F.relu(self.conv7_7(x_7), inplace=True)
        output = torch.cat([x_1, x_3, x_5, x_7], dim=1)
        return output

class Deconv(nn.Module):
    def __init__(self, in_ch, out_ch, k_s):
        super(Deconv, self).__init__()
        pad_size = int((k_s-1)/2)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                kernel_size=k_s, padding=pad_size, bias=False)
        self.deconv = nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch,
                kernel_size=2, stride=2, bias=False)
    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)
        x = F.relu(self.deconv(x), inplace=True)
        return x

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.block1 = Inception_Block(3, 16)
        self.block2 = Inception_Block(64, 32)
        self.block3 = Inception_Block(128, 32)
        self.block4 = Inception_Block(128, 16)

        self.deconv1 = Deconv(64, 64, 9)
        self.deconv2 = Deconv(64, 32, 7)
        self.deconv3 = Deconv(32, 16, 5)

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16,
                kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=3,
                kernel_size=5, padding=2, bias=False)
        self.conv_map_pre = nn.Conv2d(in_channels=3, out_channels=1,
                kernel_size=3, padding=1, bias=False)
        self.conv_bias = nn.Conv2d(in_channels=3, out_channels=1,
                kernel_size=3, padding=1, bias=False)
        self.fc1 = nn.Linear(in_features=960*560, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=1, bias=False)
    def forward(self, x):
        x = F.max_pool2d(self.block1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.block2(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.block3(x), kernel_size=2, stride=2)
        x = self.block4(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)
        map_pre = F.relu(self.conv_map_pre(x), inplace=True)
        bias_map = F.relu(self.conv_bias(x), inplace=True)
        bias_pre = self.fc2(F.relu(self.fc1(bias_map.view(bias_map.size()[0], -1))))
        return map_pre, bias_pre

if __name__ == "__main__":
    net = FCN()
    ts = (torch.randn([1, 3, 560, 960]).cuda())
    net.cuda()
    out, bias = net(ts)
    print(out.size(), bias.size())
    input('Please Enter')
