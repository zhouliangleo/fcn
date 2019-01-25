#!/usr/bin/python
#coding=utf8
"""
# Author: wei
# Created Time : 2018-09-27 11:19:50

# File Name: loss.py
# Description:

"""
import torch.nn.functional as F
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.beta = 1e8

    def forward(self, map_gt, count_gt, map_pre, num):
        map_loss = self.beta*F.mse_loss(map_pre, map_gt)
        rude_count = torch.sum(map_pre, dim=(2, 3))
        # bias = bias.squeeze()
        # print(num.item(), rude_count.item(), count_gt.item())
        count_pre = num+rude_count
        count_loss = F.smooth_l1_loss(count_pre, count_gt)
        # print(count_pre.item(), count_gt.item())
        # print(map_loss.item(), count_loss.item())
        loss = map_loss + count_loss
        return loss, map_loss, count_loss
