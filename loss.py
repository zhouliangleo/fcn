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
        self.beta = 100.
    def forward(self, map_gt, count_gt, map_pre, bias):
        map_loss = F.mse_loss(map_pre, map_gt)
        axes = [1, 2, 3]
        rude_count = map_pre
        for axe in axes:
            rude_count = torch.sum(rude_count, dim=1)
        bias = bias.squeeze()
        #print(bias, rude_count, count_gt)
        count_pre = bias+rude_count
        count_loss = F.smooth_l1_loss(count_pre, count_gt)
        loss = map_loss + self.beta*count_loss
        return loss, map_loss, count_loss
