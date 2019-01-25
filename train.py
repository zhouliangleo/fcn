#!/usr/bin/python
#coding=utf8
"""
# Author: wei
# Created Time : 2018-09-26 21:34:51

# File Name: train.py
# Description:

"""
import torch
from torch.autograd import Variable
from fcn_shallow import FCN
#from fcn_inception import FCN
#from fcn_inception_2 import FCN
from fcn_inception_smallMap import FCN
from dataloader import Datamaker
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import datetime, os
from loss import Loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

resume = False
model_name = None
lr = 1e-3
decay_epoch = 10
decay_rate = 0.1
max_epoch = 100
btch_size = 4
model_name = "fcn_shallow"

def train():
    #Data
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
    train_data = Datamaker(transform=transform, train=True)
    test_data = Datamaker(transform=transform, train=False)
    trainloader = DataLoader(train_data, batch_size=btch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(test_data, batch_size=btch_size, shuffle=True, num_workers=16)
    print("Data prepare done!")

    #Model
    net = FCN()
    if resume:
        net.load_state_dict(torch.load("./models/"+model_name))
    net = torch.nn.DataParallel(net, device_ids=[0,1])
    net.cuda()

    criterion = Loss()
    #criterion = nn.MSELoss(size_average=False)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    print("Net build done!")

    #Adjust learning rate
    def _adjust_lr(epoch, optim=optimizer, initial_lr=lr, decay_rate=0.1, decay_epoch=10):
        lr = initial_lr * (decay_rate**(epoch//decay_epoch))
        for param_group in optim.param_groups:
            param_group['lr']=lr
        return lr

    #Training
    best_loss = 1e5
    for epoch in range(max_epoch):
        train_loss = 0
        map_loss = 0
        count_loss = 0
        print("Training begin")
        net.train()
        for batch_idx, (inputs, map_gt, count_gt) in enumerate(trainloader):
            inputs = Variable(inputs.cuda())
            map_gt = Variable(map_gt.cuda())
            count_gt = Variable(count_gt.type(torch.FloatTensor).cuda())

            optimizer.zero_grad()
            map_pre, bias = net(inputs)
            #map_pre = net(inputs)
            loss, mp_lss, cnt_lss = criterion(map_gt, count_gt, map_pre, bias)
            #loss = 10*criterion(map_pre, map_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            map_loss += mp_lss.item()
            count_loss += cnt_lss.item()
            if (batch_idx+1)%20 == 0:
                print('train loss: %.3f, map_loss: %.3f, count_loss: %.3f, batch: %d/%d, epoch: %d'%(
                    train_loss/(batch_idx+1), map_loss/(batch_idx+1), count_loss/(batch_idx+1),
                    batch_idx+1, len(trainloader), epoch))
                train_loss = 0
                map_loss = 0
                count_loss = 0
            if (batch_idx+1)%100 == 0:
                print(datetime.datetime.now())
            #print("herehere")
        print("Testing begin")
        net.eval()
        test_loss, map_loss, count_loss = 0, 0, 0
        for batch_idx, (inputs, map_gt, count_gt) in enumerate(testloader):
            inputs = Variable(inputs.cuda())
            map_gt = Variable(map_gt.cuda())
            count_gt = Variable(count_gt.type(torch.FloatTensor).cuda())

            map_pre,bias = net(inputs)
            loss, mp_lss, cnt_lss = criterion(map_gt, count_gt, map_pre, bias)
            #loss = 10*criterion(map_pre, map_gt)
            test_loss += loss.item()
            map_loss += mp_lss.item()
            count_loss += cnt_lss.item()
        print('test loss: %.3f, map_loss: %.3f, count_loss: %.3f' %
                (test_loss/len(testloader), map_loss/len(testloader), count_loss/len(testloader)))

        test_loss /= len(testloader)
        if test_loss<best_loss:
            print("Saving..")
            if not os.path.exists('models'):
                os.mkdir('models')
            torch.save(net, os.path.join('models', "%s_%.4f.pth"%(model_name, test_loss)))
            best_loss = test_loss
        elif (epoch+1)%10==0:
            print("Saving..")
            if not os.path.exists('models'):
                os.mkdir('models')
            torch.save(net, os.path.join('models', '%s_%d.path'%(model_name, epoch+1)))


if __name__ == '__main__':
    train()
