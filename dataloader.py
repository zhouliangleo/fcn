#!/usr/bin/python
#coding=utf8
"""
# Author: wei
# Created Time : 2018-09-26 09:04:40

# File Name: dataloader.py
# Description:

"""
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from transform import resize, random_flip, random_crop, center_crop
from PIL import Image
#import numpy as np

class Datamaker(data.Dataset):
    def __init__(self, transform=transforms.ToTensor(), train='True', size=(960,560)):
        self.im_pth = "/home/leo/data/DETRAC/split_img"
        self.transform = transform
        self.size = size
        self.train = train
        if train:
            file_name = "/home/leo/fcn/data/detrac_train.txt"
        else:
            file_name = "/home/leo/fcn/data/detrac_val.txt"
        self.boxes = []
        self.fnames = []
        with open(file_name) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited)-1)//5
            box = []
            for i in range(num_boxes):
                xmin = splited[1+i*5]
                ymin = splited[2+i*5]
                xmax = splited[3+i*5]
                ymax = splited[4+i*5]
                box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            self.boxes.append(torch.Tensor(box))

    def  __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        fname = os.path.join(self.im_pth, self.fnames[index])
        img = Image.open(fname)
        if img.mode!='RGB':
            img = img.convert('RGB')
        boxes = self.boxes[index].clone()
        size = self.size
        #print(img.size)
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, size)
        else:
            img, boxes = center_crop(img, boxes, size)
            img, boxes = resize(img, boxes, size)
        if self.transform is not None:
            img = self.transform(img)

        dense_map = torch.zeros([1, img.size()[1], img.size()[2]], dtype=torch.float32)
        #print(dense_map.size())
        box_num = 0
        for box in boxes:
            area = (box[2]-box[0])*(box[3]-box[1])
            #print(box[0], box[1], box[2], box[3], area)
            if area<100.:
                continue
            box_num += 1
            try:
                dense_map[:, box[1].type(torch.int32):box[3].type(torch.int32), box[0].type(torch.int32):box[2].type(torch.int32)] += 1/area
            except:
                print(fname, dense_map.size())
                print(box[1].type(torch.int32), box[3].type(torch.int32), box[0].type(torch.int32), box[2].type(torch.int32), area)
        return img, dense_map, box_num

if __name__ == '__main__':
    a = Datamaker()
    from torch.utils.data import DataLoader
    loader = DataLoader(a, batch_size=16, shuffle=True, num_workers=16)
    for idx, (inputs, map_c, count_c) in enumerate(loader):
        pass
    im, dense_map, num = a.__getitem__(1)
    #im_tensor = transforms.ToTensor()(im)
    im = transforms.ToPILImage()(im)
    im.save('tmp.jpg')
    print(im.size, dense_map.size())
    dense_map = dense_map / torch.max(dense_map)
    dense_tensor = torch.stack([dense_map, dense_map, dense_map], dim=0)
    dense_img = transforms.ToPILImage()(dense_tensor)
    print(dense_img.size, dense_tensor.size())
    dense_img.save("map.jpg")
