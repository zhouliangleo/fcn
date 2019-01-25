import torch
import torchvision.transforms as transforms

from torch.autograd import Variable
from fcn_inception import FCN
from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import os
import cv2
import numpy as np
try:
    from xml.etree.cElementTree import parse
except:
    from xml.etree.ElementTree import parse
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

print('Loading model..')
net_det = RetinaNet().cuda()
net_det.load_state_dict(torch.load('../pytorch-retinanet/checkpoint/model_10_.pth')['net'])
net_det.eval()
net_det = torch.nn.DataParallel(net_det, device_ids=range(torch.cuda.device_count()))

#net_map = FCN().cuda()
#net_map = torch.load('./../dense_map/models/model_80.path')
#net_map.eval()
#net_map = torch.nn.DataParallel(net_map, device_ids=range(torch.cuda.device_count()))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

videos = ['MVI_40191','MVI_40212','MVI_40213','MVI_40981']
path = 'videos'
if not os.path.exists(path):
    os.mkdir(path)

encoder = DataEncoder()

img_path = '/home/sedlight/workspace/wei/data/DETRAC/split_img/'
ann_path = '/home/sedlight/workspace/wei/data/DETRAC/split_annotation'
w, h = 960, 560

gt = 0
pred = 0
start_time = time.time()
count = 0
for v in videos:
    print('Now is for %s'%v)
    v_path = os.path.join(img_path, v)
    imgs = os.listdir(v_path)
    imgs.sort()
    #save_path = os.path.join(path, v+'.mp4')
    #foucc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter(save_path, foucc, 30, (w,h))
    for im in imgs:
        im_path = os.path.join(v_path, im)
        an_path = os.path.join(ann_path, v, im.split('.')[0]+'.xml')
        tree = parse(an_path)
        num_gt = sum(1 for _ in tree.iterfind('object/bndbox'))
        image = Image.open(im_path).convert('RGB')
        img = image.resize((w,h))

        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x.cuda())

        #Detection
        '''
        loc_preds, cls_preds = net_det(x)
        boxes, labels = encoder.decode(loc_preds.cpu().data.squeeze(), cls_preds.cpu().data.squeeze(), (w,h))
        num_detection = len(boxes)
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box), outline='red')
        '''
        #Dense Map
        map_pre = net_map(x)
        num_denseMap = torch.sum(map_pre).cpu().item()

        #num_pre = num_detection
        num_pre = num_denseMap
        if num_pre>num_gt:
            num_pre -= num_pre-num_gt
        count += 1
        gt += num_gt
        #pred += num_detection
        pred += num_pre
        #Draw and Write
        #result = np.array(img)
        #result = result[:, :, ::-1].copy()
        #cv2.putText(result, "Ground Truth: %d"%(num_gt), (630, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        #cv2.putText(result, "Detection Based: %d"%(num_detection), (630, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        #cv2.putText(result, "Density Based: %d"%(num_denseMap), (630, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        #out.write(result)
    #out.release()
time_cost = time.time()-start_time
print('gt: %d, pred: %d, speed: %.5f' % (gt, pred, time_cost/count))
