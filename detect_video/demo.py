import torch
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.autograd import Variable
from fcn_inception_smallMap import FCN
from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import os
import cv2
import time
import math
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="1"

print('Loading model..')
net_det = RetinaNet().cuda()
net_det.load_state_dict(torch.load('../pytorch-retinanet/checkpoint/model_10_.pth')['net'])
net_det.eval()
net_det = torch.nn.DataParallel(net_det, device_ids=range(torch.cuda.device_count()))

net_map = FCN().cuda()
net_map.load_state_dict(torch.load('./../dense_finetune/models/inception_smallMap_3_0.8880.pth'))
net_map.eval()
net_map = torch.nn.DataParallel(net_map, device_ids=range(torch.cuda.device_count()))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

#videos = ['MVI_40191','MVI_40212','MVI_40213','MVI_40981']
#path = 'videos_smallMap'
#if not os.path.exists(path):
#    os.mkdir(path)

encoder = DataEncoder()

path='/home/sedlight/workspace/wei/data/track1_videos/det/'
img_path = '/home/sedlight/workspace/wei/data/track1_videos/img/'
videos=os.listdir(img_path)
finished=os.listdir('/home/sedlight/workspace/wei/data/track1_videos/det')
#ann_path = '/home/sedlight/workspace/wei/data/DETRAC/split_annotation'
w, h = 960, 560
videos=list(set(videos)-set(finished))
#gt = 0
#pred_dec = 0
#pred_map = 0
#pred_map_r = 0
#start_time = time.time()
#count = 0
for v in videos:
    print('Now is for %s'%v)
    v_path = os.path.join(img_path, v)
    imgs = os.listdir(v_path)
    imgs.sort()
    save_path = os.path.join(path, v+'.mp4')
    foucc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(save_path, foucc, 30, (w,h))
    for im in imgs:
        im_path = os.path.join(v_path, im)
        #an_path = os.path.join(ann_path, v, im.split('.')[0]+'.xml')
        #tree = parse(an_path)
        #num_gt = sum(1 for _ in tree.iterfind('object/bndbox'))
        image = Image.open(im_path).convert('RGB')
        img = image.resize((w,h))

        #if num_gt>23:
        #    img.save('img.jpg')
        #else:
        #    continue
        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x.cuda())

        #Detection
        loc_preds, cls_preds = net_det(x)
        try:
            boxes, labels = encoder.decode(loc_preds.cpu().data.squeeze(), cls_preds.cpu().data.squeeze(), (w,h))
        except:
            print(loc_preds.cpu().data.squeeze().shape)
            break
        num_detection = len(boxes)
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box), outline='red')

        #Dense Map
        map_pre, bias = net_map(x)
        map_numpy = map_pre.cpu().detach().numpy()
        #print(map_numpy.shape)
        #np.save('map.npy', map_numpy)
        #map_pil = transforms.ToPILImage()(map_pre)
        #map_pil.save("map.jpg")
        #print('quit')
        #quit()
        num_denseMap = torch.sum(map_pre).cpu().item()
        num_denseMap = int(math.ceil(num_denseMap + torch.sum(bias).cpu().item()))
        #num_map_round = int(round(num_denseMap+torch.sum(bias).cpu().item()))
        # print(math.ceil(num_denseMap), num_gt, num_detection)
        # quit()
        #num_pre = num_detection
        #if num_pre>num_gt:
        #    num_pre -= num_pre-num_gt
        #count += 1
        #gt += num_gt
        #pred_dec += num_pre


        # num_pre_map = num_denseMap
        # if num_pre_map>num_gt:
        #    num_pre_map -= num_pre_map-num_gt
        #    num_pre_map = num_gt
        #    num_denseMap = num_gt
        #pred_map += num_pre_map

        #num_map_r = num_map_round
        #if num_map_r>num_gt:
            # num_map_r -= num_map_r-num_gt

        #pred_map_r += num_map_r
        #num_map_round = pred_
        #Draw and Write
        result = np.array(img)
        result = result[:, :, ::-1].copy()
        #cv2.putText(result, "Ground Truth: %d"%(num_gt), (630, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        cv2.putText(result, "Detection Based: %d"%(num_detection), (630, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        cv2.putText(result, "Density Based: %d"%(num_denseMap), (630, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        out.write(result)
    out.release()
#print('gt: %d, pred_detection: %d, pred_denseMap: %d, pred_Map_r: %d'%(gt, pred_dec, pred_map, pred_map_r))
