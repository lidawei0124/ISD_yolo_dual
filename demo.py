from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from models.img_registration_model import TwowayResNet,ResBlock
import cv2
import torchvision.models as models
import numpy as np
from scipy.optimize import leastsq
import random
import torch.nn.functional as F
import math
from glob import glob
import matplotlib.pyplot as plt
import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
from utils.plots import colors, plot_one_box

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def draw_anno(img,txt_file,imgsize=256):
    with open(txt_file, 'r') as txt_file:
        lines=txt_file.readlines()
        for line in lines:
            nums=line.split()
            x_center = float(nums[1]) * imgsize
            y_center = float(nums[2]) * imgsize
            width = float(nums[3]) *  imgsize
            height = float(nums[4]) *  imgsize
            xmin=max(int(x_center-width/2),0)
            ymin=max(int(y_center-height/2),0)
            xmax=min(int(x_center+width/2),imgsize)
            ymax=min(int(y_center+height/2),imgsize)
            if xmax==xmin or ymax==ymin:
                d=1
            else:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    save_img=True
    save_path='test'
    save_image=True
    folder='img_registration/isddataset/images/val6'
    folderPath=folder+'/*.bmp'#./test_images/*.jpg
    img_path = glob(folderPath)
    floder_name=img_path[0].split('/')[-1]
    floder_name=floder_name.split('_')[0]
    img_list=[]
    for index in range(len(img_path)):
        img_path1=folder+'/'+floder_name+'_{}.bmp'.format(index+1)
        img_list.append(img_path1)
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    #load TwowayResNet
    trained_model='img_registration/fast_homography_acc0.07fps168.pth'
    net1=TwowayResNet(ResBlock, num_classes=2)
    net1 = load_model(net1, trained_model, False)
    print('Finished loading model!')
    cudnn.benchmark = True
    torch.set_grad_enabled(False)
    net1 = net1.cuda()
    net1.eval()

    #load yolov5_twoway
    weights='save/isd_yolo_dual/isd_yolo_dual.pt'
    imgsz=256
    conf_thres=0.3
    iou_thres=0.6
    half_precision=True
    device = torch.cuda.current_device()
    model = attempt_load(weights, map_location=lambda storage, loc: storage.cuda(device))  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    half = half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    model.eval()

    # c10=np.array([[0],[0],[1]])
    # c20=np.array([[255],[0],[1]])
    # c30=np.array([[0],[255],[1]])
    # c40=np.array([[255],[255],[1]])
    
    # top_left_point = (c10[0], c10[1])
    # bottom_left_point = (c30[0], c30[1])
    # bottom_right_point = (c40[0], c40[1])
    # top_right_point = (c20[0], c20[1])
    # four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

    distancesum=0
    count=0
    eval_resut={}
    pass_1_total=0
    pass_2_total=0
    pass_3_total=0
    pass_4_total=0
    t0=0
    t1=0
    time1 = time.time()
    for index in range(len(img_list)-1):
        count=count+1
        # if count==100:
        #     break
        img1=cv2.imread(img_list[index+1])[:,:,0]
        img2=cv2.imread(img_list[index])[:,:,0]
        img1path=img_list[index+1]

        im_zero=np.zeros((256,256))

        #resnet-18move
        training_image = np.dstack((img1, img2, im_zero))
        img = training_image.transpose(2, 0, 1)#(3,300,300)
        # tic = time.time()
        img= torch.from_numpy(img).unsqueeze(0)
        img=img.float()
        img = img.to(device)
        start_time1 = time.time()
        out = net1(img)/100  #from img2 2 img1
        pass_1=time.time()-start_time1
        pass_1_total+=pass_1

        # out=torch.tensor.detach().numpy(out.cpu())
        movexf=float(out[0][0])
        moveyf=float(out[0][1])
        movex=round(movexf)
        movey=round(moveyf)
        
        # print("target,out,distance",movex,movey)
        start_time2 = time.time()
        test_four_points=[]
        # for point in four_points:
        #     test_four_points.append((point[0] + movexf, point[1] + moveyf))
        H_test = np.array([[1,0,movex],[0,1,movey],[0,0,1]]).astype(np.float64)
        # H_test = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(test_four_points))
        img2_out = cv2.warpPerspective(img2, H_test, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        err_img=cv2.absdiff(img1,img2_out)
        pass_2=time.time()-start_time2
        pass_2_total+=pass_2

        #yolov5_twoway
        start_time3 = time.time()
        zero_img=np.zeros((256,256))
        zero_img=zero_img.astype(np.uint8)
        img_yolo_origin = np.dstack((img1, zero_img, err_img))#BGR
        # img_yolo_origin = cv2.imread(img_list[index+1])

        img_yolo = img_yolo_origin[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img_yolo = np.ascontiguousarray(img_yolo)
        img_yolo= torch.from_numpy(img_yolo).unsqueeze(0)
    
        img_yolo = img_yolo.to(device, non_blocking=True)
        img_yolo = img_yolo.half() # if half else img_yolo.float()  # uint8 to fp16/32
        img_yolo /= 255.0  # 0 - 255 to 0.0 - 1.0

        nb, _, height, width = img_yolo.shape  # batch size, channels, height, width
        
        with torch.no_grad():
            # Run model
            out= model(img_yolo)[0]  # inference and training outputs
            # Run NMS
            out  = non_max_suppression(out, conf_thres,iou_thres)
        pass_3=time.time()-start_time3
        pass_3_total+=pass_3
        
        for i, det in enumerate(out):  # detections per image
            # im0=img1.copy()
            im0=img_yolo_origin.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_yolo.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if save_img:  # Add bbox to image
                        c = int(conf)  # integer class
                        label = 'uav:'+'%.2f' % conf
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=1)
        #draw anno
        anno_txt=img1path.replace('images','labels')
        anno_txt=anno_txt.replace('.bmp','.txt')
        # draw_anno(im0,anno_txt,imgsize=256)
        img_name=img_list[index+1].split('/')[-1]
        save_file=save_path+'/'+img_name.replace('.bmp','out.bmp')
        cv2.imwrite(save_file, im0)
        # save_file_err=save_path+'/'+img_name.replace('.bmp','err.bmp')
        # cv2.imwrite(save_file_err, err_img)
        # save_file_tem=save_path+'/'+img_name.replace('.bmp','tem.bmp')
        # cv2.imwrite(save_file_tem, img2_out)
        # print(out)
    count=len(img_list)
    cost=time.time()-time1
    # fps=count/cost
    delay1=pass_1_total/count
    delay2=pass_2_total/count
    delay3=pass_3_total/count
    fps=1/(delay1+delay2+delay3)
    print(delay1,delay2,delay3,fps)
    #0.0064287687351326185 0.001211136280893085 0.010254282272889284 55.88406915912161