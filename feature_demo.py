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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def draw_feature(features,name):
    feature = features[:, 0, :, :]
    print(feature.shape)

    feature = feature.view(feature.shape[1], feature.shape[2])
    print(feature.shape)
    feature = feature.data.numpy()

    #use sigmod to [0,1]
    feature = 1.0/(1+np.exp(-1*feature))

    # to [0,255]
    feature = np.round(feature*255)
    print(feature[0])

    cv2.imwrite(name+'_feature.bmp', feature)

def show_features(model,input,show_ids):
    x = input
    for index, layer in enumerate(model(model)):
        x = layer(x)
        if index in show_ids:
            draw_feature(x,str(index))

if __name__ == '__main__':
    #load yolov5_twoway
    weights='save/isd_twoway_new/exp30/weights/best.pt'
    imgsz=256
    conf_thres=0.25
    iou_thres=0.45
    half_precision=True
    device = torch.cuda.current_device()
    model = attempt_load(weights, map_location=lambda storage, loc: storage.cuda(device))  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    half = half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    model.eval()
    print(model)
    imgpath="img_registration/isddataset/images/val5/data18_499.bmp"
    errimgpath="img_registration/isddataset/errimgs/val5/data18_499_err2last.bmp"
    img1=cv2.imread(imgpath)[:,:,0]
    errimg2=cv2.imread(errimgpath)[:,:,0]
    #yolov5_twoway
    start_time3 = time.time()
    zero_img=np.zeros((256,256))
    zero_img=zero_img.astype(np.uint8)
    img_yolo_origin = np.dstack((img1, zero_img, errimg2))#BGR

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
        # show_features(model,img_yolo,[0,1,2])
        # Run NMS
        out  = non_max_suppression(out, conf_thres,iou_thres)
        print(out)