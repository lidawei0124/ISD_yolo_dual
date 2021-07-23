from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
import model
from data import DatasetLoader
import cv2
import torchvision.models as models
import numpy as np
from scipy.optimize import leastsq
import random
import torch.nn.functional as F
import math
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error # 平方绝对误差

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def test_move(net,save_path,img_list,save_image=False,size=256):
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    points1 = np.float32([[75,55], [340,55], [33,435], [400,433]])
    points2 = np.float32([[0,0], [360,0], [0,420], [360,420]])
    M = cv2.getPerspectiveTransform(points1, points2)
    ###eval
    count=0
    distance_homo_sum=0
    distance_orb_sum=0
    distance_sift_sum=0
    pass_homo_total=0
    pass_orb_total=0
    pass_sift_total=0

    for image_path in img_list:
        count=count+1
        img1 = cv2.imread(image_path)[:,:,0]#BGR
        img1=cv2.resize(img1,(size+32,size+32))

        rho=12
        # rand_num=random.randint(0, 12)
        # #random move
        # if  rand_num<=2:#0 1 2
        #     randmovex=0
        #     randmovey=0
        # elif  rand_num<=4:#3 4 5
        #     randmovex=random.randint(-rho, rho)
        #     randmovey=0
        # elif  rand_num<=6:#6 7 8
        #     randmovex=0
        #     randmovey=random.randint(-rho, rho)
        # else:#9 10 11
        #     randmovex=random.randint(-rho, rho)
        #     randmovey=random.randint(-rho, rho)
        randmovex=random.randint(-rho, rho)
        randmovey=random.randint(-rho, rho)
        #H_groundtruth
        H_groundtruth = np.array([[1,0,randmovex],[0,1,randmovey],[0,0,1]]).astype(np.float64)
        H_inverse =np.array([[1,0,-randmovex],[0,1,-randmovey],[0,0,1]]).astype(np.float64)

        imout1=img1[16:16 + size, 16:16 + size]
        imgOut = cv2.warpPerspective(img1, H_inverse, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        imout2=imgOut[16:16 + size, 16:16 + size]

        target=np.array([randmovex,randmovey])
        im_zero=np.zeros((size,size))
        training_image = np.dstack((imout1, imout2,im_zero))
        img = training_image.transpose(2, 0, 1)#(3,300,300)
        img= torch.from_numpy(img).unsqueeze(0)
        img=img.float()
        img = img.cuda()

        #fast homography estimate network
        start_time1 = time.time()
        out = net(img)/100  #from img2 2 img1
        movexf=float(out[0][0])
        moveyf=float(out[0][1])
        movex=round(movexf)
        movey=round(moveyf)
        H_predict=np.array([[1,0,movex],[0,1,movey],[0,0,1]]).astype(np.float64)
        pass_1=time.time()-start_time1
        pass_homo_total+=pass_1

        #orb
        start_time2 = time.time()
        H_orb=img_orb(imout1,imout2)
        pass_2=time.time()-start_time2
        pass_orb_total+=pass_2

        #distance_homo
        distance_homo=mean_absolute_error(H_predict,H_groundtruth)
        distance_homo_sum=distance_homo_sum+distance_homo
        #distance_orb
        distance_orb=mean_absolute_error(H_orb,H_groundtruth)
        distance_orb_sum=distance_orb_sum+distance_orb

        print("distances(homo,orb): ",distance_homo,distance_orb)

        # save image
        if save_image:
            name = os.path.join(save_path,image_path.split("/")[-1])
            name=name[:-4]

            imgOut2_homo = cv2.warpPerspective(imout2, H_predict, (imout1.shape[1],imout1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            err_img_homo=cv2.absdiff(imout1,imgOut2_homo)

            imgOut2_orb = cv2.warpPerspective(imout2, H_orb, (imout1.shape[1],imout1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            err_img_orb=cv2.absdiff(imout1,imgOut2_orb)

            err_imgs=np.hstack([err_img_homo,err_img_orb])
            cv2.imwrite(name+"-errs.jpg",err_imgs)

    err_homo=float(distance_homo_sum)/float(count)
    err_orb=float(distance_orb_sum)/float(count)

    fps_homo=count/pass_homo_total
    fps_orb=count/pass_orb_total

    print("homo/orb(err|fps): ",err_homo,fps_homo,err_orb,fps_orb)
    return err_homo

def img_orb(img1,img2):
    # find the keypoints and descriptors with ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 20 matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)
    goodMatch = matches[:20]
    
    if len(goodMatch) > 4:
        ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
    else:
        H=np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(np.float64)
    return H


def img_anto_4temp(img1,img2):
    #1
    img2_patch1=img2[16:16 + 97, 16:16 + 97]#64,64
    img1_1=img1[0:0 + 127, 0:0 + 127]
    res = cv2.matchTemplate(img1_1,img2_patch1,cv2.TM_CCOEFF)
    #寻找最值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    loc_x1=max_loc[0]-16
    loc_y1=max_loc[1]-16
    c10=np.array([[64],[64],[1]])
    c1=np.array([[64+loc_x1],[64+loc_y1],[1]])
    #2
    img2_patch1=img2[16+127:16+97+127, 16:16 + 97]#191,64
    img1_1=img1[0+ 127:0 + 127+ 127, 0:0 + 127]
    res = cv2.matchTemplate(img1_1,img2_patch1,cv2.TM_CCOEFF)
    #寻找最值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    loc_x1=max_loc[0]-16
    loc_y1=max_loc[1]-16
    c20=np.array([[191],[64],[1]])
    c2=np.array([[191+loc_x1],[64+loc_y1],[1]])
    #3
    img2_patch1=img2[16:16 + 97, 16+127:16+97+127]#64,191
    img1_1=img1[0:0 + 127, 0+ 127:0 + 127+ 127]
    res = cv2.matchTemplate(img1_1,img2_patch1,cv2.TM_CCOEFF)
    #寻找最值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    loc_x1=max_loc[0]-16
    loc_y1=max_loc[1]-16
    c30=np.array([[64],[191],[1]])
    c3=np.array([[64+loc_x1],[191+loc_y1],[1]])
    #4
    img2_patch1=img2[16+127:16+97+127, 16+127:16+97+127]#191,191
    img1_1=img1[0+ 127:0 + 127+ 127, 0+ 127:0 + 127+ 127]
    res = cv2.matchTemplate(img1_1,img2_patch1,cv2.TM_CCOEFF)
    #寻找最值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    loc_x1=max_loc[0]-16
    loc_y1=max_loc[1]-16
    c40=np.array([[191],[191],[1]])
    c4=np.array([[191+loc_x1],[191+loc_y1],[1]])

    top_left_point = (c10[0], c10[1])
    bottom_left_point = (c30[0], c30[1])
    bottom_right_point = (c40[0], c40[1])
    top_right_point = (c20[0], c20[1])
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

    top_left_point1 = (c1[0], c1[1])
    bottom_left_point1 = (c3[0], c3[1])
    bottom_right_point1 = (c4[0], c4[1])
    top_right_point1 = (c2[0], c2[1])
    perturbed_four_points = [top_left_point1, bottom_left_point1, bottom_right_point1, top_right_point1]
    #compute H
    H11 = cv2.getPerspectiveTransform(np.float32(perturbed_four_points),np.float32(four_points))
    return H11
    
def test_move_real(net,save_path,img_list,save_image=False,size=256):
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    ###eval
    count=0
    err_sum_homo=0
    err_sum_orb=0
    pass_homo_total=0
    pass_orb_total=0

    for index in range(len(img_list)-1):
        image_path=img_list[index+1]
        count=count+1
        img1=cv2.imread(img_list[index+1])[:,:,0]
        img2=cv2.imread(img_list[index])[:,:,0]
        imout1=img1
        imout2=img2
        im_zero=np.zeros((size,size))
        training_image = np.dstack((imout1, imout2,im_zero))
        img = training_image.transpose(2, 0, 1)#(3,300,300)
        img= torch.from_numpy(img).unsqueeze(0)
        img=img.float()
        img = img.cuda()

        #fast homography estimate network
        start_time1 = time.time()
        out = net(img)/100  #from img2 2 img1
        movexf=float(out[0][0])
        moveyf=float(out[0][1])
        movex=round(movexf)
        movey=round(moveyf)
        H_predict=np.array([[1,0,movex],[0,1,movey],[0,0,1]]).astype(np.float64)
        pass_1=time.time()-start_time1
        pass_homo_total+=pass_1

        #orb
        start_time2 = time.time()
        H_orb=img_orb(imout1,imout2)
        pass_2=time.time()-start_time2
        pass_orb_total+=pass_2

        # save image
        if save_image:
            name = os.path.join(save_path,image_path.split("/")[-1])
            name=name[:-4]
            img_255=np.ones(np.shape(imout2), dtype=np.uint8)

            imgOut2_homo = cv2.warpPerspective(imout2, H_predict, (imout1.shape[1],imout1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            err_img_mask_homo = cv2.warpPerspective(img_255, H_predict, (imout1.shape[1],imout1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            err_img_homo=cv2.absdiff(imout1,imgOut2_homo)
            err_img_homo_masked=cv2.multiply(err_img_homo, err_img_mask_homo)

            err_homo=cv2.mean(err_img_homo_masked)[0]
            err_sum_homo=err_sum_homo+err_homo

            imgOut2_orb = cv2.warpPerspective(imout2, H_orb, (imout1.shape[1],imout1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            err_img_mask_orb = cv2.warpPerspective(img_255, H_orb, (imout1.shape[1],imout1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            err_img_orb=cv2.absdiff(imout1,imgOut2_orb)
            
            err_img_orb_masked=cv2.multiply(err_img_orb, err_img_mask_orb)
        

            err_orb = cv2.mean(err_img_orb_masked)[0]
            err_sum_orb=err_sum_orb+err_orb

            err_imgs=np.hstack([err_img_homo,err_img_orb])
            cv2.imwrite(name+'-'+str('%.2f' % err_homo)+'-'+str('%.2f' % err_orb)+"-errs.jpg",err_imgs)
            print("count: meanerr(homo,orb): ",count,err_homo,err_orb)

    err_homo=float(err_sum_homo)/float(count)
    err_orb=float(err_sum_orb)/float(count)

    fps_homo=count/pass_homo_total
    fps_orb=count/pass_orb_total

    print("homo/orb(err|fps)/count: ",err_homo,fps_homo,err_orb,fps_orb,count)

    return err_sum_homo,err_sum_orb,pass_homo_total,pass_orb_total,count
    #val1: 2.0822934038367795 163.61741538265935 2.359610062740149 159.21227889957785 2997
    #val2  1.9685345900686164 163.09732706650237 2.1505055076197572 162.49150259643665 760
    #val3  2.7514298790916403 161.28821837080227 3.359028515331248 129.35479029948146 748
    #val4  2.414439153189611 158.50728132014552 2.922781163995916 138.4472437489935 396
    #val5  4.169537672814229 165.41507834042875 4.679749763227805 114.57498201735315 497
if __name__ == '__main__':
    trained_model='fast_homography_acc0.07fps168.pth'
    net=model.TwowayResNet(model.ResBlock, num_classes=2)
    net = load_model(net, trained_model, False)
    print('Finished loading model!')
    cudnn.benchmark = True
    net = net.cuda()
    torch.set_grad_enabled(False)
    net.eval()
    save_path='test'
    save_image=True
    cpu=False

    real_test=True
    if real_test:
        sets=[
            'isddataset/images/val1/data5_*.bmp',
            'isddataset/images/val2/data13_*.bmp',
            'isddataset/images/val3/data15_*.bmp',
            'isddataset/images/val4/data8_*.bmp',
            'isddataset/images/val5/data18_*.bmp',
            'isddataset/images/val6/data4_*.bmp',
            'isddataset/images/val7/data9_*.bmp'
        ]
        err_homo_total=0
        time_homo_total=0
        err_orb_total=0
        time_orb_total=0
        count_total=0
        results=[]
        for set in sets:
            folder=set.replace(set.split('/')[-1]+'/*.bmp','')
            img_path = glob(folder)
            img_list=[]
            for index in range(len(img_path)-1):
                img_path1=set.replace('*',str(index+1))
                img_list.append(img_path1)
            err_sum_homo,err_sum_orb,pass_homo_total,pass_orb_total,count=test_move_real(net,save_path,img_list,save_image,size=256)
            results.append([err_sum_homo,err_sum_orb,pass_homo_total,pass_orb_total,count])
            err_homo_total+=err_sum_homo
            err_orb_total+=err_sum_orb
            time_homo_total+=pass_homo_total
            time_orb_total+=pass_orb_total
            count_total+=count

        err_homo_mean=float(err_homo_total)/float(count_total)
        err_orb_mean=float(err_orb_total)/float(count_total)

        fps_homo=count_total/time_homo_total
        fps_orb=count_total/time_orb_total

        print("homo/orb(err|fps)/count: ",err_homo_mean,fps_homo,err_orb_mean,fps_orb,count_total)
        print(results)
        #2.3674570776919364 164.40402665769133 3.0642970514990786 152.83344481363585 6190
        #2.367 6.08ms 3.064 6.54ms

#  2.0274593766170868 175.15851949713576 2.2591708526857834 175.16303848386357 6190
# [[5672.215270996094, 6131.026794433594, 17.19175124168396, 16.352716207504272, 2997], 1.893,2.046
#  [1367.4633331298828, 1443.9142608642578, 4.2554943561553955, 4.1596386432647705, 760],1.799,1.900
#   [1752.7598724365234, 1995.7484741210938, 4.184548377990723, 4.940928936004639, 748],2.343,2.668
#    [842.3330383300781, 924.5189666748047, 2.276203155517578, 2.412552833557129, 396],2.127,2.335
#     [1518.9967651367188, 1681.6861877441406, 2.9421844482421875, 3.7865424156188965, 497],3.056,3.384
#      [564.7161865234375, 925.6526031494141, 2.276801586151123, 1.4089202880859375, 396], 1.426,2.338
# [831.4890747070312, 881.7202911376953, 2.2124340534210205, 2.2772061824798584, 396]]2.100,2.227
# 2.106,2.414
    else:
        #test_compare
        test_floder='/opt/data/public02/retail/PublicDatasets/MSCOCO/val2017/'
        img_list=[]
        for path,dir_list,file_list in os.walk(test_floder):
            img_list += [os.path.join(path, i) for i in file_list if ".jpg" in i]
            img_list += [os.path.join(path, i) for i in file_list if ".bmp" in i]
            img_list += [os.path.join(path, i) for i in file_list if ".png" in i]
        random.shuffle(img_list)
        img_list_100=img_list[0:1000]
        err=test_move(net,save_path,img_list,save_image,size=256)
        print(err)