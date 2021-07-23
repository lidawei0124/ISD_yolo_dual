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
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
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
    
def img_anto_sift(img1,img2):
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
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)
    goodMatch = matches[:20]
    
    if len(goodMatch) > 4:
        ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
    else:
        H=None
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
    
def test_move_real(net,save_path,img_list,anno_txt_folder,save_folder,save_image=True,size=256):
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    c10=np.array([[0],[0],[1]])
    c20=np.array([[size-1],[0],[1]])
    c30=np.array([[0],[size-1],[1]])
    c40=np.array([[size-1],[size-1],[1]])
    
    top_left_point = (c10[0], c10[1])
    bottom_left_point = (c30[0], c30[1])
    bottom_right_point = (c40[0], c40[1])
    top_right_point = (c20[0], c20[1])
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

    distancesum=0
    count=0
    eval_resut={}
    pass_1_total=0
    pass_2_total=0
    pass_3_total=0
    for index in range(len(img_list)-1):
        img1_path=img_list[index+1]
        img1_name=img1_path.split('/')[-1]
        err_img1_last_path=save_folder+'/'+img1_name.replace('.bmp','_err2last.bmp')
        anno_txt_name=img1_name.replace('bmp','txt')
        txt_file_path=anno_txt_folder+'/'+anno_txt_name
        img1=cv2.imread(img_list[index+1])[:,:,0]
        img2=cv2.imread(img_list[index])[:,:,0]
        start_time2 = time.time()
        # H_4temp=img_anto_4temp(img1,img2)
        pass_2=time.time()-start_time2
        pass_2_total+=pass_2

        start_time3 = time.time()
        # H_sift=img_anto_sift(img1,img2)
        pass_3=time.time()-start_time3
        pass_3_total+=pass_3
        
        
        count=count+1
        # if count==100:
        #     break
        imout1=img1
        imout2=img2
        im_zero=np.zeros((size,size))

        training_image = np.dstack((imout1, imout2,im_zero))
        img = training_image.transpose(2, 0, 1)#(3,300,300)
        # tic = time.time()
        img= torch.from_numpy(img).unsqueeze(0)
        img=img.float()
        img = img.cuda()

        start_time1 = time.time()
        out = net(img)/100  #from img2 2 img1
        pass_1=time.time()-start_time1
        pass_1_total+=pass_1
        # out=torch.tensor.detach().numpy(out.cpu())
        movexf=float(out[0][0])
        moveyf=float(out[0][1])
        movex=round(movexf)
        movey=round(moveyf)
        
        print("img_name,distance",img1_name,movex,movey)
                # save image
        if save_image:
            name = os.path.join(save_path,str(index))
            test_four_points=[]
            for point in four_points:
                test_four_points.append((point[0] + movexf, point[1] + moveyf))
            #from img2 2 img1
            H_test = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(test_four_points))
            # H_test_inverse = np.linalg.inv(H_test)
            imgOut2_out = cv2.warpPerspective(imout2, H_test, (imout1.shape[1],imout1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            # imgOut2_out_4temp = cv2.warpPerspective(imout2, H_4temp, (imout1.shape[1],imout1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            # if H_sift is not None:
            #     imgOut2_out_sift = cv2.warpPerspective(imout2, H_sift, (imout1.shape[1],imout1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            #     err_img_sift=cv2.absdiff(imout1,imgOut2_out_sift)
            # else:
            #     err_img_sift=np.ones((size,size))

            err_img=cv2.absdiff(imout1,imgOut2_out)
            
            # err_img_4temp=cv2.absdiff(imout1,imgOut2_out_4temp)
            # err_origin=cv2.absdiff(imout1,imout2)
            # cv2.imwrite(name+"-imout1.jpg",imout1)
            # cv2.imwrite(name+"-imout2.jpg",imout2)
            # cv2.imwrite(name+"-error.jpg",err_img)
            # with open(txt_file_path, 'r') as txt_file:
            #     lines=txt_file.readlines()
            #     for line in lines:
            #         nums=line.split()
            #         x_center = float(nums[1]) * size
            #         y_center = float(nums[2]) * size
            #         width = float(nums[3]) *  size
            #         height = float(nums[4]) *  size
            #         xmin=max(int(x_center-width/2),0)
            #         ymin=max(int(y_center-height/2),0)
            #         xmax=min(int(x_center+width/2),size)
            #         ymax=min(int(y_center+height/2),size)
            #         # cv2.rectangle(err_img, (xmin, ymin), (xmax, ymax), color=(255, 255, 255), thickness=1)
            #         if xmax==xmin or ymax==ymin:
            #             d=0
            #         else:
            #             d=1
            #             # cv2.rectangle(err_img, (xmin, ymin), (xmax, ymax), color=(255, 255, 255), thickness=1)
            cv2.imwrite(err_img1_last_path,err_img)
            # txt_file.close()
            # plt.subplot(231)
            # plt.title("imout1")
            # plt.imshow(imout1)
            # plt.subplot(232)
            # plt.title("err_origin")
            # plt.imshow(err_origin)
            # # plt.subplot(233)
            # # plt.title("err_img_4temp")
            # # plt.imshow(err_img_4temp)
            # plt.subplot(234)
            # plt.title("err_img")
            # plt.imshow(err_img)
            # plt.subplot(235)
            # plt.title("err_img_sift")
            # plt.imshow(err_img_sift)
            # plt.savefig(name+"-resut.jpg")
    fps_1=(len(img_list)-1)/pass_1_total
    fps_2=(len(img_list)-1)/pass_2_total
    fps_3=(len(img_list)-1)/pass_3_total
    print(fps_1,fps_2,fps_3)
    return fps_1,fps_2,fps_3

if __name__ == '__main__':
    trained_model='weights/TwowayResNet_epoch_50acc0.07fps168.pth'
    net=model.TwowayResNet(model.ResBlock, num_classes=2)
    net = load_model(net, trained_model, False)
    print('Finished loading model!')
    cudnn.benchmark = True
    net = net.cuda()
    #torch.set_grad_enabled(False)
    net.eval()
    save_path='test'
    save_image=True
    cpu=False

    folder='infrared/data4'
    err_folder=folder+'/err'
    if os.path.exists(err_folder):
        shutil.rmtree(err_folder)
        os.makedirs(err_folder)
    else:
        os.makedirs(err_folder)

    anno_txt_folder='infrared/annotation/txt/data9'
    folder_name=folder.split('/')[-1]
    folderPath=folder+'/*.bmp'#./test_images/*.jpg
    img_path = glob(folderPath)
    img_list=[]
    for index in range(len(img_path)):
        img_path1=folder+'/'+folder_name+'_{}.bmp'.format(index)
        img_list.append(img_path1)
    acc=test_move_real(net,save_path,img_list,anno_txt_folder,err_folder,save_image,256)
    print(acc)