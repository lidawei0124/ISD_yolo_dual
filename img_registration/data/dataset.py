import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random


class DatasetLoader(data.Dataset):
    def __init__(self,  preproc=None, train=True,size=256):
        self.preproc = preproc
        if  train is True:
            txt_path = "/opt/data/public02/retail/PublicDatasets/MSCOCO/train2017/"
        else:
            txt_path = "/opt/data/public02/retail/PublicDatasets/MSCOCO/val2017/"
        self.imgs_path= []
        for path,dir_list,file_list in os.walk(txt_path):
            self.imgs_path += [os.path.join(path, i) for i in file_list if ".jpg" in i]
            self.imgs_path += [os.path.join(path, i) for i in file_list if ".bmp" in i]
            self.imgs_path += [os.path.join(path, i) for i in file_list if ".png" in i]
        self.size=size
        c10=np.array([[0],[0],[1]])
        c20=np.array([[self.size+31],[0],[1]])
        c30=np.array([[0],[self.size+31],[1]])
        c40=np.array([[self.size+31],[self.size+31],[1]])
        
        top_left_point = (c10[0], c10[1])
        bottom_left_point = (c30[0], c30[1])
        bottom_right_point = (c40[0], c40[1])
        top_right_point = (c20[0], c20[1])
        self.four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
                    
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img1 = cv2.imread(self.imgs_path[index])[:,:,0]#BGR
        img1=cv2.resize(img1,(self.size+32,self.size+32))
        rho=12
        rand_num=random.randint(0, 12)
        perturbed_four_points = []
        if  rand_num<=2:#0 1 2
            randmovex=0
            randmovey=0
        elif  rand_num<=4:#3 4 5
            randmovex=random.randint(-rho, rho)
            randmovey=0
        elif  rand_num<=6:#6 7 8
            randmovex=0
            randmovey=random.randint(-rho, rho)
        else:#9 10 11
            randmovex=random.randint(-rho, rho)
            randmovey=random.randint(-rho, rho)
        # if  rand_num<=2:#0 1 2
        #     randmovex=0
        #     randmovey=0
        #     target=[1,0,0,0]
        # elif  rand_num<=4:#3 4 5
        #     randmovex=10
        #     randmovey=0
        #     target=[0,1,0,0]
        # elif  rand_num<=6:#6 7 8
        #     randmovex=0
        #     randmovey=10
        #     target=[0,0,1,0]
        # else:#9 10 11
        #     randmovex=10
        #     randmovey=10
        #     target=[0,0,0,1]
        for point in self.four_points:
            perturbed_four_points.append((point[0] + randmovex, point[1] + randmovey))
            # perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))
        #compute H
        H11 = cv2.getPerspectiveTransform(np.float32(self.four_points), np.float32(perturbed_four_points))
        H_inverse = np.linalg.inv(H11)
        imout1=img1[16:16 + self.size, 16:16 + self.size]
        imgOut = cv2.warpPerspective(img1, H_inverse, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        imout2=imgOut[16:16 + self.size, 16:16 + self.size]
        #debug
        # imgOut2_reverse = cv2.warpPerspective(imout2, H11, (224,224),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        # err_img=cv2.absdiff(imout1,imgOut2_reverse)
        # cv2.imwrite("imout1.jpg",imout1)
        # cv2.imwrite("imout2.jpg",imout2)
        # cv2.imwrite("error.jpg",err_img)
        #
        target=np.array([randmovex,randmovey])
        im_zero=np.zeros((self.size,self.size))
        training_image = np.dstack((imout1, imout2,im_zero))
        img = training_image.transpose(2, 0, 1)#(3,300,300)
        # target=np.array(target)
        return torch.from_numpy(img), torch.from_numpy(target)

def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        if torch.is_tensor(sample[0]):
            imgs.append(sample[0])
            targets.append(sample[1])
    return (torch.stack(imgs,0), torch.stack(targets,0))
