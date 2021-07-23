from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import DatasetLoader, detection_collate
import time
import datetime
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.models as models
import model
import random
import numpy as np
import cv2
import shutil
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# net=model.ResNet(model.ResBlock, num_classes=2)
# net = models.resnet18(num_classes=2)
net=model.TwowayResNet(model.ResBlock, num_classes=2)
print("Printing net...")
print(net)
# print(net1)
# r_seed=random.randint(0, 1000)
r_seed=888
print("random_seed:{}".format(r_seed))
random.seed(888)
num_classes = 2
num_gpu = 2
batch_size = 256

lr_decay1=20
lr_decay2=30
lr_decay3=40
lr_decay4=50
max_epoch = 60
gpu_train = True
initial_lr = 0.2
num_workers = 8
momentum = 0.9
weight_decay = 5e-4
gamma = 0.2
save_folder = './weights/'
resume_net=None
resume_epoch=0
img_size=256

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

if resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)


def train():
    log_path='./log'
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    else:
        os.makedirs(log_path)

    writer=None
    writer = SummaryWriter(log_dir=log_path)


    test_floder='/opt/data/public02/retail/PublicDatasets/MSCOCO/val2017/'
    val_img_list=[]
    for path,dir_list,file_list in os.walk(test_floder):
        val_img_list += [os.path.join(path, i) for i in file_list if ".jpg" in i]
        val_img_list += [os.path.join(path, i) for i in file_list if ".bmp" in i]
        val_img_list += [os.path.join(path, i) for i in file_list if ".png" in i]
    
    loss_log={}
    acc_log={}
    net.train()
    acc=0
    epoch = 0 + resume_epoch
    print('Loading Dataset...')

    dataset = DatasetLoader(preproc=None,train=True,size=img_size)
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (lr_decay1* epoch_size, lr_decay2 * epoch_size, lr_decay3 * epoch_size,lr_decay4 * epoch_size)
    step_index = 0

    if resume_epoch > 0:
        start_iter = resume_epoch * epoch_size
    else:
        start_iter = 0
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 5 == 0):
                #eval and save
                print("eval-ing")
                acc=val_move(net,val_img_list,img_size,100)
                torch.save(net.state_dict(), save_folder + 'TwowayResNet'+ '_epoch_' + str(epoch) +'acc'+str(acc)+ '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.float()
        images = images.cuda()
        # targets = torch.cat(targets)
        targets=targets.float()
        targets = targets.cuda()

        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()
        #loss CSSSF
        # loss=F.binary_cross_entropy_with_logits(out,targets)
        loss = F.smooth_l1_loss(out, 100*targets, reduction='mean')
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || loss: {:.4f} || acc: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss.item(),acc, lr, batch_time, str(datetime.timedelta(seconds=eta))))
        loss_log[iteration+1]=loss.item()
        acc_log[iteration+1]=acc
        #tensorboard
        writer.add_scalar('loss/loss',loss.item(),iteration)
        writer.add_scalar('loss/acc',acc,iteration)
        writer.add_scalar('loss/lr',lr,iteration)

        y1=list(loss_log.values())
        x1=list(loss_log.keys())   
        plt.plot(x1,y1)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('loss')
        plt.savefig('./loss_log.jpg')
        plt.close()
        y2=list(acc_log.values())
        x2=list(acc_log.keys())   
        plt.plot(x2,y2)
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.title('acc')
        plt.savefig('./acc_log.jpg')
        plt.close()
                
    torch.save(net.state_dict(), save_folder + 'TwowayResNet' +'acc'+str(acc)+ '_Final.pth')

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def val_move(net,img_list,size=256,maxcount=100):

    c10=np.array([[0],[0],[1]])
    c20=np.array([[size+31],[0],[1]])
    c30=np.array([[0],[size+31],[1]])
    c40=np.array([[size+31],[size+31],[1]])
    
    top_left_point = (c10[0], c10[1])
    bottom_left_point = (c30[0], c30[1])
    bottom_right_point = (c40[0], c40[1])
    top_right_point = (c20[0], c20[1])
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

    #torch.set_grad_enabled(False)
    net.eval()
    ###eval
    distancesum=0
    count=0

    for image_path in img_list:
        count=count+1
        if count==maxcount:
            break
        img1 = cv2.imread(image_path)[:,:,0]#BGR
        img1=cv2.resize(img1,(size+32,size+32))
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

        for point in four_points:
            perturbed_four_points.append((point[0] + randmovex, point[1] + randmovey))

        H11 = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        H_inverse = np.linalg.inv(H11)
        imout1=img1[16:16 + size, 16:16 + size]
        imgOut = cv2.warpPerspective(img1, H_inverse, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        imout2=imgOut[16:16 + size, 16:16 + size]

        # target=np.array([randmovex,randmovey])
        im_zero=np.zeros((size,size))
        training_image = np.dstack((imout1, imout2,im_zero))
        img = training_image.transpose(2, 0, 1)#(3,300,300)
        img= torch.from_numpy(img).unsqueeze(0)
        # tic = time.time()
        img=img.float()
        img = img.cuda()
        # print(img.size())
        out = net(img)/100  #from img2 2 img1
        # out=torch.tensor.detach().numpy(out.cpu())
        movexf=float(out[0][0])
        moveyf=float(out[0][1])
        movex=int(out[0][0])
        movey=int(out[0][1])
        distance=math.sqrt((movexf-randmovex)*(movexf-randmovex)+(moveyf-randmovey)*(moveyf-randmovey))
        distancesum=distancesum+distance
    acc=float(distancesum)/float(count)
    net.train()
    print(acc)
    return acc

if __name__ == '__main__':
    train()
