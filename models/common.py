# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
import cv2

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print(c1 * 4, c2, k, s, p, g, act)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class AutoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(AutoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

class SELayer(nn.Module):
    def __init__(self, c1, r=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1//r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1//r, c1, bias=False)
        # self.l1 = nn.Linear(3, 1, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        # self.l2 = nn.Linear(1, 3, bias=False)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.sigmoid = nn.Sigmoid()
        #self.act=SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) 
        #self.act=SiLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
        
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out   

class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 1):
        super(ResBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(c1=places*self.expansion,c2=places*self.expansion,)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    

class Twoway_fcos_conv_c3(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 1):
        super(Twoway_fcos_conv_c3,self).__init__()
        self.focus1=nn.Sequential(
            Focus(1,32,k=3),
            # Conv(16,32,k=3,s=2),
        )
        self.focus2=nn.Sequential(
            Focus(1,32,k=3),
            # Conv(16,32,k=3,s=2),
        )
        self.focus3=nn.Sequential(
            Focus(2,32,k=3),
            # Conv(16,32,k=3,s=2),
        )
        self.conv1=nn.Sequential(
            # Focus(1,32,k=3),
            Conv(96,32,k=1,s=1),
        )


    def forward(self, x):
        input1=x[:,0,:,:].unsqueeze(1)
        input2=x[:,1,:,:].unsqueeze(1)
        input3=x[:,0:2,:,:]
        output1 = self.focus1(input1)#16
        output2 = self.focus2(input2)#16
        output3 = self.focus3(input3)#32
        out=torch.cat((output1,output2,output3),dim=1)#64
        #[1, 64, 16, 16]
        # out=torch.cat((output1,output2),dim=1)
        out=self.conv1(out)#32
        #32
        return out

class Focus_twoway_old(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus_twoway_old, self).__init__()
        #c1=3 c2=32 k=3
        # 12 32 3 1 None 1 True
        self.conv = Conv(c1 * 4, c2//2, k, s, p, g, act)

        self.conv1 = Conv(1 * 4, c2//2, k, s, p, g, act)
        self.conv2 = Conv(1 * 4, c2//2, k, s, p, g, act)
        # self.conv3 = Conv(3 * 4, c2, k, s, p, g, act)
        
        self.conv4 = Conv(2*c2//2, c2//2, 1, 1)
        self.conv5 = Conv(2*c2//2, c2//2, 1, 1)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x1=x[:,0,:,:].unsqueeze(1)
        # x1=x[:,2,:,:].unsqueeze(1)
        x2=x[:,1,:,:].unsqueeze(1)
        x3=x
        
        x_cat1=torch.cat([x1[..., ::2, ::2], x1[..., 1::2, ::2], x1[..., ::2, 1::2], x1[..., 1::2, 1::2]], 1)#4
        x_cat2=torch.cat([x2[..., ::2, ::2], x2[..., 1::2, ::2], x2[..., ::2, 1::2], x2[..., 1::2, 1::2]], 1)#4
        x_out1=self.conv1(x_cat1)#32
        x_out2=self.conv2(x_cat2)#32
        out12=torch.cat((x_out1,x_out2),dim=1)#64
        out12=self.conv4(out12)#32

        x_cat3=torch.cat([x3[..., ::2, ::2], x3[..., 1::2, ::2], x3[..., ::2, 1::2], x3[..., 1::2, 1::2]], 1)#8
        out3=self.conv(x_cat3)#32
        out=torch.cat((out12,out3),dim=1)#64
        # out=self.conv5(out)#32

        return out

        # x_origin=x
        # x_cat4=torch.cat([x_origin[..., ::2, ::2], x_origin[..., 1::2, ::2], x_origin[..., ::2, 1::2], x_origin[..., 1::2, 1::2]], 1)#8
        # return self.conv(x_cat4)

class Focus_twoway(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus_twoway, self).__init__()
        #c1=3 c2=32 k=3
        kk=(3,5,7)
        print(c1,c2,k, s, p, g, act)
        # self.conv8_32 = Conv(2 * 4, 32, k, s, p, g, act)
        self.conv4_32_1 = Conv(1 * 4, 32, 1, 1, p, g, act)
        self.conv4_16_2 = Conv(1 * 4, 16, 1, 1, p, g, act)
        # self.conv3 = Conv(3 * 4, c2, k, s, p, g, act)
        # self.conv64_32_1 = Conv(64, 32, 1, 1)
        self.conv48_32_2 = Conv(64, 32, 1, 1)
        # self.conv64_64_1 = Conv(64, 64, 1, 1)

        # self.conv64_128_2 = Conv(64, 128, 3, 2)
        # self.maxpool_3=nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2)
        self.maxpool_5=nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.avgpool_5=nn.AvgPool2d(kernel_size=5, stride=1, padding=5 // 2)
        # self.maxpool_7=nn.MaxPool2d(kernel_size=7, stride=1, padding=7 // 2)

        # self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        # self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        # self.act = nn.LeakyReLU(0.1, inplace=True)
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        # self.conv32_32_1 = Conv(32, 32, 1, 1)
        self.conv48_32_1 = Conv(48, 32, 1, 1)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)

        x1=x[:,2,:,:].unsqueeze(1)#img_origin
        x2=x[:,0,:,:].unsqueeze(1)#img_err
        # x3=torch.cat((x1,x2),dim=1)#64

        x_cat1=torch.cat([x1[..., ::2, ::2], x1[..., 1::2, ::2], x1[..., ::2, 1::2], x1[..., 1::2, 1::2]], 1)#4
        x_cat2=torch.cat([x2[..., ::2, ::2], x2[..., 1::2, ::2], x2[..., ::2, 1::2], x2[..., 1::2, 1::2]], 1)#4
        # x_cat3=torch.cat([x3[..., ::2, ::2], x3[..., 1::2, ::2], x3[..., ::2, 1::2], x3[..., 1::2, 1::2]], 1)#8

        x_out1=self.conv4_32_1(x_cat1)#32
        x_out2=self.conv4_16_2(x_cat2)#16
        # x_out3=self.conv8_32(x_cat3)#32
        
        x_out2_avg5=self.avgpool_5(x_out2)
        x_out2_max5=self.maxpool_5(x_out2)
        # x_out2_max7=self.maxpool_7(x_out2)
        # xout21=self.maxpool_list(x_out2)
        x_out22=torch.cat((x_out2,x_out2_avg5,x_out2_max5),dim=1)#48
        x_out22=self.conv48_32_1(x_out22)#32

        # print(x_out22.size())
        # # print(xout21.size())
        # xout222=torch.cat((x_out22,x_out2),dim=1)#64
        # xout222=self.conv32_32_2(xout222)#32

        # xout2222=self.cv2(torch.cat([x_out2] + [m(x_out2) for m in self.maxpool_list], 1))#32

        out=torch.cat((x_out1,x_out22),dim=1)#64

        # out=self.conv64_128_2(out)


        # out12=torch.cat((x_out1,x_out2),dim=1)#64
        # # print(out12.size())
        # out12=self.conv64_32_1(out12)#64
        # # print(out12.size())

        # out=torch.cat((out12,x_out3),dim=1)#64
        # out=self.conv64_64_1(out)#64

        return out

class Focus_2in(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus_2in, self).__init__()
        # print(c1 * 4, c2, k, s, p, g, act)
        c1=2
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x1=x[:,2,:,:].unsqueeze(1)#img_origin
        x2=x[:,0,:,:].unsqueeze(1)#img_err
        x3=torch.cat((x1,x2),dim=1)#64
        # print(x3.size())
        return self.conv(torch.cat([x3[..., ::2, ::2], x3[..., 1::2, ::2], x3[..., ::2, 1::2], x3[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

class Focus_twoway_V2(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus_twoway_V2, self).__init__()
        #c1=3 c2=32 k=3
        kk=(3,5,7)
        print(c1,c2,k, s, p, g, act)
        # self.conv8_32 = Conv(2 * 4, 32, k, s, p, g, act)
        self.conv4_32_1 = Conv(1 * 4, 32, 1, 1, p, g, act)
        self.conv4_16_2 = Conv(1 * 4, 16, 1, 1, p, g, act)
        # self.conv3 = Conv(3 * 4, c2, k, s, p, g, act)
        # self.conv64_32_1 = Conv(64, 32, 1, 1)
        self.conv48_32_2 = Conv(64, 32, 1, 1)
        # self.conv64_64_1 = Conv(64, 64, 1, 1)

        # self.conv64_128_2 = Conv(64, 128, 3, 2)
        # self.maxpool_3=nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2)
        self.maxpool_3=nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2)
        self.avgpool_3=nn.AvgPool2d(kernel_size=3, stride=1, padding=3 // 2)
        self.maxpool_5=nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.avgpool_5=nn.AvgPool2d(kernel_size=5, stride=1, padding=5 // 2)
        # self.maxpool_7=nn.MaxPool2d(kernel_size=7, stride=1, padding=7 // 2)

        # self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        # self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        # self.act = nn.LeakyReLU(0.1, inplace=True)
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        # self.conv32_32_1 = Conv(32, 32, 1, 1)
        self.conv80_32_1 = Conv(80, 32, 1, 1)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)

        x1=x[:,2,:,:].unsqueeze(1)#img_origin
        x2=x[:,0,:,:].unsqueeze(1)#img_err
        # x3=torch.cat((x1,x2),dim=1)#64

        x_cat1=torch.cat([x1[..., ::2, ::2], x1[..., 1::2, ::2], x1[..., ::2, 1::2], x1[..., 1::2, 1::2]], 1)#4
        x_cat2=torch.cat([x2[..., ::2, ::2], x2[..., 1::2, ::2], x2[..., ::2, 1::2], x2[..., 1::2, 1::2]], 1)#4
        # x_cat3=torch.cat([x3[..., ::2, ::2], x3[..., 1::2, ::2], x3[..., ::2, 1::2], x3[..., 1::2, 1::2]], 1)#8

        x_out1=self.conv4_32_1(x_cat1)#32
        x_out2=self.conv4_16_2(x_cat2)#16
        # x_out3=self.conv8_32(x_cat3)#32
        
        x_out2_avg5=self.avgpool_3(x_out2)
        x_out2_max5=self.maxpool_3(x_out2)
        x_out2_avg7=self.avgpool_5(x_out2)
        x_out2_max7=self.maxpool_5(x_out2)
        # x_out2_max7=self.maxpool_7(x_out2)
        # xout21=self.maxpool_list(x_out2)
        x_out22=torch.cat((x_out2,x_out2_avg5,x_out2_max5,x_out2_avg7,x_out2_max7),dim=1)#80
        x_out22=self.conv80_32_1(x_out22)#32

        # print(x_out22.size())
        # # print(xout21.size())
        # xout222=torch.cat((x_out22,x_out2),dim=1)#64
        # xout222=self.conv32_32_2(xout222)#32

        # xout2222=self.cv2(torch.cat([x_out2] + [m(x_out2) for m in self.maxpool_list], 1))#32

        out=torch.cat((x_out1,x_out22),dim=1)#64

        # out=self.conv64_128_2(out)


        # out12=torch.cat((x_out1,x_out2),dim=1)#64
        # # print(out12.size())
        # out12=self.conv64_32_1(out12)#64
        # # print(out12.size())

        # out=torch.cat((out12,x_out3),dim=1)#64
        # out=self.conv64_64_1(out)#64

        return out