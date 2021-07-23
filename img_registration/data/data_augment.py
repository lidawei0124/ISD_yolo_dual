import cv2
import numpy as np
import random
import random
key_point_number1 = 7

def _rand_crop(image, boxes, labels, landm, img_dim,rand_ratio=0.1):
    height, width, _ = image.shape
    rand_h=int(height*rand_ratio)
    rand_w=int(width*rand_ratio)
    for i in range(boxes.shape[1]):
        if i % 2 == 0:
            boxes[0,i] = min(max(boxes[0,i], 0), width-1)
        else:
            boxes[0,i] = min(max(boxes[0,i], 0), height-1)

    for i in range(landm.shape[1]):
        if i % 2 == 0:
            landm[0,i] = min(max(landm[0,i], 0), width-1)
        else:
            landm[0,i] = min(max(landm[0,i], 0), height-1)

    boxes = boxes.astype("int")

    x1 = random.randint(max(0, boxes[0,0]-rand_w),boxes[0,0]+rand_w)
    y1 = random.randint(max(0, boxes[0,1]-rand_w),boxes[0,1]+rand_w)
    x2 = random.randint(boxes[0,2]-rand_w, min(width-1, boxes[0,2]+rand_w))
    y2 = random.randint(boxes[0,3]-rand_h, min(height-1, boxes[0,3]+rand_h))

    image = image[y1:y2, x1:x2, :]
    height, width, _ = image.shape
    landm[:, 0::2] = landm[:, 0::2] - x1
    landm[:, 1::2] = landm[:, 1::2] - y1
    return image, landm
def _crop(image, boxes, labels, landm, img_dim):
    height, width, _ = image.shape
    for i in range(boxes.shape[1]):
        if i % 2 == 0:
            boxes[0,i] = min(max(boxes[0,i], 0), width-1)
        else:
            boxes[0,i] = min(max(boxes[0,i], 0), height-1)

    for i in range(landm.shape[1]):
        if i % 2 == 0:
            landm[0,i] = min(max(landm[0,i], 0), width-1)
        else:
            landm[0,i] = min(max(landm[0,i], 0), height-1)

    boxes = boxes.astype("int")

    x1 = random.randint(max(0, boxes[0,0]-rand_w),boxes[0,0]+rand_w)
    y1 = random.randint(max(0, boxes[0,1]-rand_w),boxes[0,1]+rand_w)
    x2 = random.randint(boxes[0,2]-rand_w, min(width-1, boxes[0,2]+rand_w))
    y2 = random.randint(boxes[0,3]-rand_h, min(height-1, boxes[0,3]+rand_h))

    image = image[y1:y2, x1:x2, :]
    height, width, _ = image.shape
    landm[:, 0::2] = landm[:, 0::2] - x1
    landm[:, 1::2] = landm[:, 1::2] - y1
    return image, landm



def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, key_point_number1, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, key_point_number1 * 2])

    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    # image -= rgb_mean
    image /= 255
    return image


class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        idx = random.randint(0,targets.shape[0]-1)#从anno中随机取一个instance
        targets = targets[idx,:].reshape((1, -1))
        boxes = targets[:, 1:5].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 5:-1].copy()
        isswitch=int(targets[0].copy()[0])
        label_switch=np.ones((1, 1)) * (0)
        label_switch=labels
        
        image_t, landm_t = _rand_crop(image, boxes, labels, landm, self.img_dim,rand_ratio=0.1)
        image_t_cv=image_t
        image_t = _distort(image_t)
        #image_t = _pad_to_square(image_t,self.rgb_means, 1)
        # image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height
        return image_t,image_t_cv,label_switch