# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import math
import random

import cv2
import numpy as np

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box
from utils.metrics import bbox_ioa


class GridMask(object):
    def __init__(self, mode=1, rotate=1, r_ratio=0.5):
        self.mode = mode
        self.rotate = rotate
        self.r_ratio = r_ratio
        self.colors = [(0, 0, 0), (127, 127, 127), (255, 255, 255)]
    
    def rotate_bound(self, image, angle):
        # CREDIT: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH))

    def put_grid(self, image):
        h, w, _ = image.shape

        # Size of mask
        L = math.ceil((math.sqrt(h*h + w*w))) * 3
        # Size of (black cell + white cell)
        d = np.random.randint(2, min(h,w))
        # Size of a white cell
        l = min(max(int(d * self.r_ratio + 0.5), 1), d-1)
        # Create mask
        mask = np.ones((L, L), np.float32)
        # Size of black cell
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(L//d):
            s = d*i + st_h
            t = min(s+l, L)
            mask[s:t,:] *= 0
    
        for i in range(L//d):
                s = d*i + st_w
                t = min(s+l, L)
                mask[:,s:t] *= 0
        # Rotate mask
        r = np.random.randint(self.rotate)
        mask = self.rotate_bound(mask, angle=r)
        # Fit mask to size of input image
        mask = mask[(L-h)//2:(L-h)//2+h, (L-w)//2:(L-w)//2+w]

        if self.mode == 1:
            mask = 1  -mask
        # Mask th input image
        color = random.choice(self.colors)
        image[mask == 0] = color
        
        return image
    
    def __call__(self, image):
        if random.random() < 1:
            image = self.put_grid(image.copy())

        return image


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            self.transform = A.Compose(
                [
                    # A.Blur(blur_limit=3, p=0.5), 
                    A.MedianBlur(blur_limit=3, p=0.5),
                    # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8),p=0.5), 
                    A.RandomBrightnessContrast(brightness_by_max=True, p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.ImageCompression(quality_lower=75, p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.GaussNoise(var_limit=(1.0, 100.0), mean=0, per_channel=True, p=0.5),
                    A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, p=0.5),
                    A.ToGray(p=1),
                ], 
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
                p=1
            )
            
            self.spatial_transform = A.OneOf(
                [
                    A.ElasticTransform(
                        alpha = 75,
                        sigma = 30,
                        alpha_affine = 1,
                        border_mode=4,
                        interpolation=2, 
                        p=0.5
                    ),
                    A.GridDistortion(
                        num_steps=8, 
                        distort_limit=0.35, 
                        border_mode=4,
                        interpolation=2, 
                        p=0.5),
                    A.OpticalDistortion(
                        distort_limit=0.15, 
                        shift_limit=0.2, 
                        border_mode=4,
                        interpolation=2, 
                        p=0.5),
                ], p=1
            )

            self.large_spatial_transform = A.OneOf(
                [
                    A.ElasticTransform(
                        alpha = 60,
                        sigma = 20,
                        alpha_affine = 10,
                        border_mode=4,
                        interpolation=2, 
                        p=0.5
                    ),
                    A.GridDistortion(
                        num_steps=10, 
                        distort_limit=0.5, 
                        border_mode=4,
                        interpolation=2, 
                        p=0.5),
                    A.OpticalDistortion(
                        distort_limit=0.3, 
                        shift_limit=0.4, 
                        border_mode=4,
                        interpolation=2, 
                        p=0.5),
                ], p=1
            )

            self.grid_mask = GridMask(mode=1, rotate=45, r_ratio=0.75)

            LOGGER.info(colorstr('albumentations pixel: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
            LOGGER.info(colorstr('albumentations spatial: ') + ', '.join(f'{x}' for x in self.spatial_transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def _autocontrast(self, img):
        """
        source: https://stackoverflow.com/questions/66917293/autocontrast-in-albumentations
        """
        h = cv2.calcHist([img], [0], None, [256], (0, 256)).ravel()

        for lo in range(256):
            if h[lo]:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break

        if hi > lo:
            lut = np.zeros(256, dtype=np.uint8)
            scale_coef = 255.0 / (hi - lo)
            offset = -lo * scale_coef
            for ix in range(256):
                lut[ix] = int(np.clip(ix * scale_coef + offset, 0, 255))

            img = cv2.LUT(img, lut)

        return img

    def __call__(self, im, labels, p=1.0):
        # if random.random() < 0.5:
            # im = self._autocontrast(im)

        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

        # DO spatial transform but keep box size
        if self.spatial_transform and random.random() < p:
            h, w = im.shape[:2]
            for label in labels: # label: np.array shape (5,)
                xc = int(label[1] * w)
                yc = int(label[2] * h)
                bw = int(label[3] * w)
                bh = int(label[4] * h)
                # box
                xmin = max(0, xc - bw // 2)
                ymin = max(0, yc - bh // 2)
                xmax = min(w, xmin + bw)
                ymax = min(h, ymin + bh)

                # Crop RoI
                src = im[ymin:ymax, xmin:xmax, :]
                if bw > 100 and bh > 100:
                    new = self.large_spatial_transform(image=src)
                else:
                    new = self.spatial_transform(image=src)
                dst = new['image']
                # Paste
                im[ymin:ymax, xmin:xmax, :] = dst

        # Do GridMask dataaugmentation
        if random.random() < 0.5:
            im = self.grid_mask(im)

        return im, labels


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
