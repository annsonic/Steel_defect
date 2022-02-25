import cv2
import numpy as np
import random
import math
import os


class GridMask(object):
    def __init__(self, mode=1, rotate=1, r_ratio=0.5):
        self.mode = mode
        self.rotate = rotate
        self.r_ratio = r_ratio
    
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
        image[mask == 0] = (255, 255, 255)
        
        return image
    
    def __call__(self, image, boxes, labels):
        if random.random() < 1:
            image = self.put_grid(image.copy())

        return image, boxes, labels
    

IMG_DIR = os.path.join('dataset', 'VOC2012', 'JPEGImages')
path_img = os.path.join(IMG_DIR, 'synth_1757.jpg')
img = cv2.imread(path_img)

transform = GridMask(mode=1, rotate=45, r_ratio=0.5)
img2, _, _ = transform(img, None, None)

merged = np.hstack([img, img2])
merged = cv2.resize(merged, None, fx=0.5, fy=0.5)
cv2.imshow('viz', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()