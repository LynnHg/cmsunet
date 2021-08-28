import cv2
import math
import sys
import numbers
import random
import numpy as np
from PIL import Image, ImageOps


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if (self.size <= img.shape[0]) and (self.size <= img.shape[1]):
            x = math.ceil((img.shape[0] - self.size) / 2.)
            y = math.ceil((img.shape[1] - self.size) / 2.)

            if len(img.shape) == 3:
                return img[x:x + self.size, y:y + self.size, :]
            else:
                return img[x:x + self.size, y:y + self.size]
        else:
            raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
                self.size, self.size, img.shape[0], img.shape[1]))


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if (self.size <= img.shape[0]) and (self.size <= img.shape[1]):
            x = img.shape[0] - self.size
            y = img.shape[1] - self.size
            offsetx = np.random.randint(x)
            offsety = np.random.randint(y)

            if len(img.shape) == 3:
                return img[offsetx:offsetx + self.size, offsety:offsety + self.size, :]
            else:
                return img[offsetx:offsetx + self.size, offsety:offsety + self.size]
        else:
            raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
                self.size, self.size, img.shape[0], img.shape[1]))


class RanddomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        flag = np.random.randint(2)
        if flag:
            M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), self.angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
            return img
        return img