import cv2
import math
import sys
import numbers
import random
from PIL import Image, ImageOps
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import helpers


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class SingleCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop_npy(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.shape == mask.shape
        if (self.size <= img.shape[1]) and (self.size <= img.shape[0]):
            x = math.ceil((img.shape[1] - self.size) / 2.)
            y = math.ceil((img.shape[0] - self.size) / 2.)

            if len(mask.shape) == 3:
                return img[y:y + self.size, x:x + self.size, :], mask[y:y + self.size, x:x + self.size, :]
            else:
                return img[y:y + self.size, x:x + self.size, :], mask[y:y + self.size, x:x + self.size]
        else:
            raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
                self.size, self.size, img.shape[0], img.shape[1]))


class ROICenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def ConnectComponent(self, bw_img):
        labeled_img, num = measure.label(bw_img, background=0, connectivity=1, return_num=True)

        lcc = np.array(labeled_img, dtype=np.uint8)
        return lcc, num

    def get_bounding_box(self, mask):
        stan = np.array(helpers.array_to_img(np.expand_dims(mask, axis=2)))
        max_value = np.max(stan)
        ret, bmask = cv2.threshold(stan, 1, max_value, cv2.THRESH_BINARY)
        lcc, _ = self.ConnectComponent(bmask)
        props = measure.regionprops(lcc)
        MINC, MINR, MAXC, MAXR = sys.maxsize, sys.maxsize, 0, 0
        for i in range(len(props)):
            minr, minc, maxr, maxc = props[i].bbox
            MINC, MINR = min(MINC, minc), min(MINR, minr)
            MAXC, MAXR = max(MAXC, maxc), max(MAXR, maxr)

        return MINR, MINC, MAXR, MAXC

    def __call__(self, img, mask):
        assert img.shape == mask.shape
        th, tw = self.size

        minr, minc, maxr, maxc = self.get_bounding_box(mask)

        w = maxc - minc
        h = maxr - minr
        offset_w = math.ceil((th - w) / 2.)
        offset_h = math.ceil((th - h) / 2.)
        col, row = minc - offset_w, minr - offset_h

        # visualization
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(mask, cmap='gray')
        # bbox = Rectangle((col, row), tw, th, fill=False, edgecolor='blue', linewidth=2)
        # ax.add_patch(bbox)
        # plt.show()

        return img[row:row+th, col:col+tw], mask[row:row+th, col:col+tw]


# just test
class ROICrop(object):

    def __init__(self):
        pass

    def ConnectComponent(self, bw_img):
        labeled_img, num = measure.label(bw_img, background=0, connectivity=1, return_num=True)

        lcc = np.array(labeled_img, dtype=np.uint8)
        return lcc, num

    def get_bounding_box(self, mask):
        stan = np.array(helpers.array_to_img(np.expand_dims(mask, axis=2)))
        max_value = np.max(stan)
        ret, bmask = cv2.threshold(stan, 1, max_value, cv2.THRESH_BINARY)
        lcc, _ = self.ConnectComponent(bmask)
        props = measure.regionprops(lcc)
        MINC, MINR, MAXC, MAXR = sys.maxsize, sys.maxsize, 0, 0
        for i in range(len(props)):
            minr, minc, maxr, maxc = props[i].bbox
            MINC, MINR = min(MINC, minc), min(MINR, minr)
            MAXC, MAXR = max(MAXC, maxc), max(MAXR, maxr)

        return MINR, MINC, MAXR, MAXC

    def __call__(self, img, mask):
        assert img.shape == mask.shape

        minr, minc, maxr, maxc = self.get_bounding_box(mask)

        w = maxc - minc
        h = maxr - minr
        leader = max(w, h)

        crop_size = math.ceil(leader / 16) * 16

        col, row = minc, minr

        # visualization
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(mask, cmap='gray')
        # bbox = Rectangle((col, row), crop_size, crop_size, fill=False, edgecolor='blue', linewidth=2)
        # ax.add_patch(bbox)
        # plt.show()

        return img[row:row+crop_size, col:col+crop_size], mask[row:row+crop_size, col:col+crop_size]


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size=0, scale_rate=0.95, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_rate = scale_rate
        self.fill = fill

    def __call__(self, im, gt):
        img = im.copy()
        mask = gt.copy()
        # random scale (short edge)
        short_size = int(self.base_size * self.scale_rate)
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        # x1 = random.randint(0, w - self.crop_size)
        # y1 = random.randint(0, h - self.crop_size)
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        tw = th = self.crop_size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        img, mask = img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))
        return img, mask


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))






