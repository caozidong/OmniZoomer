import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from datasets import register
from utils import to_pixel_samples

@register('sr-paired')
class SRPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None, test=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        h_lr, w_lr = img_lr.shape[-2:]
        img_hr = img_hr[:, :h_lr * s, :w_lr * s]
        crop_lr, crop_hr = img_lr, img_hr

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x
           
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        
        lr_coord, _ = to_pixel_samples(crop_lr.contiguous())
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        return {
            'inp': crop_lr,
            'coord_lr': lr_coord,
            'coord_hr': hr_coord,
            'gt': hr_rgb
        }
