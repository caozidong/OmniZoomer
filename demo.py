import argparse
import os
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

import utils
import mobius_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--use_pretrain', action="store_true", help="if false, do not need pre-trained models")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    if args.use_pretrain:
        model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    h = int(img.shape[-2] * int(args.scale))
    w = int(img.shape[-1] * int(args.scale))
    scale = h / img.shape[-2]
    
    # Set horizontal (theta) and vertical (phi) angles, and zoom level 
    # (the first element in the M_scale matrix)
    theta = 0
    phi = 0
    M_scale = np.array([[2, 0], [0, 1]])
    M_horizon = np.array([[np.cos(theta) + 1j * np.sin(theta), 0], [0, 1]])
    M_vertical = np.array([[np.cos(phi / 2), np.sin(phi / 2)], [-np.sin(phi / 2), np.cos(phi / 2)]])
    M = M_horizon @ M_vertical @ M_scale
    M = torch.from_numpy(M).cuda()

    coord_hr = make_coord([h, w], flatten=True).unsqueeze(0)
    coord_t_hr = mobius_utils.warp_mobius_coord(coord_hr, M, \
                                    coord_hr, get_sphere=False, pole='Equator')
    
    if not args.use_pretrain:
        hr_pred = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale, \
                                              mode="bicubic").view(1, 3, -1).permute(0, 2, 1)
        pred = mobius_utils.warp_mobius_image(hr_pred.cuda(), M, coord_hr.cuda(), pole='Equator')
        transforms.ToPILImage()(pred[0].clamp(0, 1).view(1024, 2048, 3).permute(2, 0, 1)).save(args.output)
    else:
        with torch.no_grad():
            pred = model(((img - 0.5) / 0.5).cuda().unsqueeze(0), coord_t_hr)
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
        transforms.ToPILImage()(pred).save(args.output)

    # python demo.py --input /home/ps/data/zidongcao/Dataset/lau_dataset/odisr/validation/LR/X8/002.jpg --model save/_train_rcan_skip-x8/epoch-best.pth --scale 8 --output test.png --gpu 6