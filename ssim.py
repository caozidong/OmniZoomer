import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from utils import rgb2y

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True, ordinary=False):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    if ordinary:
        C1 = 6.5025
        C2 = 58.5225
    else:
        C1 = 0.01**2
        C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    mw = np.load('/home/ps/data/zidongcao/OmniZoomer/mw.npy')
    mw = torch.from_numpy(mw).cuda()

    ws_SSIM = torch.sum(ssim_map * mw) / (torch.sum(mw) + 1e-4)

    return ws_SSIM.item()

def ssim(img1, img2, window_size = 11, size_average = True, ordinary = False):
    '''Ordinary: If True, calculate SSIM for the common SR task.
                 If False, calculate SSIM for the Mobius SR task.'''
    if ordinary:
        img1 = img1 * 255.0
        img2 = img2 * 255.0
        img1 = img1.clamp(0, 255).round()
        img2 = img2.clamp(0, 255).round()
        img1 = rgb2y(img1.cpu().data).unsqueeze(1).cuda()
        img2 = rgb2y(img2.cpu().data).unsqueeze(1).cuda()

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average, ordinary)