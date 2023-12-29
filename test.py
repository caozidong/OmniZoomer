import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

import datasets
import models
import utils
import mobius_utils
import ssim
import profile
import time


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False,
              verbose=False, test_bicubic=False, test_ssim=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('osr'):
        scale = int(eval_type.split('-')[1])
        if test_ssim:
            metric_fn = ssim.ssim
        else:
            metric_fn = utils.calc_psnr_ws
        
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    idx = 0
    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div

        M = torch.load('./' + str(idx).zfill(3) + '.pt').cuda()

        '''For testing common SR task'''
        # M = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128).cuda()
        
        gt = mobius_utils.warp_mobius_image(batch['gt'], M, batch['coord_hr'])

        coord_t_hr = mobius_utils.warp_mobius_coord(batch['coord_hr'], M, \
                                        batch['coord_hr'], get_sphere=False)

        '''Check if testing the baseline of Bicubic interpolation'''
        if test_bicubic:
            inp = inp * gt_div + gt_sub
            hr_pred = torch.nn.functional.interpolate(inp, scale_factor=int(eval_type.split('-')[1]), mode="bicubic").view(1, 3, -1).permute(0, 2, 1)
            pred = mobius_utils.warp_mobius_image(hr_pred, M, batch['coord_hr'])
            # pred.clamp_(0, 1)
        else:
            with torch.no_grad():
                pred = model(inp, coord_t_hr)
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)

        '''Check if saving the predictions'''
        # pic = transforms.ToPILImage()(pred[0].view(1024, 2048, 3).permute(2, 0, 1))
        # pic.save('./' + str(idx).zfill(3) + '.png')

        if eval_type is not None and fast == False: # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord_hr'].shape[1] / batch['coord_lr'].shape[1])
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            gt = gt.view(*shape) \
                .permute(0, 3, 1, 2).contiguous() # (B x 3 x H x W)
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
        res = metric_fn(pred, gt)
        # print(idx, res)
        if eval_type.startswith('osr'):
            val_res.add(res, inp.shape[0])
        else:
            val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
        
        idx = idx + 1
            
    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--fast', default=False)
    parser.add_argument('--gpu', default='0', help="available gpus")
    parser.add_argument('--bicubic', action="store_true", help="if true, test the bicubic result")
    parser.add_argument('--ssim', action="store_true", help="if true, test ssim metric, else test psnr metric")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        window_size=int(args.window),
        scale_max = int(args.scale_max),
        fast = args.fast,
        verbose=True,
        test_bicubic=args.bicubic,
        test_ssim=args.ssim)
    print('result: {:.4f}'.format(res))

    # python test.py --config configs/test-osr/test-osr-8.yaml --model save/_train_rcan_skip-x8/epoch-best.pth --gpu 3