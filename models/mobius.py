import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import profile


@register('mobius')
class Mobius(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

    def gen_feat(self, inp, coord_t):
        self.feat = self.encoder(inp, coord_t)
        return self.feat

    def query_rgb(self):
        feat = self.feat
        self.ret = torch.flatten(feat, start_dim=2, end_dim=3).permute(0, 2, 1)
        return self.ret

    def forward(self, inp, coord_t):
        self.gen_feat(inp, coord_t)
        return self.query_rgb()
