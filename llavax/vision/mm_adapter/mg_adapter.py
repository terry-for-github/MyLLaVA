import math

import torch
from torch import nn
import torch.nn.functional as F


class MGAdapter(nn.Module):
    # change channel+gate+sum
    def __init__(self, low_res_dim, high_res_dim, zero_init=True):
        super().__init__()

        self.slow_conv = nn.Conv2d(high_res_dim, high_res_dim, 1)
        self.slow_proj = nn.Conv2d(high_res_dim, low_res_dim, 1)

        self.fast_conv = nn.Conv2d(low_res_dim, low_res_dim, 7, padding=3, groups=low_res_dim)
        self.fast_proj = nn.Conv2d(low_res_dim, low_res_dim, 1)

        self.gate = nn.Sequential(
            nn.Linear(low_res_dim * 2, low_res_dim // 2),
            nn.GELU(),
            nn.Linear(low_res_dim // 2, 1),
        )

        nn.init.xavier_uniform_(self.slow_conv.weight)
        nn.init.xavier_uniform_(self.fast_conv.weight)
        nn.init.zeros_(self.slow_conv.bias)
        nn.init.zeros_(self.fast_conv.bias)
        if zero_init:
            nn.init.zeros_(self.slow_proj.weight)
            nn.init.zeros_(self.fast_proj.weight)
        else:
            nn.init.xavier_uniform_(self.slow_proj.weight)
            nn.init.xavier_uniform_(self.fast_proj.weight)
        nn.init.zeros_(self.slow_proj.bias)
        nn.init.zeros_(self.fast_proj.bias)

    def forward(self, features): # batch 
        low_res_feat, high_res_feat, object_feat = features
        ## low_res_feat  clip 
        ## high_res_feat dino ?
        ## object_feat [sgg+ocr] ? ?
        b, t, h_dim = high_res_feat.shape ## t token 576 24 * 24
        _, _, l_dim = low_res_feat.shape ### _, 最后一样
        # high res
        dst_size = int(math.sqrt(high_res_feat.shape[1] + 0.1)) # 高宽
        high_res_feat = high_res_feat.transpose(1, 2).view(b, h_dim, dst_size, dst_size)
        high_res_feat = self.slow_proj(F.gelu(self.slow_conv(high_res_feat)))
        high_res_feat = high_res_feat.view(b, l_dim, -1).transpose(1, 2)
        # low res
        dst_size = int(math.sqrt(low_res_feat.shape[1] + 0.1))
        low_res_feat = low_res_feat.transpose(1, 2).view(b, l_dim, dst_size, dst_size)
        low_res_feat = low_res_feat + self.fast_proj(F.gelu(self.fast_conv(low_res_feat)))
        low_res_feat = low_res_feat.view(b, l_dim, -1).transpose(1, 2)
        # gate
        gate = self.gate(torch.cat([low_res_feat, high_res_feat], -1).mean(1)).unsqueeze(1)

        #### object_feat sgg guide cross attention image-level feature ocr-sgg object-level
        ### 高低分辨率 富集？ MLP 缺点：视觉特征的融合，将token做平姐，token层级融合。
        return torch.cat([low_res_feat + high_res_feat * gate.tanh(), object_feat], dim=1)
