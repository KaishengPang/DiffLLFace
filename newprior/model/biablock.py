import torch.nn as nn
import torch
from newprior.model import common
import torch.nn.functional as F
import cv2
import os
import numpy as np

forward_count = 0

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()
    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()
        N, C, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) 
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N,  1, 1).unsqueeze(3) / (H - 1) * 2 - 1 
        wg = wg.float().repeat(N, 1,1).unsqueeze(3) / (W - 1) * 2 - 1  
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([hg, wg, guidemap], dim=3).unsqueeze(1) 
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)



class ApplyCoeffs_adIllu(nn.Module):
    def __init__(self):
        super(ApplyCoeffs_adIllu, self).__init__()

    def forward(self, coeff, full_res_input):

        res = full_res_input*coeff*(1-full_res_input) + full_res_input
        return res

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
    
class BasicBlock2(nn.Module):
    def __init__(self):
        super(BasicBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)


        self.slice = Slice()
        self.adjust = nn.Sequential(*[nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.Sigmoid()])
        #conv+sig
        self.apply_coeffs = ApplyCoeffs_adIllu()
        self.conv1280 = nn.Conv2d(64, 1280, kernel_size=1, stride=1, padding=0)

    def forward(self, x, guide, coeffs):
        
        slice_coeffs = self.slice(coeffs, guide)      
        slice_coeffs = self.adjust(slice_coeffs)
        slice_coeffs = self.conv1280(slice_coeffs)
        slice_coeffs = F.interpolate(slice_coeffs, size=(4, 4), mode='bilinear', align_corners=False)
        C_t = slice_coeffs
        return C_t
    
class BiaGroup(nn.Module):
    def __init__(self, n_resblocks):
        super(BiaGroup, self).__init__()
        modules_body = []
        for _ in range(n_resblocks):
            modules_body.append(BasicBlock2())
        self.tail = common.ConvBNReLU2D(64, 64, kernel_size=3, padding=1, act='relu')
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x, guidance, grid):
        res = x
        for i in range(len(self.body)):
            res = self.body[i](res, guidance, grid)
        return res
class Biablock(nn.Module):
    def __init__(self, n_resblocks):
        super(Biablock, self).__init__()
        self.body = BiaGroup(n_resblocks)
    def forward(self, x, guide, coeffs):
        res = self.body(x, guide, coeffs)
        return res