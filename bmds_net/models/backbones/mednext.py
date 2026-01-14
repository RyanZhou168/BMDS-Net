"""
MedNeXt Baseline Model.
A modern ConvNeXt-inspired architecture for 3D medical image segmentation.
"""

import torch
import torch.nn as nn

class MedNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=True):
        super().__init__()
        self.do_res = do_res
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels)
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        self.conv2 = nn.Conv3d(in_channels, exp_r*in_channels, kernel_size=1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(exp_r*in_channels, out_channels, kernel_size=1)
        
        if in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        x1 = self.conv3(self.act(self.conv2(self.norm(self.conv1(x)))))
        return x1 + self.res_conv(x) if self.do_res else x1

class MedNeXt(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, model_id='B', kernel_size=3):
        super().__init__()
        # Config for 'B' (Base) variant
        cfg = {'channels': [32, 64, 128, 256, 512], 'exp_r': 4} if model_id == 'B' else \
              {'channels': [32, 64, 128, 256, 320], 'exp_r': 2} # S variant
        
        c = cfg['channels']
        k = kernel_size
        r = cfg['exp_r']
        
        self.stem = nn.Conv3d(in_channels, c[0], kernel_size=1)
        self.enc1 = MedNeXtBlock(c[0], c[0], r, k)
        self.down1 = nn.Sequential(nn.Conv3d(c[0], c[1], 2, 2), MedNeXtBlock(c[1], c[1], r, k))
        self.down2 = nn.Sequential(nn.Conv3d(c[1], c[2], 2, 2), MedNeXtBlock(c[2], c[2], r, k))
        self.down3 = nn.Sequential(nn.Conv3d(c[2], c[3], 2, 2), MedNeXtBlock(c[3], c[3], r, k))
        self.bottleneck = nn.Sequential(nn.Conv3d(c[3], c[4], 2, 2), MedNeXtBlock(c[4], c[4], r, k))
        
        self.up3 = nn.ConvTranspose3d(c[4], c[3], 2, 2)
        self.dec3 = MedNeXtBlock(c[3]*2, c[3], r, k)
        self.up2 = nn.ConvTranspose3d(c[3], c[2], 2, 2)
        self.dec2 = MedNeXtBlock(c[2]*2, c[2], r, k)
        self.up1 = nn.ConvTranspose3d(c[2], c[1], 2, 2)
        self.dec1 = MedNeXtBlock(c[1]*2, c[1], r, k)
        self.up0 = nn.ConvTranspose3d(c[1], c[0], 2, 2)
        self.dec0 = MedNeXtBlock(c[0]*2, c[0], r, k)
        
        self.out = nn.Conv3d(c[0], out_channels, 1)

    def forward(self, x):
        s0 = self.enc1(self.stem(x))
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bottleneck(s3)
        
        d3 = self.dec3(torch.cat([self.up3(b), s3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], 1))
        d0 = self.dec0(torch.cat([self.up0(d1), s0], 1))
        
        return self.out(d0)
