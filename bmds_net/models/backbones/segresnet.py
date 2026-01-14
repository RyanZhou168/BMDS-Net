"""
SegResNet Baseline Model.
Standard implementation from MONAI, widely used as a CNN baseline for BraTS.
"""

import torch.nn as nn
from monai.networks.nets import SegResNet

class SegResNetBaseline(nn.Module):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 init_filters=32, 
                 dropout_prob=0.2):
        super().__init__()
        
        self.model = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=dropout_prob
        )
    
    def forward(self, x):
        return self.model(x)
