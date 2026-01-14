"""
Swin UNETR Baseline Model.

Standard implementation wrapping MONAI's SwinUNETR.
Used as the primary baseline for comparison in Table 1.
"""

import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRBaseline(nn.Module):
    """
    Wrapper for MONAI SwinUNETR to match the factory interface.
    """
    def __init__(self, 
                 img_size=(128, 128, 128),
                 in_channels=4,
                 out_channels=4,
                 feature_size=48,
                 depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24),
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 dropout_path_rate=0.0,
                 use_checkpoint=True,
                 spatial_dims=3):
        
        super().__init__()
        
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims
        )
        
    def forward(self, x):
        return self.model(x)
