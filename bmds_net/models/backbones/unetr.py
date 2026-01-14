"""
UNETR Baseline Model.
Standard implementation from MONAI for comparison purposes.
"""

import torch.nn as nn
from monai.networks.nets import UNETR

class UNETRBaseline(nn.Module):
    def __init__(self, 
                 img_size=(128, 128, 128),
                 in_channels=4,
                 out_channels=4,
                 feature_size=16,
                 hidden_size=768,
                 mlp_dim=3072,
                 num_heads=12,
                 dropout_rate=0.0):
        super().__init__()
        
        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=dropout_rate
        )
        
    def forward(self, x):
        return self.model(x)
