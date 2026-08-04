"""
Zero-Init Multimodal Contextual Fusion (MMCF) Module.
Key Innovation: Uses a zero-initialized residual connection to ensure the model 
starts as an identity mapping (equivalent to baseline) and gradually learns 
cross-modal attention.
"""

import torch
import torch.nn as nn

class ZeroInitMMCF(nn.Module):
    """
    Zero-initialized Multimodal Contextual Fusion Module.
    
    Args:
        in_channels (int): Number of input modalities (default: 4).
        base_channels (int): Internal channel dimension for the lightweight encoder.
    """
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        
        # Lightweight Encoder to extract modality-specific features
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_channels),
            nn.PReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_channels),
            nn.PReLU()
        )
        
        # Branch 1: Attention Map Generation
        # Output: [B, C, H, W, D], values in [0, 1]
        self.attention_conv = nn.Conv3d(base_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Learnable scalar initialized to 0 (Zero-Init)
        # This ensures x_fused = x at the start of training.
        self.alpha = nn.Parameter(torch.zeros(1)) 

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Explicitly zero-out the last attention layer to ensure smooth start
        nn.init.constant_(self.attention_conv.weight, 0)
        nn.init.constant_(self.attention_conv.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 4, H, W, D]
        Returns:
            x_fused: Contextually fused features.
            att_map: Attention weights (for visualization/distillation).
        """
        feat = self.encoder(x)
        
        # Generate Attention Map
        att_map = self.sigmoid(self.attention_conv(feat))

        # Residual Fusion: x + alpha * (x * attention)
        x_fused = x + self.alpha * (x * att_map)
        
        return x_fused, att_map


# Backward-compatible alias for earlier public releases.
UncertaintyAwareMMCF = ZeroInitMMCF
