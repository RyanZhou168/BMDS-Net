"""
Residue-Gated Deep Decoder Supervision (DDS) Module.
Key Innovation: Uses the global attention map from MMCF to gate the 
deep decoder features, enforcing semantic consistency and boundary precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidueGatedDDSHead(nn.Module):
    """
    DDS Head applied at decoder stages (e.g., 32x, 64x downsampling).
    """
    def __init__(self, dds_channels, mmcf_channels):
        super().__init__()
        
        # Learnable scaling factor for the gate
        self.alpha = nn.Parameter(torch.tensor(0.1))  
        
        # Project MMCF weights to match decoder feature channels
        self.weight_proj = nn.Sequential(
            nn.Conv3d(mmcf_channels, dds_channels, 1),
            nn.InstanceNorm3d(dds_channels),
            nn.Sigmoid()
        )
        
        # Refinement convolution after gating
        self.conv_out = nn.Conv3d(dds_channels, dds_channels, 3, padding=1)
        self.norm = nn.InstanceNorm3d(dds_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Init
        nn.init.kaiming_normal_(self.weight_proj[0].weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, dds_feature, mmcf_weights):
        """
        Args:
            dds_feature: Decoder feature map [B, C_dec, H', W', D']
            mmcf_weights: Attention map from MMCF [B, 4, H, W, D]
        """
        # Project weights to decoder channel space
        weight_map = self.weight_proj(mmcf_weights)
        
        # Residue Gating: Feature' = Feature * (1 + alpha * weight)
        # 1.0 ensures gradient flow even if weight is 0.
        gate = 1 + self.alpha * torch.sigmoid(weight_map)
        gated_feature = dds_feature * gate
        
        # Refine
        out = self.relu(self.norm(self.conv_out(gated_feature)))
        return out


class BidirectionalDistillationLoss(nn.Module):
    """
    Consistecy loss between Encoder Attention (MMCF) and Decoder Activations (DDS).
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, mmcf_weights, dds_features):
        """
        Args:
            mmcf_weights: [B, 4, H, W, D]
            dds_features: List of decoder features at different scales.
        """
        total_distill_loss = 0.0
        
        for idx, dds_feat in enumerate(dds_features):
            # 1. Calculate Activation Map of decoder features (L2 Norm across channels)
            activation_map = torch.norm(dds_feat, p=2, dim=1, keepdim=True)
            # Normalize
            activation_map = activation_map / (activation_map.mean() + 1e-8)
            
            # 2. Downsample MMCF weights to match decoder resolution
            target_size = dds_feat.shape[2:]
            weight_map_down = F.interpolate(
                mmcf_weights, size=target_size, mode='trilinear', align_corners=False
            )
            # Collapse channels (L2 Norm)
            weight_map_down = torch.norm(weight_map_down, p=2, dim=1, keepdim=True)
            # Normalize
            weight_map_down = weight_map_down / (weight_map_down.mean() + 1e-8)
            
            # 3. MSE Consistency Loss
            consistency_loss = F.mse_loss(weight_map_down, activation_map)
            
            # 4. Sparsity regularization
            sparsity_loss = weight_map_down.mean()
            
            # Weighted sum (deeper layers get less weight)
            level_weight = 0.1 * (2 ** -idx)
            total_distill_loss += level_weight * (consistency_loss + 0.01 * sparsity_loss)
        
        return total_distill_loss
