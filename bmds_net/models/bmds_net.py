"""
BMDS-Net Main Architecture.
Combines SwinUNETR backbone with MMCF and DDS modules.
Supports both Deterministic (Stage 1) and Bayesian (Stage 2) modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

from .modules.mmcf import UncertaintyAwareMMCF
from .modules.dds import ResidueGatedDDSHead
from .modules.bayesian_layers import BayesianConv3d

class BMDSNet(nn.Module):
    """
    Deterministic BMDS-Net (Stage 1).
    Integrates MMCF at input and DDS at decoder.
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
                 use_checkpoint=True):
        
        super().__init__()
        self.img_size = img_size
        
        # 1. MMCF Module (Input Fusion)
        self.mmcf = UncertaintyAwareMMCF(in_channels=in_channels)
        
        # 2. Backbone (SwinUNETR)
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=3
        )
        
        # 3. DDS Modules (Deep Supervision)
        # Applied at 32x (decoder4) and 64x (decoder3) downsampling features
        self.dds_head_32 = ResidueGatedDDSHead(feature_size * 4, in_channels)
        self.dds_head_64 = ResidueGatedDDSHead(feature_size * 2, in_channels)
        
        # Auxiliary Classification Heads
        self.aux_head_32 = nn.Conv3d(feature_size * 4, out_channels, 1)
        self.aux_head_64 = nn.Conv3d(feature_size * 2, out_channels, 1)
        
        self._dds_features = [] # Store for distillation loss

    def forward(self, x):
        # 1. MMCF Fusion
        x_fused, att_map, uncertainty = self.mmcf(x)
        
        # 2. Encoder (SwinViT)
        hidden_states_out = self.swin_unetr.swinViT(x_fused)
        enc0 = self.swin_unetr.encoder1(x_fused)
        enc1 = self.swin_unetr.encoder2(hidden_states_out[0])
        enc2 = self.swin_unetr.encoder3(hidden_states_out[1])
        enc3 = self.swin_unetr.encoder4(hidden_states_out[2])
        dec4 = self.swin_unetr.encoder10(hidden_states_out[4])
        
        # 3. Decoder with DDS Hooks
        dec3 = self.swin_unetr.decoder5(dec4, hidden_states_out[3])
        dec2 = self.swin_unetr.decoder4(dec3, enc3)
        dec1 = self.swin_unetr.decoder3(dec2, enc2)
        dec0 = self.swin_unetr.decoder2(dec1, enc1)
        out_final = self.swin_unetr.decoder1(dec0, enc0)
        
        # Main Output
        logits_final = self.swin_unetr.out(out_final)
        
        # If in training mode, compute auxiliary outputs
        if self.training:
            # Interpolate attention weights for DDS gating
            w_32 = F.interpolate(att_map, size=dec2.shape[2:], mode='trilinear')
            w_64 = F.interpolate(att_map, size=dec1.shape[2:], mode='trilinear')
            
            # Apply DDS Gating
            feat_32_gated = self.dds_head_32(dec2, w_32)
            feat_64_gated = self.dds_head_64(dec1, w_64)
            
            # Aux Heads
            logits_32 = self.aux_head_32(feat_32_gated)
            logits_64 = self.aux_head_64(feat_64_gated)
            
            # Upsample Aux outputs
            logits_32_up = F.interpolate(logits_32, size=self.img_size, mode='trilinear')
            logits_64_up = F.interpolate(logits_64, size=self.img_size, mode='trilinear')
            
            # Store features for distillation
            self._dds_features = [dec1, dec2]
            
            return logits_final, logits_64_up, logits_32_up, att_map, uncertainty
        
        return logits_final

    def get_dds_features(self):
        return self._dds_features


class BayesianBMDSNet(nn.Module):
    """
    Bayesian Wrapper for BMDS-Net (Stage 2).
    Replaces the final output layer with a Bayesian Conv3d layer.
    """
    def __init__(self, deterministic_model: BMDSNet, mc_dropout_rate=0.1):
        super().__init__()
        self.backbone = deterministic_model
        self.mc_dropout_rate = mc_dropout_rate
        
        # Replace the final layer
        self._replace_final_layer()
        
    def _replace_final_layer(self):
        # Locate the original output layer (SwinUNETR.out is usually UnetOutBlock -> Conv3d)
        original_out = self.backbone.swin_unetr.out
        
        # Find the Conv3d layer inside UnetOutBlock
        conv_layer = None
        if isinstance(original_out, nn.Conv3d):
            conv_layer = original_out
        elif hasattr(original_out, 'conv') and isinstance(original_out.conv, nn.Conv3d):
            conv_layer = original_out.conv
        
        if conv_layer is None:
            raise ValueError("Could not locate final Conv3d layer in SwinUNETR.")
            
        # Create Bayesian Layer
        bayesian_conv = BayesianConv3d(
            in_channels=conv_layer.in_channels,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size[0]
        )
        
        # Weight Transfer (Initialize Bayesian Mu with Deterministic Weights)
        with torch.no_grad():
            bayesian_conv.weight_mu.data.copy_(conv_layer.weight.data)
            if conv_layer.bias is not None:
                bayesian_conv.bias_mu.data.copy_(conv_layer.bias.data)
        
        # Perform replacement
        if isinstance(original_out, nn.Conv3d):
            self.backbone.swin_unetr.out = bayesian_conv
        else:
            original_out.conv = bayesian_conv
            
    def forward(self, x):
        # Apply MC Dropout during inference if needed
        if self.mc_dropout_rate > 0:
            for m in self.backbone.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    m.p = self.mc_dropout_rate
                    m.train()
        
        # Forward pass through backbone
        # In Stage 2, we usually only care about the main output
        out = self.backbone(x)
        
        if isinstance(out, tuple):
            return out[0] # Return only logits
        return out

    def nn_kl_divergence(self):
        """Get KL divergence from the final Bayesian layer."""
        out_layer = self.backbone.swin_unetr.out
        if isinstance(out_layer, BayesianConv3d):
            return out_layer.kl_divergence()
        elif hasattr(out_layer, 'conv') and isinstance(out_layer.conv, BayesianConv3d):
            return out_layer.conv.kl_divergence()
        return torch.tensor(0.0)
