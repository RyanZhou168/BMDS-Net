"""
Model Factory Module.
Provides a unified interface to instantiate BMDS-Net and baselines.
"""

import torch.nn as nn
from .bmds_net import BMDSNet, BayesianBMDSNet
# Note: Baselines will be imported here once migrated
# from .backbones.segresnet import SegResNet
# from .backbones.mednext import MedNeXt

def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name (str): 'bmds_net', 'bayesian_bmds_net', or baseline names.
        **kwargs: Arguments passed to the model constructor.
    """
    
    # Clean kwargs (remove None values or irrelevant keys if necessary)
    valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    if model_name == 'bmds_net':
        return BMDSNet(**valid_kwargs)
        
    elif model_name == 'bayesian_bmds_net':
        # Separate MC dropout rate from structural args
        mc_rate = valid_kwargs.pop('mc_dropout_rate', 0.1)
        
        # Create deterministic backbone
        backbone = BMDSNet(**valid_kwargs)
        
        # Wrap with Bayesian logic
        return BayesianBMDSNet(backbone, mc_dropout_rate=mc_rate)
    
    # Placeholder for baselines (to be implemented in next batch)
    elif model_name == 'segresnet':
        from monai.networks.nets import SegResNet
        return SegResNet(**valid_kwargs)
        
    elif model_name == 'unetr':
        from monai.networks.nets import UNETR
        return UNETR(**valid_kwargs)
        
    elif model_name == 'swin_unetr':
        from .backbones.swin_unetr import SwinUNETRBaseline
        return SwinUNETRBaseline(**valid_kwargs)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
