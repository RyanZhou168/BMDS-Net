"""
Inference Engine Module.
Handles Sliding Window Inference for large 3D volumes.
Supports both Deterministic and Bayesian (MC Dropout) modes.
"""

import torch
import numpy as np
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm
from typing import Tuple, Optional

class BMDSInferer:
    """
    Unified Inference Engine for BMDS-Net.
    """
    def __init__(self, 
                 roi_size: Tuple[int, int, int] = (128, 128, 128),
                 sw_batch_size: int = 4,
                 overlap: float = 0.5):
        self.inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode='gaussian', # Gaussian weighting for smoother stitching
            sigma_scale=0.125
        )

    def deterministic_infer(self, model, image, device) -> np.ndarray:
        """
        Standard single-pass inference.
        Returns: Segmentation Map [H, W, D]
        """
        model.eval()
        with torch.no_grad():
            # [B, C, H, W, D]
            logits = self.inferer(image.to(device), model)
            
            # If model returns tuple (e.g. during training), take the first element
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
                
            pred_mask = torch.argmax(logits, dim=1) # [B, H, W, D]
            
        return pred_mask[0].cpu().numpy()

    def bayesian_infer(self, model, image, device, mc_samples=20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bayesian inference via Monte Carlo Sampling.
        Returns: 
            - Segmentation Map [H, W, D]
            - Uncertainty Map [H, W, D]
        """
        model.eval()
        # Enable Dropout for MC Sampling
        if hasattr(model, 'enable_uncertainty'):
            model.enable_uncertainty() # Custom method in BayesianBMDSNet
        else:
            # Fallback: manually enable dropout layers
            for m in model.modules():
                if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                    m.train()

        predictions = []
        
        with torch.no_grad():
            for _ in tqdm(range(mc_samples), desc="MC Sampling", leave=False):
                logits = self.inferer(image.to(device), model)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu()) # Keep on CPU to save GPU memory

        # Stack: [Samples, B, C, H, W, D]
        predictions = torch.stack(predictions)
        
        # Mean Prediction
        mean_probs = torch.mean(predictions, dim=0) # [B, C, H, W, D]
        seg_result = torch.argmax(mean_probs, dim=1)[0].numpy()
        
        # Uncertainty (Predictive Variance)
        # Average variance across all channels
        uncertainty = torch.var(predictions, dim=0).mean(dim=1)[0].numpy()
        
        return seg_result, uncertainty
