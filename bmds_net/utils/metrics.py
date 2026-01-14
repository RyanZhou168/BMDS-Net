"""
Unified Metrics Module.
Includes:
1. Deterministic Metrics: Dice Score, Hausdorff Distance (HD95).
2. Bayesian Metrics: Expected Calibration Error (ECE), Negative Log Likelihood (NLL).
"""

import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance
from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# Deterministic Metrics (Segmentation Quality)
# -----------------------------------------------------------------------------

def calculate_dice(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate Dice scores for BraTS regions (WT, TC, ET).
    Args:
        preds: Shape [B, C, H, W, D] (One-hot or Logits)
        targets: Shape [B, H, W, D] (Integer labels 0-3)
    """
    # Ensure preds are class indices
    if preds.dim() == 5:
        preds = torch.argmax(preds, dim=1)
    
    # BraTS Regions
    # WT (Whole Tumor): Label 1 + 2 + 3 (Note: Label 4 is usually mapped to 3)
    pred_wt = (preds > 0).float()
    target_wt = (targets > 0).float()
    
    # TC (Tumor Core): Label 1 + 3
    pred_tc = ((preds == 1) | (preds == 3)).float()
    target_tc = ((targets == 1) | (targets == 3)).float()
    
    # ET (Enhancing Tumor): Label 3
    pred_et = (preds == 3).float()
    target_et = (targets == 3).float()
    
    def _dice(p, t):
        intersection = torch.sum(p * t)
        union = torch.sum(p) + torch.sum(t)
        if union == 0: return 1.0
        return (2.0 * intersection / union).item()

    return {
        'dice_wt': _dice(pred_wt, target_wt),
        'dice_tc': _dice(pred_tc, target_tc),
        'dice_et': _dice(pred_et, target_et),
    }

def calculate_hd95(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate Hausdorff Distance 95% using MONAI logic.
    Robust against empty masks (returns NaN or Inf).
    """
    if preds.dim() == 5:
        preds = torch.argmax(preds, dim=1)
        
    # Prepare regions (B, 1, H, W, D) for MONAI metric
    def _prepare_mask(mask):
        return mask.unsqueeze(1).float()

    pred_wt = _prepare_mask(preds > 0)
    target_wt = _prepare_mask(targets > 0)
    
    pred_tc = _prepare_mask((preds == 1) | (preds == 3))
    target_tc = _prepare_mask((targets == 1) | (targets == 3))
    
    pred_et = _prepare_mask(preds == 3)
    target_et = _prepare_mask(targets == 3)

    def _hd95(p, t):
        if p.sum() == 0 or t.sum() == 0:
            return float('nan') # Handle empty cases
        return compute_hausdorff_distance(
            y_pred=p, y=t, include_background=False, percentile=95
        ).item()

    return {
        'hd95_wt': _hd95(pred_wt, target_wt),
        'hd95_tc': _hd95(pred_tc, target_tc),
        'hd95_et': _hd95(pred_et, target_et),
    }

# -----------------------------------------------------------------------------
# Bayesian Metrics (Uncertainty & Calibration)
# -----------------------------------------------------------------------------

def calculate_ece(preds_mean: torch.Tensor, targets: torch.Tensor, num_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    Args:
        preds_mean: Softmax probabilities [B, C, H, W, D]
        targets: Ground truth labels [B, H, W, D]
    """
    # Flatten
    preds_flat = preds_mean.permute(0, 2, 3, 4, 1).reshape(-1, preds_mean.shape[1])
    targets_flat = targets.flatten()
    
    confidences, predictions = torch.max(preds_flat, 1)
    correct = (predictions == targets_flat).float()
    
    ece = torch.zeros(1, device=preds_mean.device)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=preds_mean.device)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

def calculate_nll(preds_mean: torch.Tensor, preds_var: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Negative Log Likelihood (NLL) assuming Gaussian posterior.
    """
    # One-hot targets
    num_classes = preds_mean.shape[1]
    targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float() # [B, C, H, W, D]
    
    # NLL = 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
    # Adding epsilon for stability
    var = preds_var + 1e-8
    nll = 0.5 * torch.log(2 * np.pi * var) + (targets_one_hot - preds_mean)**2 / (2 * var)
    
    return nll.mean().item()
