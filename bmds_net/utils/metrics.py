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


def _label_to_regions(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if labels.dim() == 5 and labels.shape[1] == 3:
        wt = labels[:, 0] > 0.5
        tc = labels[:, 1] > 0.5
        et = labels[:, 2] > 0.5
        return wt.float(), tc.float(), et.float()
    if labels.dim() == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    labels = labels.long()
    wt = labels > 0
    tc = (labels == 1) | (labels == 3) | (labels == 4)
    et = (labels == 3) | (labels == 4)
    return wt.float(), tc.float(), et.float()


def _pred_to_regions(preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if preds.dim() == 5 and preds.shape[1] == 3:
        probs = torch.sigmoid(preds) if (preds.min() < 0 or preds.max() > 1) else preds
        return (probs[:, 0] > 0.5).float(), (probs[:, 1] > 0.5).float(), (probs[:, 2] > 0.5).float()
    if preds.dim() == 5:
        preds = torch.argmax(preds, dim=1)
    return _label_to_regions(preds)

# -----------------------------------------------------------------------------
# Deterministic Metrics (Segmentation Quality)
# -----------------------------------------------------------------------------

def calculate_dice(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate Dice scores for BraTS regions (WT, TC, ET).
    Args:
        preds: Label maps, 4-class logits, or 3-channel WT/TC/ET logits.
        targets: Integer labels or 3-channel WT/TC/ET masks.
    """
    pred_wt, pred_tc, pred_et = _pred_to_regions(preds)
    target_wt, target_tc, target_et = _label_to_regions(targets)
    
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
    # Prepare regions (B, 1, H, W, D) for MONAI metric
    def _prepare_mask(mask):
        return mask.unsqueeze(1).float()

    pred_wt, pred_tc, pred_et = _pred_to_regions(preds)
    target_wt, target_tc, target_et = _label_to_regions(targets)

    pred_wt = _prepare_mask(pred_wt)
    target_wt = _prepare_mask(target_wt)
    pred_tc = _prepare_mask(pred_tc)
    target_tc = _prepare_mask(target_tc)
    pred_et = _prepare_mask(pred_et)
    target_et = _prepare_mask(target_et)

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

def calculate_ece(preds_mean: torch.Tensor, targets: torch.Tensor, num_bins: int = 15) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    Args:
        preds_mean: WT/TC/ET sigmoid probabilities [B, 3, H, W, D]
        targets: Ground truth labels or WT/TC/ET masks
    """
    targets_region = torch.stack(_label_to_regions(targets), dim=1).to(preds_mean.device)
    confidences = preds_mean.reshape(-1)
    correct = (((preds_mean > 0.5).float() == targets_region).float()).reshape(-1)
    
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
    Calculate mean three-channel binary cross-entropy NLL.
    """
    del preds_var
    targets_region = torch.stack(_label_to_regions(targets), dim=1).to(preds_mean.device)
    return torch.nn.functional.binary_cross_entropy(preds_mean.clamp(1e-6, 1 - 1e-6), targets_region).item()
