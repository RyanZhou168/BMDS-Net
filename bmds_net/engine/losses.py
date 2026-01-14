"""
Loss Functions Module.
Implements the Robust Boundary-Focal-Dice Loss used in BMDS-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

class RobustBoundaryFocalDiceLoss(nn.Module):
    """
    Robust Composite Loss Function: Dice + CrossEntropy + Boundary.
    
    Strategy:
    1. Core: DiceCELoss (Stability).
    2. Refinement: Boundary Loss (Precision), enabled via warmup.
    
    Args:
        alpha (float): Weight for DiceCE Loss.
        beta (float): Weight for Boundary Loss.
        gamma (float): Weight for Focal/Auxiliary components (if any).
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5):
        super().__init__()
        # Use MONAI's mature implementation as the backbone
        self.dice_ce = DiceCELoss(
            to_onehot_y=True, 
            softmax=True, 
            batch=True,
            smooth_nr=1e-5, 
            smooth_dr=1e-5
        )
        self.alpha = alpha
        self.beta = beta
        
        # Pre-calculated kernel for boundary extraction (Approximate Erosion)
        # 3x3x3 kernel with sum=1
        self.kernel = torch.ones((1, 1, 3, 3, 3)).float() / 27.0

    def _get_boundary(self, mask):
        """
        Extract boundary using average pooling approximation.
        Boundary = Region where local average is between 0 and 1.
        Faster than Sobel filter for 3D volumes.
        """
        kernel = self.kernel.to(mask.device)
        # Add channel dim for conv3d if needed
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)
            
        avg_mask = F.conv3d(mask.float(), kernel, padding=1)
        # Pixels with mix of 0s and 1s in neighborhood are boundaries
        boundary = ((avg_mask > 0.01) & (avg_mask < 0.99)).float()
        return boundary

    def forward(self, logits, labels, current_epoch=0):
        """
        Args:
            logits: [B, C, H, W, D]
            labels: [B, H, W, D] or [B, 1, H, W, D]
            current_epoch: Used for Boundary Loss Warmup.
        """
        # Ensure labels have channel dim for DiceCE
        if labels.dim() == 4:
            labels = labels.unsqueeze(1)

        # 1. Main Dice + CE Loss
        main_loss = self.dice_ce(logits, labels)
        
        # 2. Boundary Loss Warmup Strategy
        # Avoid unstable gradients from boundary loss in early epochs
        if current_epoch < 20:
            boundary_weight = 0.0
        elif current_epoch < 50:
            boundary_weight = self.beta * ((current_epoch - 20) / 30.0)
        else:
            boundary_weight = self.beta

        if boundary_weight <= 0:
            return self.alpha * main_loss

        # 3. Boundary Loss Calculation (Only for foreground classes)
        probs = torch.softmax(logits, dim=1)
        total_boundary_loss = 0.0
        
        # Iterate over foreground classes (1, 2, 3)
        # Assuming channel 0 is background
        num_classes = logits.shape[1]
        
        for c in range(1, num_classes):
            gt_c = (labels == c).float()
            pred_c = probs[:, c:c+1]
            
            gt_boundary = self._get_boundary(gt_c)
            
            # Force FP32 for precision in boundary calculation
            with torch.cuda.amp.autocast(enabled=False):
                b_loss = F.binary_cross_entropy(
                    pred_c.float(),
                    gt_c.float(),
                    weight=(gt_boundary + 0.1).float() # Weight boundaries higher
                )
            
            total_boundary_loss += b_loss

        return self.alpha * main_loss + boundary_weight * (total_boundary_loss / (num_classes - 1))
