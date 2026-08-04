"""
Loss functions for BMDS-Net.

The manuscript reports a BoundaryFocalDice objective for three overlapping
BraTS region channels: whole tumor (WT), tumor core (TC), and enhancing tumor
(ET). The implementation below uses independent sigmoid channels rather than
softmax one-hot labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


def brats_label_to_regions(labels: torch.Tensor) -> torch.Tensor:
    """Convert BraTS integer labels to multi-hot WT/TC/ET region masks.

    Accepted input shapes are [B, H, W, D], [B, 1, H, W, D], or already
    multi-hot [B, 3, H, W, D]. Label value 4 is treated as ET for compatibility
    with raw BraTS annotations; preprocessed labels that map ET to 3 are also
    supported.
    """
    if labels.dim() == 5 and labels.shape[1] == 3:
        return labels.float()
    if labels.dim() == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.dim() != 4:
        raise ValueError(f"Expected labels with shape [B,H,W,D], [B,1,H,W,D], or [B,3,H,W,D], got {tuple(labels.shape)}")

    labels = labels.long()
    wt = labels > 0
    tc = (labels == 1) | (labels == 3) | (labels == 4)
    et = (labels == 3) | (labels == 4)
    return torch.stack([wt, tc, et], dim=1).float()


class BoundaryFocalDiceLoss(nn.Module):
    """Composite Dice + binary focal + boundary BCE loss for WT/TC/ET masks."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        boundary_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
        boundary_threshold: float = 0.5,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.boundary_threshold = boundary_threshold
        self.eps = eps

        sobel_1d = torch.tensor([-1.0, 0.0, 1.0])
        smooth_1d = torch.tensor([1.0, 2.0, 1.0])
        kx = sobel_1d[:, None, None] * smooth_1d[None, :, None] * smooth_1d[None, None, :]
        ky = smooth_1d[:, None, None] * sobel_1d[None, :, None] * smooth_1d[None, None, :]
        kz = smooth_1d[:, None, None] * smooth_1d[None, :, None] * sobel_1d[None, None, :]
        kernel = torch.stack([kx, ky, kz]).unsqueeze(1)
        self.register_buffer("sobel_kernel", kernel)

    def _boundary_mask(self, target_regions: torch.Tensor) -> torch.Tensor:
        batch, channels = target_regions.shape[:2]
        masks = target_regions.reshape(batch * channels, 1, *target_regions.shape[2:])
        kernel = self.sobel_kernel.to(dtype=masks.dtype, device=masks.device)
        grad = F.conv3d(masks, kernel, padding=1)
        mag = torch.linalg.vector_norm(grad, ord=2, dim=1, keepdim=True)
        boundary = (mag > self.boundary_threshold).float()
        return boundary.reshape(batch, channels, *target_regions.shape[2:])

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, current_epoch: int = 0) -> torch.Tensor:
        del current_epoch
        targets = brats_label_to_regions(labels).to(device=logits.device, dtype=logits.dtype)
        if logits.shape[1] != targets.shape[1]:
            raise ValueError(f"BoundaryFocalDice expects {targets.shape[1]} output channels, got {logits.shape[1]}")

        probs = torch.sigmoid(logits)
        reduce_dims = tuple(range(2, logits.dim()))

        intersection = (probs * targets).sum(dim=reduce_dims)
        denom = probs.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
        dice = 1.0 - ((2.0 * intersection + self.eps) / (denom + self.eps)).mean()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (((1.0 - p_t).clamp_min(0.0) ** self.focal_gamma) * bce).mean()

        boundary = self._boundary_mask(targets)
        boundary_denom = boundary.sum(dim=reduce_dims).clamp_min(self.eps)
        boundary_loss = ((boundary * bce).sum(dim=reduce_dims) / boundary_denom).sum(dim=1).mean()

        return (
            self.dice_weight * dice
            + self.boundary_weight * boundary_loss
            + self.focal_weight * focal
        )


# Backward-compatible name used by older scripts in this repository.
RobustBoundaryFocalDiceLoss = BoundaryFocalDiceLoss


class DiceCELossWrapper(nn.Module):
    """MONAI DiceCELoss with the same call signature as BoundaryFocalDiceLoss."""

    def __init__(self):
        super().__init__()
        self.loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            batch=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, current_epoch: int = 0) -> torch.Tensor:
        del current_epoch
        return self.loss(logits, labels)


def build_loss(config: dict) -> nn.Module:
    """Build a loss function from a YAML loss section."""
    loss_type = config.get("type", "boundary_focal_dice")
    if loss_type in {"boundary_focal_dice", "robust_boundary_dice"}:
        return BoundaryFocalDiceLoss(
            dice_weight=config.get("dice_weight", config.get("alpha", 1.0)),
            boundary_weight=config.get("boundary_weight", config.get("beta", 1.0)),
            focal_weight=config.get("focal_weight", config.get("gamma", 1.0)),
            focal_gamma=config.get("focal_gamma", 2.0),
        )
    if loss_type == "dice_ce":
        return DiceCELossWrapper()
    raise ValueError(f"Unknown loss type: {loss_type}")
