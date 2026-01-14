"""
Checkpoint Utility Module.
Handles saving and loading of model weights, optimizer states, and schedulers.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[object],
                    epoch: int,
                    metrics: Dict[str, float],
                    save_dir: str,
                    filename: Optional[str] = None,
                    is_best: bool = False) -> str:
    """
    Save training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch: Current epoch.
        metrics: Dictionary of current metrics (e.g., {'dice': 0.9}).
        save_dir: Directory to save the file.
        filename: Custom filename.
        is_best: If True, saves as 'best_model.pth'.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
    
    # Handle DDP (DistributedDataParallel) model wrapper
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    return save_path

def load_checkpoint(checkpoint_path: str,
                    model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[object] = None,
                    device: Optional[torch.device] = None,
                    strict: bool = False) -> Dict[str, Any]:
    """
    Load a checkpoint into the model.
    Handles key matching between DDP and non-DDP models automatically.
    """
    if device is None:
        device = torch.device('cpu')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # Assume it's just the state dict

    # Clean state_dict keys (remove 'module.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # Load model weights
    try:
        model.load_state_dict(new_state_dict, strict=strict)
    except Exception as e:
        print(f"[Warning] Strict loading failed: {e}. Retrying with strict=False.")
        model.load_state_dict(new_state_dict, strict=False)
    
    # Load optimizer and scheduler if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'best_dice': checkpoint.get('best_val_dice', 0.0)
    }
