"""
Visualization Utilities Module.

This module is reserved for visualization functions (e.g., uncertainty maps, 
segmentation overlays) used in qualitative analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def save_overlay(image, mask, save_path, alpha=0.5):
    """
    Placeholder for saving overlay images.
    
    Args:
        image: Source image (numpy array)
        mask: Segmentation mask (numpy array)
        save_path: Output path
        alpha: Transparency factor
    """
    # Visualization logic would go here
    pass


def visualize_uncertainty_map(
    image: np.ndarray, 
    segmentation: np.ndarray, 
    uncertainty: np.ndarray, 
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Visualize uncertainty map overlaid on the original image.
    
    Args:
        image: Input image array of shape (H, W, D) or (H, W)
        segmentation: Segmentation mask of shape (H, W, D) or (H, W)
        uncertainty: Uncertainty map of same shape as segmentation
        slice_idx: Slice index to visualize (for 3D data). If None, uses middle slice
        save_path: If provided, saves the visualization to this path
    """
    # Determine if the data is 3D or 2D
    is_3d = len(image.shape) == 3
    
    if is_3d:
        if slice_idx is None:
            slice_idx = image.shape[2] // 2  # Use middle slice
        
        image_slice = image[:, :, slice_idx]
        seg_slice = segmentation[:, :, slice_idx]
        uncert_slice = uncertainty[:, :, slice_idx]
    else:
        image_slice = image
        seg_slice = segmentation
        uncert_slice = uncertainty
    
    # Normalize image to [0, 1] range
    image_norm = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_norm, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation overlay
    axes[1].imshow(image_norm, cmap='gray')
    axes[1].contour(seg_slice, colors='red', linewidths=1)
    axes[1].set_title('Segmentation Overlay')
    axes[1].axis('off')
    
    # Uncertainty map
    im = axes[2].imshow(uncert_slice, cmap='jet', vmin=0, vmax=uncert_slice.max())
    axes[2].set_title('Uncertainty Map')
    axes[2].axis('off')
    
    # Add colorbar for uncertainty
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_segmentation_with_contours(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Plot segmentation results with contour overlays.
    
    Args:
        image: Input image array of shape (H, W, D) or (H, W)
        prediction: Predicted segmentation of shape (H, W, D) or (H, W)
        ground_truth: Ground truth segmentation (optional)
        slice_idx: Slice index to visualize (for 3D data). If None, uses middle slice
        save_path: If provided, saves the visualization to this path
    """
    # Determine if the data is 3D or 2D
    is_3d = len(image.shape) == 3
    
    if is_3d:
        if slice_idx is None:
            slice_idx = image.shape[2] // 2  # Use middle slice
        
        image_slice = image[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx]
    else:
        image_slice = image
        pred_slice = prediction
    
    # Normalize image to [0, 1] range
    image_norm = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
    
    if ground_truth is not None:
        gt_slice = ground_truth[:, :, slice_idx] if is_3d else ground_truth
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_norm, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction contours on image
        axes[1].imshow(image_norm, cmap='gray')
        axes[1].contour(pred_slice, colors='yellow', linewidths=1.5, levels=[0.5, 1.5, 2.5])
        if ground_truth is not None:
            axes[1].contour(gt_slice, colors='red', linewidths=1, levels=[0.5, 1.5, 2.5])
        axes[1].set_title('Prediction vs Ground Truth')
        axes[1].axis('off')
        
        # Difference map
        diff = np.abs(pred_slice - gt_slice)
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Difference Map')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        axes[0].imshow(image_norm, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction contours on image
        axes[1].imshow(image_norm, cmap='gray')
        axes[1].contour(pred_slice, colors='yellow', linewidths=1.5, levels=[0.5, 1.5, 2.5])
        axes[1].set_title('Prediction Contours')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
