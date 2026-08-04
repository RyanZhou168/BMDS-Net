"""
Unified Testing Script.
Performs Inference -> Metric Calculation -> Saving Results.
Supports both Deterministic (Stage 1 / Baselines) and Bayesian (Stage 2) modes.
"""

import os
import argparse
import yaml
import torch
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm

from bmds_net.models import create_model
from bmds_net.data.dataset import BraTSDataset
from bmds_net.data.transforms import BraTSTransform
from bmds_net.engine.inferer import BMDSInferer
from bmds_net.utils.metrics import calculate_dice, calculate_hd95

def save_nifti(data, affine, path):
    """Helper to save numpy array as NIfTI."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--mode', type=str, default='deterministic', 
                        choices=['deterministic', 'bayesian'], 
                        help='Inference mode')
    parser.add_argument('--save_preds', action='store_true', help='Save segmentation masks')
    args = parser.parse_args()

    # Load Config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(config['output']['dir'], 'inference_results')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Build Model
    # Automatically switch to Bayesian wrapper if mode is bayesian
    model_name = 'bayesian_bmds_net' if args.mode == 'bayesian' else config['model']['name']
    model = create_model(model_name, **config['model'])
    
    # Load Weights
    print(f"Loading weights from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Clean keys (remove DDP prefix)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load (strict=False to allow loading Stage 1 weights into Stage 2 wrapper for testing)
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. Setup Data
    # Use 'validation' split for testing. No cropping (target_size=None) for full volume inference.
    roi_size = tuple(config['data']['target_size'])
    ds = BraTSDataset(config['data']['root_dir'], 'validation', 
                     transform=BraTSTransform(train=False, target_size=None))
    
    inferer = BMDSInferer(roi_size=roi_size)
    
    results = []
    print(f"Starting Inference on {len(ds)} cases in [{args.mode}] mode...")
    
    for i in tqdm(range(len(ds))):
        case_id = ds.cases[i]
        sample = ds[i]
        image, label = sample['image'], sample['label'] # image: [4, H, W, D], label: [H, W, D]
        
        # Add batch dimension for inference
        image_input = image.unsqueeze(0)
        
        # --- Inference ---
        if args.mode == 'deterministic':
            pred_mask = inferer.deterministic_infer(model, image_input, device)
            uncertainty_map = None
        else:
            pred_mask, uncertainty_map = inferer.bayesian_infer(model, image_input, device)
            
        # --- Metrics ---
        # Convert to Tensor for metric calculation (on CPU)
        # Pred: [1, H, W, D], Label: [1, H, W, D]
        p_t = torch.from_numpy(pred_mask).unsqueeze(0)
        l_t = label.unsqueeze(0)
        
        # Calculate Dice & HD95
        metrics = calculate_dice(p_t, l_t)
        hd95_metrics = calculate_hd95(p_t, l_t)
        metrics.update(hd95_metrics)
        metrics['case_id'] = case_id
        results.append(metrics)
        
        # --- Saving (Optional) ---
        if args.save_preds:
            case_save_dir = os.path.join(output_dir, case_id)
            os.makedirs(case_save_dir, exist_ok=True)
            
            # Load original affine for saving
            original_nii_path = os.path.join(ds.data_dir, case_id, f"{case_id}_seg.nii.gz")
            if os.path.exists(original_nii_path):
                affine = nib.load(original_nii_path).affine
            else:
                affine = np.eye(4) # Fallback
            
            # Save Prediction
            # Remap label 3 back to 4 for BraTS standard if needed, but usually we keep 0-3 for internal consistency
            save_nifti(pred_mask, affine, os.path.join(case_save_dir, f"{case_id}_pred.nii.gz"))
            
            # Save Uncertainty (if available)
            if uncertainty_map is not None:
                save_nifti(uncertainty_map, affine, os.path.join(case_save_dir, f"{case_id}_uncertainty.nii.gz"))

    # 3. Summary
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*40)
    print(f"Results saved to: {csv_path}")
    print("Average Metrics:")
    print(f"  WT Dice: {df['dice_wt'].mean():.4f}")
    print(f"  TC Dice: {df['dice_tc'].mean():.4f}")
    print(f"  ET Dice: {df['dice_et'].mean():.4f}")
    print(f"  WT HD95: {df['hd95_wt'].mean():.4f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()
