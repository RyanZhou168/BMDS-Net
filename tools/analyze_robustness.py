"""
Robustness Analysis Tool.

Evaluates model performance under missing modality scenarios:
- Missing T1
- Missing T1ce
- Missing T2
- Missing FLAIR

Generates metrics for robustness comparison (Table 3 in paper).
"""

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from bmds_net.models import create_model
from bmds_net.data.dataset import BraTSDataset
from bmds_net.data.transforms import BraTSTransform
from bmds_net.engine.inferer import BMDSInferer
from bmds_net.utils.metrics import calculate_dice

def get_corrupted_image(image, missing_modality_idx):
    """
    Simulate missing modality by zeroing out the corresponding channel.
    Args:
        image: [4, H, W, D]
        missing_modality_idx: 0=FLAIR, 1=T1, 2=T1ce, 3=T2 (BraTS standard order)
    """
    corrupted = image.clone()
    if missing_modality_idx is not None:
        corrupted[missing_modality_idx] = 0.0
    return corrupted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='robustness_results.csv')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = create_model(config['model']['name'], **config['model'])
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Handle checkpoints
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=False)
    
    model.to(device)
    model.eval()
    
    # Load Data (Validation Set)
    roi_size = tuple(config['data']['target_size'])
    ds = BraTSDataset(config['data']['root_dir'], 'validation', 
                     transform=BraTSTransform(train=False, target_size=None))
    
    inferer = BMDSInferer(roi_size=roi_size)
    
    # Scenarios
    scenarios = {
        'Full': None,
        'Missing_FLAIR': 0,
        'Missing_T1': 1,
        'Missing_T1ce': 2,
        'Missing_T2': 3
    }
    
    results = []
    
    print(f"Starting Robustness Analysis on {len(ds)} cases...")
    
    for case_idx in tqdm(range(len(ds))):
        case_id = ds.cases[case_idx]
        sample = ds[case_idx]
        image, label = sample['image'], sample['label'] # image: [4, H, W, D]
        
        # Prepare Label
        l_t = label.unsqueeze(0).unsqueeze(0) # [1, 1, H, W, D]
        
        case_result = {'Case': case_id}
        
        for name, mod_idx in scenarios.items():
            # Corrupt Data
            img_input = get_corrupted_image(image, mod_idx).unsqueeze(0)
            
            # Inference
            pred_mask = inferer.deterministic_infer(model, img_input, device)
            p_t = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)
            
            # Metrics for WT/TC/ET regions
            metrics = calculate_dice(p_t, l_t)
            case_result[f'{name}_WT'] = metrics['dice_wt']
            case_result[f'{name}_TC'] = metrics['dice_tc']
            case_result[f'{name}_ET'] = metrics['dice_et']
            
        results.append(case_result)
        
    # Summary
    df = pd.DataFrame(results)
    print("\nRobustness Summary (Mean Dice):")
    summary = df.mean(numeric_only=True)
    print(summary)
    
    df.to_csv(args.output_csv, index=False)
    print(f"\nDetailed results saved to {args.output_csv}")

if __name__ == '__main__':
    main()
