"""
Training Script for Stage 2 (Bayesian Fine-tuning).
Loads a pre-trained deterministic model and fine-tunes the Bayesian layer.
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from bmds_net.models import create_model
from bmds_net.utils.checkpoint import load_checkpoint
from bmds_net.data.dataset import BraTSDataset
from bmds_net.data.transforms import BraTSTransform
from bmds_net.engine.trainer_bayes import BayesianTrainer
from bmds_net.engine.losses import RobustBoundaryFocalDiceLoss
from bmds_net.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup (Single GPU for fine-tuning is usually sufficient and simpler)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['output']['dir'], exist_ok=True)
    logger = setup_logger('train_stage2', os.path.join(config['output']['dir'], 'logs'))
    
    logger.info("Starting Stage 2: Bayesian Fine-tuning")

    # 1. Load Pretrained Deterministic Model
    logger.info(f"Loading pretrained weights from: {config['pretrained']['path']}")
    
    # We first create the Bayesian wrapper structure
    # The 'create_model' factory handles the wrapping logic
    model = create_model('bayesian_bmds_net', **config['model'])
    
    # Load weights: We need to be careful here.
    # The checkpoint contains keys for 'BMDSNet'.
    # Our model is 'BayesianBMDSNet' which contains 'BMDSNet' as 'self.backbone'.
    # The load_checkpoint utility should handle this, or we load into backbone manually.
    
    ckpt = torch.load(config['pretrained']['path'], map_location='cpu')
    state_dict = ckpt['model_state_dict']
    
    # Prefix adjustment: add 'backbone.' to keys since BayesianBMDSNet wraps it
    new_state_dict = {}
    for k, v in state_dict.items():
        # Handle DDP prefix removal first
        k = k.replace('module.', '')
        # Add backbone prefix
        new_state_dict[f'backbone.{k}'] = v
        
    # Load with strict=False because the final layer (Conv3d) in state_dict 
    # won't match the BayesianConv3d in the new model.
    # The BayesianConv3d weights are initialized from the deterministic ones 
    # inside __init__, so we are safe ignoring the final layer mismatch here.
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    
    # 2. Freeze Backbone (Optional, but recommended for efficiency)
    # Only train the Bayesian layer
    for name, param in model.named_parameters():
        if 'backbone.swin_unetr.out' not in name:
            param.requires_grad = False
    
    logger.info("Backbone frozen. Training only Bayesian Output Layer.")

    # 3. Data
    roi_size = tuple(config['data']['target_size'])
    train_ds = BraTSDataset(config['data']['root_dir'], 'train', 
                           transform=BraTSTransform(train=True, target_size=roi_size))
    val_ds = BraTSDataset(config['data']['root_dir'], 'validation', 
                         transform=BraTSTransform(train=False, target_size=roi_size))
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # 4. Optimizer
    # Use lower LR for fine-tuning
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=config['training']['lr'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    criterion = RobustBoundaryFocalDiceLoss().to(device)

    # 5. Trainer
    trainer = BayesianTrainer(
        model, train_loader, val_loader, optimizer, scheduler, criterion, 
        device, config, logger
    )
    trainer.run()

if __name__ == '__main__':
    main()
