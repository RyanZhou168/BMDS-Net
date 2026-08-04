"""
Stage 1: Deterministic Trainer.
Handles the training loop for the robust backbone (MMCF + DDS).
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from ..models.modules.dds import BidirectionalDistillationLoss
from ..utils.metrics import calculate_dice

class DeterministicTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 optimizer, 
                 scheduler, 
                 criterion, 
                 device, 
                 config, 
                 logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger
        
        self.scaler = GradScaler(enabled=config['training']['use_amp'])
        self.distill_criterion = BidirectionalDistillationLoss().to(device)
        
        self.start_epoch = 0
        self.best_dice = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch in pbar:
            images, labels = batch['image'].to(self.device), batch['label'].to(self.device)
            
            # Ensure labels have channel dim for Loss
            if labels.dim() == 4:
                labels = labels.unsqueeze(1)

            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config['training']['use_amp']):
                # Forward Pass
                outputs = self.model(images)
                
                # Check for multiple outputs (BMDS-Net returns tuple)
                if isinstance(outputs, (list, tuple)):
                    logits_final, logits_64, logits_32, mmcf_weights = outputs
                    
                    # 1. Main Loss
                    loss = self.criterion(logits_final, labels, current_epoch=epoch)
                    
                    # 2. Deep Supervision Loss
                    weights = self.config['loss'].get('dds_weights', [1.0, 0.4, 0.2])
                    loss += weights[1] * self.criterion(logits_64, labels, current_epoch=epoch)
                    loss += weights[2] * self.criterion(logits_32, labels, current_epoch=epoch)
                    
                    # 3. Distillation Loss
                    # Access raw model if wrapped in DDP
                    raw_model = self.model.module if hasattr(self.model, 'module') else self.model
                    distill_weight = self.config['loss'].get('distill_weight', 0.05)
                    if distill_weight > 0 and hasattr(raw_model, 'get_dds_features'):
                        dds_feats = raw_model.get_dds_features()
                        d_loss = self.distill_criterion(mmcf_weights, dds_feats)
                        loss += distill_weight * d_loss
                    
                else:
                    # Baseline models (Single output)
                    loss = self.criterion(outputs, labels, current_epoch=epoch)

            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        val_dice = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                
                with autocast(enabled=self.config['training']['use_amp']):
                    outputs = self.model(images)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    
                    # Calculate Dice (WT)
                    metrics = calculate_dice(outputs, labels)
                    val_dice += metrics['dice_wt'] # Use Whole Tumor as primary metric
        
        avg_dice = val_dice / len(self.val_loader)
        return avg_dice

    def run(self):
        epochs = self.config['training']['epochs']
        save_dir = os.path.join(self.config['output']['dir'], 'models')
        
        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_epoch(epoch)
            val_dice = self.validate(epoch)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val WT Dice: {val_dice:.4f} - LR: {current_lr:.6f}")
            
            # Save Best
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                from ..utils.checkpoint import save_checkpoint
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch,
                    {'dice': val_dice}, save_dir, is_best=True
                )
                self.logger.info("  [*] Saved New Best Model")
