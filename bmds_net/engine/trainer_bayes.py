"""
Stage 2: Bayesian Trainer.
Handles fine-tuning of the Bayesian layer using Variational Inference (VI).
"""

import os
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from ..utils.metrics import calculate_dice

class BayesianTrainer:
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
        self.best_dice = 0.0
        
        # KL Annealing parameters
        self.kl_weight = config['training'].get('kl_weight', 0.005)

    def _get_kl(self):
        """Helper to get KL divergence from model (handles DDP)"""
        if hasattr(self.model, 'module'):
            return self.model.module.nn_kl_divergence()
        return self.model.nn_kl_divergence()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_kl = 0.0
        
        # KL Annealing Strategy: Gradually increase KL weight
        # Start from epoch 0, ramp up over 10 epochs
        kl_anneal_factor = min(1.0, epoch / 10.0)
        current_kl_weight = self.kl_weight * kl_anneal_factor
        
        pbar = tqdm(self.train_loader, desc=f"Bayes Epoch {epoch+1}", leave=False)
        
        for batch in pbar:
            images, labels = batch['image'].to(self.device), batch['label'].to(self.device)
            
            if labels.dim() == 4:
                labels = labels.unsqueeze(1)

            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config['training']['use_amp']):
                # Forward Pass (One MC Sample during training is standard for efficiency)
                outputs = self.model(images)
                
                # Task Loss (Reconstruction)
                task_loss = self.criterion(outputs, labels)
                
                # KL Divergence Loss (Regularization)
                kl_loss = self._get_kl() * current_kl_weight
                
                # ELBO = Task Loss + KL
                loss = task_loss + kl_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            total_kl += kl_loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'kl': f'{kl_loss.item():.4f}'})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        """
        Validation using MC Sampling (Ensemble average).
        """
        self.model.eval()
        # Enable uncertainty mode (Dropout active)
        if hasattr(self.model, 'enable_uncertainty'):
            self.model.enable_uncertainty()
            
        val_dice = 0.0
        mc_samples = 5 # Use fewer samples for validation speed
        
        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                
                # MC Integration
                outputs_list = []
                for _ in range(mc_samples):
                    out = self.model(images)
                    outputs_list.append(out)
                
                # Mean Prediction
                mean_output = torch.stack(outputs_list).mean(dim=0)
                
                metrics = calculate_dice(mean_output, labels)
                val_dice += metrics['dice_wt']
        
        return val_dice / len(self.val_loader)

    def run(self):
        epochs = self.config['training']['epochs']
        save_dir = os.path.join(self.config['output']['dir'], 'models')
        
        for epoch in range(epochs):
            loss = self.train_epoch(epoch)
            val_dice = self.validate(epoch)
            
            self.scheduler.step()
            
            self.logger.info(f"Epoch {epoch+1} - ELBO: {loss:.4f} - Val Dice: {val_dice:.4f}")
            
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                from ..utils.checkpoint import save_checkpoint
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch,
                    {'dice': val_dice}, save_dir, is_best=True
                )
