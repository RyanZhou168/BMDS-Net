"""
Training Script for Stage 1 (Deterministic).
Usage:
    python -m torch.distributed.run --nproc_per_node=4 tools/train_stage1.py --config configs/bmds_net/stage1_deterministic.yaml
"""

import os
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from bmds_net.models import create_model
from bmds_net.data.dataset import BraTSDataset
from bmds_net.data.transforms import BraTSTransform
from bmds_net.engine.trainer_det import DeterministicTrainer
from bmds_net.engine.losses import build_loss
from bmds_net.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    # 1. DDP Setup
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        world_size = int(os.environ['WORLD_SIZE'])
        is_master = (int(os.environ['RANK']) == 0)
    else:
        local_rank = 0
        world_size = 1
        is_master = True

    # 2. Config & Logger
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    logger = None
    if is_master:
        os.makedirs(config['output']['dir'], exist_ok=True)
        logger = setup_logger('train_stage1', os.path.join(config['output']['dir'], 'logs'))
        logger.info(f"Starting Stage 1 Training with {world_size} GPUs")

    # 3. Data
    roi_size = tuple(config['data']['target_size'])
    train_ds = BraTSDataset(config['data']['root_dir'], 'train', 
                           transform=BraTSTransform(train=True, target_size=roi_size))
    val_ds = BraTSDataset(config['data']['root_dir'], 'validation', 
                         transform=BraTSTransform(train=False, target_size=roi_size))
    
    train_sampler = DistributedSampler(train_ds) if world_size > 1 else None
    train_loader = DataLoader(train_ds, batch_size=config['data']['batch_size'],
                            sampler=train_sampler, shuffle=(train_sampler is None), 
                            num_workers=config['data'].get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config['data'].get('num_workers', 2))

    # 4. Model
    model = create_model(config['model']['name'], **config['model'])
    model = model.cuda(local_rank)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 5. Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 1e-4),
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = build_loss(config.get('loss', {})).cuda(local_rank)

    # 6. Trainer
    trainer = DeterministicTrainer(
        model, train_loader, val_loader, optimizer, scheduler, criterion, 
        local_rank, config, logger
    )
    
    if is_master:
        trainer.run()
    else:
        # Workers just participate in DDP sync inside trainer
        trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
