#!/bin/bash
# Script to run all baseline experiments sequentially.
# Usage: bash scripts/run_baselines.sh

set -e

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1
PORT=29500

echo "=========================================================="
echo "   Running BraTS 2021 Baselines                           "
echo "=========================================================="

# 1. Swin UNETR (The direct competitor)
echo ""
echo ">>> [Baseline 1/3] Training Swin UNETR..."
python -m torch.distributed.run --nproc_per_node=2 --master_port=$PORT \
    tools/train_stage1.py \
    --config configs/baselines/swin_unetr.yaml

# 2. SegResNet (CNN Baseline)
echo ""
echo ">>> [Baseline 2/3] Training SegResNet..."
python -m torch.distributed.run --nproc_per_node=2 --master_port=$PORT \
    tools/train_stage1.py \
    --config configs/baselines/segresnet.yaml

# 3. MedNeXt (SOTA CNN)
echo ""
echo ">>> [Baseline 3/3] Training MedNeXt..."
python -m torch.distributed.run --nproc_per_node=2 --master_port=$PORT \
    tools/train_stage1.py \
    --config configs/baselines/mednext.yaml

echo "=========================================================="
echo "   All Baselines Completed!                               "
echo "   Note: nnU-Net requires a separate manual pipeline.     "
echo "=========================================================="
