#!/bin/bash

# BMDS-Net Full Pipeline Script
# Usage: bash scripts/run_all_stages.sh

set -e # Exit immediately if a command exits with a non-zero status.

# Define GPUs
export CUDA_VISIBLE_DEVICES=0,1

echo "=========================================================="
echo "   BMDS-Net: Deployment-aware Segmentation Pipeline       "
echo "=========================================================="

# ------------------------------------------------------------------
# Stage 1: Deterministic Training (MMCF + DDS)
# ------------------------------------------------------------------
echo ""
echo ">>> [Stage 1] Starting Deterministic Training..."
python -m torch.distributed.run --nproc_per_node=2 --master_port=29500 \
    tools/train_stage1.py \
    --config configs/bmds_net/stage1_deterministic.yaml

echo ">>> [Stage 1] Completed. Checkpoints saved to work_dirs/stage1_deterministic"

# ------------------------------------------------------------------
# Stage 2: Bayesian Fine-tuning
# ------------------------------------------------------------------
echo ""
echo ">>> [Stage 2] Starting Bayesian Fine-tuning..."
# Usually runs on single GPU for fine-tuning
CUDA_VISIBLE_DEVICES=0 python tools/train_stage2.py \
    --config configs/bmds_net/stage2_bayesian.yaml

echo ">>> [Stage 2] Completed. Checkpoints saved to work_dirs/stage2_bayesian"

# ------------------------------------------------------------------
# Final Evaluation (Uncertainty Estimation)
# ------------------------------------------------------------------
echo ""
echo ">>> [Evaluation] Running Bayesian Inference & Testing..."
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    --config configs/bmds_net/stage2_bayesian.yaml \
    --checkpoint work_dirs/stage2_bayesian/models/best_model.pth \
    --mode bayesian \
    --save_preds

echo "=========================================================="
echo "   All Stages Completed Successfully!                     "
echo "=========================================================="
