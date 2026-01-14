#!/bin/bash
# Helper script to setup nnU-Net environment variables and directory structure.
# Usage: source scripts/prepare_nnunet.sh

# 1. Define paths (Modify these to your actual storage location)
# Suggestion: Use a fast SSD for nnUNet_preprocessed
export nnUNet_raw="./data/nnUNet_raw"
export nnUNet_preprocessed="./data/nnUNet_preprocessed"
export nnUNet_results="./work_dirs/nnUNet_results"

echo "=================================================="
echo "   nnU-Net Environment Setup                      "
echo "=================================================="
echo "nnUNet_raw:          $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results:      $nnUNet_results"

# 2. Create directories
mkdir -p $nnUNet_raw
mkdir -p $nnUNet_preprocessed
mkdir -p $nnUNet_results

# 3. Instruction for user
echo ""
echo ">>> Environment variables exported."
echo ">>> NOTE: Please run this command before using nnU-Net tools:"
echo "    source scripts/prepare_nnunet.sh"
echo ""
echo ">>> Next steps:"
echo "    1. Convert BraTS data to nnU-Net format (Task 137)."
echo "    2. Run: nnUNetv2_plan_and_preprocess -d 137 -c 3d_fullres"
