# BMDS-Net: A Bayesian Multi-Modal Deep Supervision Network

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)

Official implementation of the paper: **"BMDS-Net: A Bayesian Multi-Modal Deep Supervision Network for Robust Brain Tumor Segmentation"**.

## 🌟 Key Features
- **Robustness:** Zero-Init MMCF module handles missing modalities gracefully.
- **Precision:** Residue-Gated DDS enforces boundary precision (Low HD95).
- **Trustworthiness:** Bayesian Uncertainty Estimation via efficient fine-tuning.

## 📂 Project Structure
```text
BMDS-Net/
├── bmds_net/ # Source code
│ ├── models/ # BMDS-Net, SegResNet, MedNeXt
│ ├── engine/ # Trainers & Inferers
│ └── data/ # Data loaders & Transforms
├── configs/ # YAML configurations
├── tools/ # Training & Testing scripts
└── scripts/ # Shell scripts for one-click execution
```

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/YourUsername/BMDS-Net.git
cd BMDS-Net
pip install -r requirements.txt
```

### 2. Data Preparation
Download the BraTS 2021 dataset and organize it as follows:
```text
data/BraTS2021/
├── train/
│ ├── BraTS2021_00001/
│ │ ├── BraTS2021_00001_t1.nii.gz
│ │ ├── ...
│ │ └── BraTS2021_00001_seg.nii.gz
└── validation/
```

### 3. Training (Two-Stage Strategy)
To run the full pipeline (Deterministic Pre-training -> Bayesian Fine-tuning):
```bash
bash scripts/run_all_stages.sh
```

Or run stages individually:
```bash
# Stage 1: Robust Backbone
python -m torch.distributed.run --nproc_per_node=2 tools/train_stage1.py --config configs/bmds_net/stage1_deterministic.yaml

# Stage 2: Bayesian Fine-tuning
python tools/train_stage2.py --config configs/bmds_net/stage2_bayesian.yaml
```

### 4. Evaluation
```bash
python tools/test.py \
--config configs/bmds_net/stage2_bayesian.yaml \
--checkpoint work_dirs/stage2_bayesian/models/best_model.pth \
--mode bayesian \
--save_preds
```

## 📊 Results (BraTS 2021)

| Method | WT Dice | TC Dice | ET Dice | HD95 (Avg) |
| :--- | :---: | :---: | :---: | :---: |
| SegResNet | 0.905 | 0.881 | 0.858 | 8.49 |
| **BMDS-Net (Ours)** | **0.929** | **0.910** | **0.868** | **2.59** |

## 📜 Citation
If you find this code useful, please cite our paper:
```bibtex
@article{zhou2024bmdsnet,
title={BMDS-Net: A Bayesian Multi-Modal Deep Supervision Network for Robust Brain Tumor Segmentation},
author={Zhou, Yan and others},
journal={arXiv preprint},
year={2024}
}
```
```

---