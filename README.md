# BMDS-Net

Official implementation for:

**BMDS-Net: Deployment-aware multi-modal brain tumor segmentation with adaptive fusion, decoder regularization, and Bayesian calibration**

BMDS-Net augments a Swin UNETR backbone with three deployment-oriented components:

- **MMCF**: zero-initialized multimodal contextual fusion for adaptive input-level modality reweighting.
- **DDS**: residual-gated deep decoder supervision for boundary-sensitive reconstruction.
- **Bayesian output layer**: last-layer stochastic fine-tuning for Monte Carlo uncertainty estimates.

The current code follows the manuscript convention of predicting three overlapping BraTS region channels with independent sigmoid outputs: whole tumor (WT), tumor core (TC), and enhancing tumor (ET).

## Repository structure

```text
BMDS-Net/
├── bmds_net/
│   ├── data/          # BraTS NIfTI dataset and crop transforms
│   ├── engine/        # losses, trainers, inferer
│   ├── models/        # BMDS-Net, Bayesian layer, baseline wrappers
│   └── utils/         # metrics, logging, checkpoints
├── configs/
│   ├── bmds_net/      # two-stage BMDS-Net training
│   ├── ablation/      # configuration-comparison recipes
│   └── baselines/     # representative baseline configs
├── tools/             # training, testing, robustness analysis
└── scripts/           # convenience launch scripts
```

## Installation

```bash
git clone git@github.com:RyanZhou168/BMDS-Net.git
cd BMDS-Net
pip install -r requirements.txt
pip install -e .
```

## Data layout

Download BraTS and organize cases as NIfTI folders:

```text
data/BraTS2021/
├── train/
│   └── BraTS2021_00001/
│       ├── BraTS2021_00001_flair.nii.gz
│       ├── BraTS2021_00001_t1.nii.gz
│       ├── BraTS2021_00001_t1ce.nii.gz
│       ├── BraTS2021_00001_t2.nii.gz
│       └── BraTS2021_00001_seg.nii.gz
└── validation/
    └── ...
```

The loader uses modality order `FLAIR, T1, T1ce, T2`. Raw BraTS label value `4` is mapped to ET internally and region metrics are computed as WT, TC, and ET masks.

## Training

Run the full two-stage pipeline:

```bash
bash scripts/run_all_stages.sh
```

Or run each stage separately:

```bash
# Stage 1: deterministic BMDS-Net
python -m torch.distributed.run --nproc_per_node=2 tools/train_stage1.py \
  --config configs/bmds_net/stage1_deterministic.yaml

# Stage 2: last-layer Bayesian fine-tuning
python tools/train_stage2.py \
  --config configs/bmds_net/stage2_bayesian.yaml
```

## Configuration comparison

The manuscript reports configuration-level comparisons rather than strict single-factor isolation for every row. The public configs use the following naming:

```text
configs/baselines/swin_unetr.yaml        # Baseline
configs/ablation/loss_only.yaml          # BoundaryFocalDice only
configs/ablation/aux_supervision_only.yaml
configs/ablation/mmcf_only.yaml
configs/bmds_net/stage1_deterministic.yaml
```

`aux_supervision_only` corresponds to training-time auxiliary decoder heads without the complete MMCF-derived gated DDS inference pathway. The complete BMDS-Net uses BoundaryFocalDice, MMCF residual fusion, auxiliary decoder supervision, gated DDS, and feature consistency regularization.

## Evaluation

Deterministic inference:

```bash
python tools/test.py \
  --config configs/bmds_net/stage1_deterministic.yaml \
  --checkpoint work_dirs/stage1_deterministic/models/best_model.pth \
  --mode deterministic \
  --save_preds
```

Bayesian Monte Carlo inference:

```bash
python tools/test.py \
  --config configs/bmds_net/stage2_bayesian.yaml \
  --checkpoint work_dirs/stage2_bayesian/models/best_model.pth \
  --mode bayesian \
  --save_preds
```

Missing-modality robustness analysis:

```bash
python tools/analyze_robustness.py \
  --config configs/bmds_net/stage1_deterministic.yaml \
  --checkpoint work_dirs/stage1_deterministic/models/best_model.pth \
  --output_csv robustness_results.csv
```

## Manuscript results

BraTS 2021 validation:

| Model | WT Dice | TC Dice | ET Dice | WT HD95 | TC HD95 | ET HD95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin UNETR | 0.9279 | 0.9104 | 0.8635 | 2.30 | 2.39 | 3.84 |
| BMDS-Net | 0.9293 | 0.9098 | 0.8675 | 2.27 | 2.22 | 3.27 |

BraTS 2020 same-protocol internal configuration comparison:

| Configuration | WT Dice | TC Dice | ET Dice | Mean Dice |
| :--- | ---: | ---: | ---: | ---: |
| Baseline | 0.8976 | 0.8435 | 0.7158 | 0.8189 |
| Loss only | 0.8967 | 0.8329 | 0.7193 | 0.8163 |
| Aux. supervision only | 0.8889 | 0.8085 | 0.6931 | 0.7968 |
| MMCF only | 0.8601 | 0.7526 | 0.7147 | 0.7758 |
| BMDS-Net | 0.8969 | 0.8443 | 0.7273 | 0.8229 |

System-level calibration on BraTS 2021:

| Model | Training cost | ECE | NLL |
| :--- | ---: | ---: | ---: |
| Deterministic Swin UNETR | 1.0x | 0.0152 | 0.0240 |
| Three-model ensemble | 3.0x | 0.0035 | 0.0033 |
| Bayesian BMDS-Net | 1.2x | 0.0037 | 0.0037 |

## Citation

```bibtex
@article{zhou2026bmdsnet,
  title={BMDS-Net: Deployment-aware multi-modal brain tumor segmentation with adaptive fusion, decoder regularization, and Bayesian calibration},
  author={Zhou, Yan and others},
  journal={Under Review},
  year={2026}
}
```
