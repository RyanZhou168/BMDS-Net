"""BraTS folder dataset.

Expected structure:

data/BraTS2021/
  train/
    BraTS2021_00001/
      BraTS2021_00001_flair.nii.gz
      BraTS2021_00001_t1.nii.gz
      BraTS2021_00001_t1ce.nii.gz
      BraTS2021_00001_t2.nii.gz
      BraTS2021_00001_seg.nii.gz
  validation/
    ...
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    modalities = ("flair", "t1", "t1ce", "t2")

    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.data_dir = self.root_dir / split
        self.transform = transform
        if not self.data_dir.exists():
            raise FileNotFoundError(f"BraTS split directory not found: {self.data_dir}")
        self.case_dirs: List[Path] = sorted(p for p in self.data_dir.iterdir() if p.is_dir())
        self.cases = [p.name for p in self.case_dirs]
        if not self.case_dirs:
            raise RuntimeError(f"No case folders found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.case_dirs)

    def _find_file(self, case_dir: Path, suffix: str) -> Path:
        case_id = case_dir.name
        candidates = [
            case_dir / f"{case_id}_{suffix}.nii.gz",
            case_dir / f"{case_id}_{suffix}.nii",
        ]
        for path in candidates:
            if path.exists():
                return path
        matches = sorted(case_dir.glob(f"*_{suffix}.nii*"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Missing {suffix} image for case {case_id}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case_dir = self.case_dirs[index]
        images = []
        for modality in self.modalities:
            arr = nib.load(str(self._find_file(case_dir, modality))).get_fdata(dtype=np.float32)
            images.append(arr)
        image = np.stack(images, axis=0)

        label = nib.load(str(self._find_file(case_dir, "seg"))).get_fdata(dtype=np.float32).astype(np.int64)
        label[label == 4] = 3

        sample = {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).long(),
            "case_id": case_dir.name,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
