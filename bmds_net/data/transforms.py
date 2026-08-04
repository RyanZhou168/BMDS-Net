"""Lightweight BraTS preprocessing and crop transforms."""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class BraTSTransform:
    def __init__(self, train: bool, target_size: Optional[Sequence[int]] = (128, 128, 128)):
        self.train = train
        self.target_size = tuple(target_size) if target_size is not None else None

    @staticmethod
    def _zscore(image: torch.Tensor) -> torch.Tensor:
        out = image.clone()
        for c in range(out.shape[0]):
            channel = out[c]
            mask = channel != 0
            if mask.any():
                mean = channel[mask].mean()
                std = channel[mask].std().clamp_min(1e-6)
                out[c] = (channel - mean) / std
        return out

    @staticmethod
    def _pad_to_size(image: torch.Tensor, label: torch.Tensor, size: Tuple[int, int, int]):
        spatial = image.shape[-3:]
        pads = []
        for current, target in reversed(list(zip(spatial, size))):
            total = max(target - current, 0)
            pads.extend([total // 2, total - total // 2])
        if any(pads):
            image = F.pad(image, pads)
            label = F.pad(label.unsqueeze(0).float(), pads).squeeze(0).long()
        return image, label

    def _crop(self, image: torch.Tensor, label: torch.Tensor):
        if self.target_size is None:
            return image, label
        image, label = self._pad_to_size(image, label, self.target_size)
        spatial = image.shape[-3:]
        starts = []
        for current, target in zip(spatial, self.target_size):
            max_start = current - target
            if self.train and max_start > 0:
                starts.append(int(torch.randint(0, max_start + 1, (1,)).item()))
            else:
                starts.append(max_start // 2)
        h, w, d = starts
        th, tw, td = self.target_size
        return image[:, h:h + th, w:w + tw, d:d + td], label[h:h + th, w:w + tw, d:d + td]

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = self._zscore(sample["image"])
        label = sample["label"]
        image, label = self._crop(image, label)
        sample["image"] = image.contiguous()
        sample["label"] = label.contiguous()
        return sample
