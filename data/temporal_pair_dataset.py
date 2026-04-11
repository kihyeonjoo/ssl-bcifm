"""
TemporalPairDataset — same-trial segment pair sampler for ℒ_aux.

Wraps a SEEDDataset and yields pairs (t₁, t₂) drawn from the same EEG trial,
required by the Temporal Delta Asymmetry auxiliary task.
"""

from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from data.seed_dataset import SEEDDataset


class TemporalPairDataset(Dataset):
    """Pairs of segments from the same trial for temporal asymmetry prediction.

    For each index *i*, the dataset returns the segment at *i* as t₁ and
    randomly samples a *different* segment from the same trial as t₂.

    Items
    -----
    {
        'left_t1'  : Tensor (C_h, 4, T_frames),
        'right_t1' : Tensor (C_h, 4, T_frames),
        'left_t2'  : Tensor (C_h, 4, T_frames),
        'right_t2' : Tensor (C_h, 4, T_frames),
        'label'    : Tensor scalar,
    }

    Trials with only one segment are skipped (filtered out at init).
    """

    def __init__(self, base_dataset: SEEDDataset) -> None:
        super().__init__()
        self.base = base_dataset

        # Only keep indices whose trial has at least 2 segments
        self._valid_indices: List[int] = []
        for trial_id, seg_indices in base_dataset._trial_to_indices.items():
            if len(seg_indices) >= 2:
                self._valid_indices.extend(seg_indices)

        # Build reverse map: segment index → trial_id
        self._seg_to_trial: Dict[int, int] = {}
        for trial_id, seg_indices in base_dataset._trial_to_indices.items():
            for si in seg_indices:
                self._seg_to_trial[si] = trial_id

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i1 = self._valid_indices[idx]
        trial_id = self._seg_to_trial[i1]
        siblings = self.base._trial_to_indices[trial_id]

        # Sample a different segment from the same trial
        i2 = i1
        while i2 == i1:
            i2 = random.choice(siblings)

        item1 = self.base[i1]
        item2 = self.base[i2]

        return {
            "left_t1":  item1["left"],
            "right_t1": item1["right"],
            "left_t2":  item2["left"],
            "right_t2": item2["right"],
            "label":    item1["label"],
        }
