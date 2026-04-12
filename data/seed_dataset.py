"""
PyTorch Dataset for the SEED EEG corpus.

Expected directory layout
─────────────────────────
<root>/
  Preprocessed_EEG/
    1/                  ← subject 1
      <session>.mat     ← raw trial EEG; each variable is (62, T) @ 200 Hz
    2/
    …
    15/

Each .mat file contains variables whose values are (62, T) ndarrays.
Variable names typically look like '<subject_initials>_eeg<N>' (e.g. 'djc_eeg1').
Any variable whose shape is (62, T) is treated as a trial (in order).

Labels follow the official SEED 15-clip sequence:
  [+1, 0, −1, −1, 0, +1, −1, 0, +1, +1, 0, −1, 0, +1, −1]
where  1 = positive,  0 = neutral,  −1 = negative.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from data.preprocessing import BandSTFT, LEFT_IDX, RIGHT_IDX


# Official SEED sentiment label sequence for the 15 film clips (1-indexed clip → label)
_SEED_LABEL_SEQ: List[int] = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]


class SEEDDataset(Dataset):
    """Sliding-window EEG segments from SEED, split into left / right hemispheres.

    Each item is a dict:
    {
        'left'  : Tensor (C_h, 4, T_frames),  # left  hemisphere band power
        'right' : Tensor (C_h, 4, T_frames),  # right hemisphere band power
        'label' : Tensor scalar  (0 / 1 / 2 after remapping, or −1/0/1 raw)
    }

    C_h = 27  (midline channels discarded)
    4   = [theta, alpha, beta, gamma] bands
    T_frames ≈ (segment_length − n_fft) // hop_length + 1

    Parameters
    ----------
    root : str
        Path to the SEED dataset root (contains Preprocessed_EEG/).
    subjects : list[int] | None
        1-based subject IDs to load.  None → all 15 subjects.
    sessions : list[int] | None
        Session IDs to load (1, 2, or 3).  None → all sessions.
    segment_length : int
        Window size in samples (default 800 = 4 s @ 200 Hz).
    step : int
        Sliding window stride in samples (default 200 = 1 s, 75 % overlap).
    n_fft : int
        STFT FFT size (default 200 → 1 Hz resolution at 200 Hz).
    hop_length : int
        STFT hop in samples (default 40 → 0.2 s time step).
    remap_labels : bool
        If True, remap −1/0/1 → 0/1/2 for CrossEntropyLoss compatibility.
    transform : callable | None
        Optional transform applied to each (left, right) feature tensor pair.
    """

    FS: int = 200

    def __init__(
        self,
        root: str,
        subjects: Optional[List[int]] = None,
        sessions: Optional[List[int]] = None,
        segment_length: int = 800,
        step: int = 200,
        n_fft: int = 200,
        hop_length: int = 40,
        remap_labels: bool = True,
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.segment_length = segment_length
        self.step = step
        self.remap_labels = remap_labels
        self.transform = transform
        self.normalize = normalize

        self.band_stft = BandSTFT(fs=self.FS, n_fft=n_fft, hop_length=hop_length)

        # List of (eeg_segment: np.ndarray (62, L), label: int, trial_id: int)
        # trial_id is a global integer that uniquely identifies each EEG trial
        # across subjects and sessions — used by TemporalPairDataset.
        self._segments: List[Tuple[np.ndarray, int, int]] = []
        self._trial_to_indices: Dict[int, List[int]] = {}  # trial_id → segment indices
        self._trial_counter: int = 0   # incremented once per trial loaded
        self._load(subjects or list(range(1, 16)), sessions or [1, 2, 3])

    # ── public helpers ────────────────────────────────────────────────────────

    @property
    def n_channels_per_hemi(self) -> int:
        return len(LEFT_IDX)  # 27

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        eeg_np, label, _trial_id = self._segments[idx]

        eeg = torch.from_numpy(eeg_np)  # (62, L)

        # Hemisphere split — midline electrodes are implicitly dropped
        left_eeg  = eeg[LEFT_IDX]   # (27, L)
        right_eeg = eeg[RIGHT_IDX]  # (27, L)

        # STFT band-power envelopes  →  (27, 4, T_frames)
        left_feat  = self.band_stft(left_eeg)
        right_feat = self.band_stft(right_eeg)

        # Per-channel z-score normalisation (across bands × time)
        # Removes subject-level scale drift while preserving band structure.
        if self.normalize:
            left_feat  = self._zscore_per_channel(left_feat)
            right_feat = self._zscore_per_channel(right_feat)

        if self.transform is not None:
            left_feat, right_feat = self.transform(left_feat, right_feat)

        return {
            "left":  left_feat,                                   # (27, 4, T_frames)
            "right": right_feat,                                  # (27, 4, T_frames)
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def _zscore_per_channel(feat: torch.Tensor) -> torch.Tensor:
        """Per-channel z-score across (bands, time).

        Input  : (C, B, T)
        Output : (C, B, T)  — each channel has mean 0, std 1 across B × T
        """
        mean = feat.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
        std  = feat.std(dim=(1, 2), keepdim=True) + 1e-6
        return (feat - mean) / std

    # ── internal loading ──────────────────────────────────────────────────────

    def _load(self, subjects: List[int], sessions: List[int]) -> None:
        for subj in subjects:
            for sess in sessions:
                path = self._find_mat(subj, sess)
                if path is None:
                    continue
                self._load_mat(path)

    def _load_mat(self, path: str) -> None:
        mat = sio.loadmat(path, verify_compressed_data_integrity=False)

        # Collect trial arrays: any (62, T) ndarray that is not a metadata key
        trial_arrays: List[np.ndarray] = [
            mat[k].astype(np.float32)
            for k in sorted(mat.keys())
            if not k.startswith("_")
            and isinstance(mat[k], np.ndarray)
            and mat[k].ndim == 2
            and mat[k].shape[0] == 62
        ]

        for trial_idx, eeg in enumerate(trial_arrays):
            raw_label = _SEED_LABEL_SEQ[trial_idx % len(_SEED_LABEL_SEQ)]
            label = raw_label + 1 if self.remap_labels else raw_label  # −1/0/1 → 0/1/2
            self._slice_trial(eeg, label)
            self._trial_counter += 1

    def _slice_trial(self, eeg: np.ndarray, label: int) -> None:
        """Slice a single trial (62, T) into overlapping windows."""
        trial_id = self._trial_counter
        T = eeg.shape[1]
        start = 0
        while start + self.segment_length <= T:
            seg_idx = len(self._segments)
            seg = eeg[:, start : start + self.segment_length]  # (62, L)  — copy-on-slice
            self._segments.append((np.array(seg, copy=False), label, trial_id))
            self._trial_to_indices.setdefault(trial_id, []).append(seg_idx)
            start += self.step

    def _find_mat(self, subj: int, sess: int) -> Optional[str]:
        """Return the .mat file path for a given subject / session, or None.

        Supports two directory layouts:
          Layout A (flat)  : Preprocessed_EEG/<subj>_<date>.mat
          Layout B (nested): Preprocessed_EEG/<subj>/<date>.mat
        """
        eeg_dir = os.path.join(self.root, "Preprocessed_EEG")

        # ── Layout A: flat files named  <subj>_<date>.mat ────────────────
        prefix = f"{subj}_"
        candidates = sorted(
            f for f in os.listdir(eeg_dir)
            if f.startswith(prefix) and f.endswith(".mat")
        )
        if candidates:
            if len(candidates) >= sess:
                return os.path.join(eeg_dir, candidates[sess - 1])
            return None

        # ── Layout B: nested  <subj>/<session>.mat ───────────────────────
        subj_dir = os.path.join(eeg_dir, str(subj))
        if not os.path.isdir(subj_dir):
            return None
        nested = sorted(f for f in os.listdir(subj_dir) if f.endswith(".mat"))
        if len(nested) >= sess:
            return os.path.join(subj_dir, nested[sess - 1])
        return None
