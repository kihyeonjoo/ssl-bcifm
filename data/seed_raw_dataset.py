"""
SEED Raw EEG Dataset — outputs raw time-domain EEG (not STFT features).

Used by the LaBraM-based pipeline. LaBraM expects input of shape
    (B, n_channels, n_patches, patch_size)
where patch_size = 200 samples (1 second at 200 Hz) and n_patches is
determined by the segment length (4 s → 4 patches).

This is a separate dataset from the original ``SEEDDataset`` which emits
log-power STFT features. The original pipeline is untouched.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from data.preprocessing import LEFT_IDX, RIGHT_IDX
from data.alignment import EuclideanAligner


# SEED sentiment labels for the 15 film clips (1-indexed → label)
_SEED_LABEL_SEQ: List[int] = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# SEED official subject-dependent protocol: first 9 clips train, last 6 test.
SD_TRAIN_CLIPS: int = 9


def _trial_no(key: str) -> int:
    """Film-clip index (1..15) encoded in a SEED variable name.

    SEED stores one variable per clip, named ``<initials>_eeg<N>``
    (e.g. ``djc_eeg12``).  The numeric suffix is the clip index and is what
    ``_SEED_LABEL_SEQ`` is indexed by — NOT the variable's position in a
    lexicographic sort, which orders ``_eeg10`` directly after ``_eeg1``.
    """
    m = re.search(r"(\d+)$", key)
    if m is None:
        raise ValueError(f"cannot parse trial number from SEED key {key!r}")
    return int(m.group(1))

# 62-channel SEED layout (same order as preprocessing.SEED_CH_NAMES).
# This matches LaBraM's standard_1020 naming convention.
SEED_CH_NAMES: List[str] = [
    "FP1",  "FPZ",  "FP2",
    "AF3",  "AF4",
    "F7",  "F5",  "F3",  "F1",  "FZ",  "F2",  "F4",  "F6",  "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7",  "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",  "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7",  "P5",  "P3",  "P1",  "PZ",  "P2",  "P4",  "P6",  "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1",  "OZ",  "O2",  "CB2",
]


class SEEDRawDataset(Dataset):
    """Raw EEG segments in LaBraM-compatible shape.

    Each item:
      {
        'eeg'   : Tensor (n_channels, n_patches, patch_size)
                  — 62ch, 4 patches, 200 samples/patch for a 4 s segment
        'label' : Tensor scalar (0 / 1 / 2 after remapping)
      }

    The raw (62, L) segment is z-score normalised per channel across the
    segment's time axis — standard LaBraM preprocessing. Then reshaped
    into patches of ``patch_size`` samples each.

    Parameters
    ----------
    root            : str
    subjects        : list[int] | None
    sessions        : list[int] | None
    segment_length  : int   — samples per segment (default 800 = 4 s)
    step            : int   — sliding window stride
    patch_size      : int   — LaBraM patch size (default 200 = 1 s)
    norm            : str   — 'scale100' (LaBraM official: µV → 0.1 mV),
                              'zscore' (per-channel z-score within a segment)
                              or 'none'
    normalize       : bool  — DEPRECATED alias: True → 'zscore', False → 'none'
    ea              : bool  — apply Euclidean Alignment (transductive: the
                              transform for a subject is fitted on that
                              subject's own unlabelled EEG)
    ea_scope        : str   — 'session' (default; each recording aligned
                              separately) or 'subject'
    ea_scale        : float — global constant applied after whitening, to put
                              the signal back in the backbone's input range.
                              0.2 matches the median channel std of the
                              scale100 pipeline (0.131) on SEED.
                              MUST be identical across train/val/test splits,
                              otherwise it reintroduces a subject-wise scale.
    ea_trim         : float — fraction of highest-power segments dropped when
                              estimating the covariance (artifact robustness)
    ea_eps          : float — eigenvalue floor, relative to the largest
    """

    FS: int = 200

    def __init__(
        self,
        root: str,
        subjects: Optional[List[int]] = None,
        sessions: Optional[List[int]] = None,
        segment_length: int = 800,
        step: int = 200,
        patch_size: int = 200,
        norm: str = "scale100",
        normalize: Optional[bool] = None,
        ea: bool = False,
        ea_scope: str = "session",
        ea_scale: float = 0.2,
        ea_trim: float = 0.05,
        ea_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert segment_length % patch_size == 0, (
            f"segment_length ({segment_length}) must be divisible by "
            f"patch_size ({patch_size})"
        )
        self.root = root
        self.segment_length = segment_length
        self.step = step
        self.patch_size = patch_size
        self.n_patches = segment_length // patch_size

        if normalize is not None:
            warnings.warn(
                "SEEDRawDataset(normalize=...) is deprecated; use norm=",
                DeprecationWarning, stacklevel=2,
            )
            norm = "zscore" if normalize else "none"
        if norm not in ("scale100", "zscore", "none"):
            raise ValueError(f"norm must be scale100|zscore|none, got {norm!r}")
        self.norm = norm

        # (eeg_segment (62, L), label, trial_id)
        self._segments: List[Tuple[np.ndarray, int, int]] = []
        # per-segment provenance, parallel to ``_segments``
        self._seg_meta: List[Tuple[int, int, int]] = []   # (subject, session, clip 1..15)
        self._trial_to_indices: Dict[int, List[int]] = {}
        self._trial_counter: int = 0
        self._load(subjects or list(range(1, 16)), sessions or [1, 2, 3])

        if ea_scope not in ("session", "subject"):
            raise ValueError(f"ea_scope must be session|subject, got {ea_scope!r}")
        self.ea = ea
        self.ea_scope = ea_scope
        if ea and norm != "none":
            # EA sets the output amplitude itself (via ea_scale); `norm` is
            # bypassed in __getitem__ so the two cannot compound.
            warnings.warn(
                f"ea=True supersedes norm={norm!r}; ea_scale governs the "
                f"input amplitude instead.",
                stacklevel=2,
            )
        self._aligner: Optional[EuclideanAligner] = None
        self._seg_group: List[tuple] = []
        if ea:
            self._fit_alignment(ea_scale, ea_trim, ea_eps)

    # ── public ─────────────────────────────────────────────────────────
    @property
    def n_channels(self) -> int:
        return 62

    @property
    def ch_names(self) -> List[str]:
        return SEED_CH_NAMES

    @property
    def subjects_of(self) -> List[int]:
        """Subject id per segment."""
        return [m[0] for m in self._seg_meta]

    @property
    def sessions_of(self) -> List[int]:
        """Session index (1..3) per segment."""
        return [m[1] for m in self._seg_meta]

    @property
    def clips_of(self) -> List[int]:
        """Film-clip index (1..15) per segment — use this for protocol splits."""
        return [m[2] for m in self._seg_meta]

    def sd_split(self, n_train_clips: int = SD_TRAIN_CLIPS):
        """SEED official subject-dependent split indices.

        Clips 1..``n_train_clips`` → train, the rest → test, applied inside
        every (subject, session) present in this dataset.  Train and test
        never share a film clip, so overlapping sliding windows cannot leak.
        """
        train_idx = [i for i, c in enumerate(self.clips_of) if c <= n_train_clips]
        test_idx  = [i for i, c in enumerate(self.clips_of) if c >  n_train_clips]
        return train_idx, test_idx

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        eeg_np, label, _tid = self._segments[idx]
        if self.ea:
            # Whitening is scale-invariant, so it commutes with `norm` below;
            # the global ea_scale sets the final amplitude.
            eeg_np = self._aligner.apply(self._seg_group[idx], eeg_np)
        eeg = torch.from_numpy(np.ascontiguousarray(eeg_np)).float()  # (62, L)

        if self.ea:
            # EA already fixed the amplitude via ea_scale; applying `norm` on
            # top would rescale it again.
            pass
        elif self.norm == "scale100":
            # LaBraM official preprocessing: µV → 0.1 mV, preserving the
            # absolute amplitude scale the backbone was pretrained on.
            eeg = eeg / 100.0
        elif self.norm == "zscore":
            mean = eeg.mean(dim=-1, keepdim=True)
            std  = eeg.std(dim=-1, keepdim=True) + 1e-6
            eeg = (eeg - mean) / std

        # Reshape into (n_channels, n_patches, patch_size)
        eeg = eeg.reshape(self.n_channels, self.n_patches, self.patch_size)

        return {
            "eeg":   eeg,
            "label": torch.tensor(label, dtype=torch.long),
        }

    # ── Euclidean Alignment ────────────────────────────────────────────
    def _group_key(self, subj: int, sess: int):
        return (subj, sess) if self.ea_scope == "session" else (subj,)

    def _fit_alignment(self, scale: float, trim: float, eps: float) -> None:
        """Fit one whitening transform per group from this dataset's segments.

        Only this dataset's own data is used, so fitting a train/val/test
        split separately is correct — every subject's transform comes from
        that subject alone.
        """
        groups: Dict[tuple, List[np.ndarray]] = {}
        self._seg_group = []
        for (seg, _lab, _tid), (subj, sess, _clip) in zip(
            self._segments, self._seg_meta
        ):
            key = self._group_key(subj, sess)
            self._seg_group.append(key)
            groups.setdefault(key, []).append(seg)

        self._aligner = EuclideanAligner(trim=trim, eps=eps, scale=scale).fit(groups)

    def alignment_report(self):
        """Per-group whitening diagnostics; see EuclideanAligner.report."""
        if self._aligner is None:
            return []
        groups: Dict[tuple, List[np.ndarray]] = {}
        for (seg, _l, _t), key in zip(self._segments, self._seg_group):
            groups.setdefault(key, []).append(seg)
        return self._aligner.report(groups)

    # ── internal loading ───────────────────────────────────────────────
    def _load(self, subjects: List[int], sessions: List[int]) -> None:
        for subj in subjects:
            for sess in sessions:
                path = self._find_mat(subj, sess)
                if path is None:
                    continue
                self._load_mat(path, subj, sess)

    def _load_mat(self, path: str, subj: int, sess: int) -> None:
        mat = sio.loadmat(path, verify_compressed_data_integrity=False)
        eeg_keys = sorted(
            (
                k for k in mat.keys()
                if not k.startswith("_")
                and isinstance(mat[k], np.ndarray)
                and mat[k].ndim == 2
                and mat[k].shape[0] == 62
            ),
            key=_trial_no,          # NOT lexicographic — see _trial_no docstring
        )
        for k in eeg_keys:
            clip = _trial_no(k)                        # 1-indexed film clip
            label = _SEED_LABEL_SEQ[clip - 1] + 1      # -1/0/1 → 0/1/2
            self._slice_trial(
                mat[k].astype(np.float32), label, subj, sess, clip,
            )
            self._trial_counter += 1

    def _slice_trial(
        self,
        eeg: np.ndarray,
        label: int,
        subj: int,
        sess: int,
        clip: int,
    ) -> None:
        trial_id = self._trial_counter
        T = eeg.shape[1]
        start = 0
        while start + self.segment_length <= T:
            seg_idx = len(self._segments)
            seg = eeg[:, start : start + self.segment_length]
            self._segments.append((np.array(seg, copy=False), label, trial_id))
            self._seg_meta.append((subj, sess, clip))
            self._trial_to_indices.setdefault(trial_id, []).append(seg_idx)
            start += self.step

    def _find_mat(self, subj: int, sess: int) -> Optional[str]:
        """Locate a SEED .mat file — supports flat and nested layouts."""
        eeg_dir = os.path.join(self.root, "Preprocessed_EEG")

        prefix = f"{subj}_"
        candidates = sorted(
            f for f in os.listdir(eeg_dir)
            if f.startswith(prefix) and f.endswith(".mat")
        )
        if candidates:
            if len(candidates) >= sess:
                return os.path.join(eeg_dir, candidates[sess - 1])
            return None

        subj_dir = os.path.join(eeg_dir, str(subj))
        if not os.path.isdir(subj_dir):
            return None
        nested = sorted(f for f in os.listdir(subj_dir) if f.endswith(".mat"))
        if len(nested) >= sess:
            return os.path.join(subj_dir, nested[sess - 1])
        return None


class SEEDRawPairDataset(Dataset):
    """Same-trial segment pairs for the temporal delta asymmetry task.

    Wraps a SEEDRawDataset and returns two segments (t₁, t₂) drawn from
    the same EEG trial.
    """

    def __init__(self, base_dataset: SEEDRawDataset) -> None:
        super().__init__()
        import random
        self._random = random
        self.base = base_dataset

        self._valid_indices: List[int] = []
        for trial_id, seg_indices in base_dataset._trial_to_indices.items():
            if len(seg_indices) >= 2:
                self._valid_indices.extend(seg_indices)

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
        i2 = i1
        while i2 == i1:
            i2 = self._random.choice(siblings)

        item1 = self.base[i1]
        item2 = self.base[i2]

        return {
            "eeg_t1": item1["eeg"],
            "eeg_t2": item2["eeg"],
            "label":  item1["label"],
        }
