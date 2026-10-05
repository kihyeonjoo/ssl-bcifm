"""
DEAP raw EEG dataset — same interface as ``SEEDRawDataset``.

DEAP (Koelstra et al., 2012): 32 participants each watched the same 40
one-minute music-video excerpts in one session.  This loader reads the
*preprocessed MATLAB release* (``s01.mat`` .. ``s32.mat``):

* ``data``   (40 trials, 40 channels, 8064 samples) — 128 Hz, band-passed
  4-45 Hz, EOG removed, common-average referenced, trials re-ordered from
  presentation order to video (Experiment_id) order.  Each trial is 63 s:
  a 3-s pre-trial baseline followed by the 60-s excerpt.  Channels 1-32 are
  EEG in Geneva order, 33-40 are peripheral (EOG, EMG, GSR, ...).
* ``labels`` (40, 4) — the participant's OWN ratings (1-9) of valence,
  arousal, dominance and liking.

Labels (2026-10-05 user decision): four valence-arousal quadrants from the
participant's own ratings, threshold 5 —
``q = 2 * [valence > 5] + [arousal > 5]`` → 0 LVLA, 1 LVHA, 2 HVLA, 3 HVHA.
Binary valence / arousal are derived from the same model as ``q // 2`` and
``q % 2``.  Unlike SEED, the video does not fix the label: on average only
55% of participants share a video's majority quadrant.

Processing: drop the 3-s baseline, keep the 32 EEG channels, resample
128 → 200 Hz (``resample_poly`` 25/16) for LaBraM's 200-sample patches, so a
trial is 12000 samples = fifteen 4-s windows.  Amplitudes are µV like SEED,
so ``norm='scale100'`` applies unchanged (DEAP's 4-45 Hz band makes the
per-channel std about a third of SEED's).
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from data.seed_raw_dataset import SEEDRawDataset


# Geneva channel order of the preprocessed release, in LaBraM's 10-20 naming.
DEAP_CH_NAMES: List[str] = [
    "FP1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "OZ", "PZ",
    "FP2", "AF4", "FZ", "F4", "F8", "FC6", "FC2", "CZ",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]
DEAP_QUADRANTS = ("LVLA", "LVHA", "HVLA", "HVHA")
DEAP_N_TRIALS: int = 40
DEAP_FS_IN: int = 128
DEAP_BASELINE: int = 3 * DEAP_FS_IN          # 384 samples of pre-trial baseline


def quadrant(valence: float, arousal: float) -> int:
    """0 LVLA, 1 LVHA, 2 HVLA, 3 HVHA from 1-9 ratings (threshold 5, strict)."""
    return 2 * int(valence > 5) + int(arousal > 5)


class DEAPRawDataset(SEEDRawDataset):
    """DEAP with the SEEDRawDataset API.  One session per subject (session 1)."""

    n_subjects = 32
    n_classes = 4

    @property
    def n_channels(self) -> int:
        return len(DEAP_CH_NAMES)

    @property
    def ch_names(self) -> List[str]:
        return DEAP_CH_NAMES

    def _find_mat(self, subj: int, sess: int) -> Optional[str]:
        if sess != 1:                                # DEAP has a single session
            return None
        path = os.path.join(self.root, f"s{subj:02d}.mat")
        return path if os.path.exists(path) else None

    def _load_mat(self, path: str, subj: int, sess: int) -> None:
        z = sio.loadmat(path, variable_names=["data", "labels"])
        data, lab = z["data"], z["labels"]
        if data.shape != (DEAP_N_TRIALS, 40, 8064) or lab.shape != (DEAP_N_TRIALS, 4):
            raise ValueError(f"{path}: unexpected shapes data {data.shape}, labels {lab.shape}")
        for v in range(DEAP_N_TRIALS):
            eeg = data[v, :len(DEAP_CH_NAMES), DEAP_BASELINE:]          # (32, 7680) @ 128 Hz
            eeg = resample_poly(eeg, 25, 16, axis=1)                    # (32, 12000) @ 200 Hz
            self._slice_trial(
                eeg.astype(np.float32), quadrant(lab[v, 0], lab[v, 1]), subj, sess, v + 1,
            )
            self._trial_counter += 1
