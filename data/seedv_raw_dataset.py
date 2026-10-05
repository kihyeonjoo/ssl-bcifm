"""
SEED-V raw EEG dataset — same interface as ``SEEDRawDataset``.

SEED-V ships **unprocessed** Neuroscan ``.cnt`` recordings, unlike SEED and
SEED-IV which ship a cleaned 200 Hz ``.mat``.  To make the three datasets
comparable this module reproduces SEED's published preprocessing:

1. keep the 62 SEED electrodes, in SEED's order (the file also carries
   M1, M2, VEO, HEO — mastoids and EOG, which SEED drops);
2. band-pass 0.3–75 Hz, mirroring SEED's 0–75 Hz cleanup;
3. resample 1000 Hz → 200 Hz;
4. cut the 15 film clips with the official timestamps.

Step 1–3 cost about a minute per recording, so the result is cached as
``<cache_dir>/<stem>.npy``; later runs read the cache.  The dataset root is
usually read-only, so the cache defaults to ``SEEDV_CACHE`` in the environment
or a directory beside the project.

Labels are 0 disgust, 1 fear, 2 sad, 3 neutral, 4 happy.  Note the official
stimulus table gives sessions 2 and 3 the *same* clip order — that is what the
spreadsheet says, not a copy-paste error on our side.

SEED-V is also the one SEED-family set that LaBraM did **not** pretrain on,
which makes it the cleanest external check of anything measured on SEED.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np

from data.seed_raw_dataset import SEEDRawDataset, SEED_CH_NAMES

SEEDV_N_CLIPS: int = 15
SEEDV_N_CLASSES: int = 5

_EMO = {"disgust": 0, "fear": 1, "sad": 2, "neutral": 3, "happy": 4}

_ORDER: Dict[int, List[str]] = {
    1: ["happy", "fear", "neutral", "sad", "disgust"] * 3,
    2: ["sad", "fear", "neutral", "disgust", "happy",
        "happy", "disgust", "neutral", "sad", "fear",
        "neutral", "happy", "fear", "sad", "disgust"],
}
_ORDER[3] = list(_ORDER[2])          # the spreadsheet repeats session 2's order

SEEDV_LABEL_SEQ: Dict[int, List[int]] = {
    s: [_EMO[e] for e in order] for s, order in _ORDER.items()
}

# trial_start_end_timestamp.txt, in seconds.
SEEDV_START: Dict[int, List[int]] = {
    1: [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
    2: [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
    3: [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
}
SEEDV_END: Dict[int, List[int]] = {
    1: [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
    2: [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
    3: [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066],
}

TARGET_SFREQ = 200

# The dataset root is typically read-only, so cache elsewhere.
DEFAULT_CACHE = os.environ.get(
    "SEEDV_CACHE", os.path.expanduser("~/seedv_cache200"))


def _preprocess_cnt(path: str, cache_dir: str) -> np.ndarray:
    """(62, T) float32 at 200 Hz, cached on disk."""
    stem = os.path.splitext(os.path.basename(path))[0]
    cached = os.path.join(cache_dir, stem + ".npy")
    if os.path.exists(cached):
        return np.load(cached, mmap_mode="r")

    import mne
    mne.set_log_level("ERROR")

    try:
        raw = mne.io.read_raw_cnt(path, preload=True)
    except RuntimeError:
        # Some recordings (subjects 7-9) have a sample count that overflows in
        # the CNT header, so MNE cannot infer the word size and refuses.  Every
        # file in this set that MNE *does* read comes out at 4 bytes/sample, and
        # int32 is the only choice giving a plausible duration here, so force it.
        raw = mne.io.read_raw_cnt(path, preload=True, data_format="int32")
    rename = {c: c.upper() for c in raw.ch_names if c != c.upper()}
    if rename:
        raw.rename_channels(rename)
    missing = [c for c in SEED_CH_NAMES if c not in raw.ch_names]
    if missing:
        raise ValueError(f"{path}: missing SEED channels {missing}")
    raw.pick(SEED_CH_NAMES)                       # also fixes the order
    raw.filter(0.3, 75.0, fir_design="firwin")
    raw.resample(TARGET_SFREQ)

    arr = raw.get_data().astype(np.float32) * 1e6   # volts -> microvolts
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cached, arr)
    return arr


class SEEDVRawDataset(SEEDRawDataset):
    """SEED-V with the SEEDRawDataset API (``subjects_of``, ``clips_of``, EA…)."""

    n_classes = SEEDV_N_CLASSES
    n_subjects = 16

    def _find_mat(self, subj: int, sess: int) -> Optional[str]:
        d = os.path.join(self.root, "EEG_raw")
        if not os.path.isdir(d):
            return None
        pat = re.compile(rf"^{subj}_{sess}_\d+\.cnt$")
        hits = sorted(f for f in os.listdir(d) if pat.match(f))
        return os.path.join(d, hits[0]) if hits else None

    def _load_mat(self, path: str, subj: int, sess: int) -> None:
        eeg = _preprocess_cnt(path, DEFAULT_CACHE)
        seq = SEEDV_LABEL_SEQ[sess]
        starts, ends = SEEDV_START[sess], SEEDV_END[sess]
        for clip in range(1, SEEDV_N_CLIPS + 1):
            a = starts[clip - 1] * TARGET_SFREQ
            b = ends[clip - 1] * TARGET_SFREQ
            if b > eeg.shape[1]:
                raise ValueError(
                    f"{path}: clip {clip} ends at sample {b} but recording has "
                    f"{eeg.shape[1]} — timestamps and recording disagree")
            self._slice_trial(
                np.ascontiguousarray(eeg[:, a:b], dtype=np.float32),
                seq[clip - 1], subj, sess, clip,
            )
            self._trial_counter += 1
