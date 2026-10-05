"""
SEED-IV raw EEG dataset — same interface as ``SEEDRawDataset``.

SEED-IV is the four-emotion sibling of SEED: 15 subjects x 3 sessions,
24 film clips per session, 62 channels at 200 Hz, stored as one MATLAB
variable per clip exactly like SEED (``<initials>_eeg<N>``).

Two things differ and both matter:

* **Labels vary by session.**  SEED reuses one clip order across all three
  sessions, so a single sequence suffices.  SEED-IV shuffles the clips per
  session, so the label depends on (session, clip).  Indexing the wrong
  session's sequence silently mislabels roughly three quarters of the data —
  the same class of bug that voided the first round of SEED results.
* **Four classes**, already 0-3 (neutral / sad / fear / happy), no remap.

Directory layout::

    SEED-IV/eeg_raw_data/{1,2,3}/<subject>_<date>.mat
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import scipy.io as sio

from data.seed_raw_dataset import SEEDRawDataset, _trial_no


# Official label sequences, one per session (ReadMe.txt).
# 0 neutral, 1 sad, 2 fear, 3 happy.
SEEDIV_LABEL_SEQ: dict[int, List[int]] = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1,
        1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3,
        2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0,
        2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

SEEDIV_N_CLIPS: int = 24
SEEDIV_N_CLASSES: int = 4


class SEEDIVRawDataset(SEEDRawDataset):
    """SEED-IV with the SEEDRawDataset API (``subjects_of``, ``clips_of``, EA…)."""

    n_classes = SEEDIV_N_CLASSES

    def _find_mat(self, subj: int, sess: int) -> Optional[str]:
        d = os.path.join(self.root, "eeg_raw_data", str(sess))
        if not os.path.isdir(d):
            return None
        hits = sorted(
            f for f in os.listdir(d)
            if f.endswith(".mat") and f.split("_")[0] == str(subj)
        )
        return os.path.join(d, hits[0]) if hits else None

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
            key=_trial_no,
        )
        seq = SEEDIV_LABEL_SEQ[sess]        # per-session, not shared
        for k in eeg_keys:
            clip = _trial_no(k)             # 1..24
            if not 1 <= clip <= SEEDIV_N_CLIPS:
                raise ValueError(f"{path}: clip index {clip} out of range 1..24")
            self._slice_trial(
                mat[k].astype(np.float32), seq[clip - 1], subj, sess, clip,
            )
            self._trial_counter += 1
