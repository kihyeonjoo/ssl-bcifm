"""
Euclidean Alignment (EA) for cross-subject EEG transfer.

He & Wu (2020), "Transfer Learning for Brain-Computer Interfaces:
A Euclidean Space Data Alignment Approach", IEEE TBME 67(2).

Per group ``g`` (one subject, or one recording session):

    R̄_g  = mean_i ( X_i X_iᵀ / T )         mean spatial covariance
    X'_i  = α · R̄_g^(-1/2) X_i             whiten, then restore a global scale

Whitening drives every group's mean covariance to the identity, so both the
spatial correlation structure and the overall amplitude — each dominated by
electrode impedance, placement and skull geometry rather than by the task —
become comparable across subjects.

No labels are involved, so the transform applies to the held-out test subject
too.  That makes EA **transductive**: it needs the test subject's (unlabelled)
EEG before inference, i.e. a calibration recording at deployment time.  Any
result obtained with EA has to say so.

The scalar ``α`` is applied identically to every group, so it reintroduces no
subject-specific scale.  It exists only to put the whitened signal back into
the amplitude range the pretrained backbone expects.

Robustness
----------
SEED's ``Preprocessed_EEG`` is band-passed but not artifact-rejected: single
channels reach ~1000x the median amplitude on blinks and movement.  A plain
mean covariance would let a handful of those segments define R̄ and hence the
whitening transform for the whole subject.  ``trim`` drops the highest-power
segments before averaging.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np


# ── primitives ──────────────────────────────────────────────────────────────

def segment_power(X: np.ndarray) -> float:
    """tr(X Xᵀ)/T — mean power per channel, used for artifact trimming."""
    return float(np.square(np.asarray(X, dtype=np.float64)).sum() / X.shape[1])


def inverse_sqrt(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Symmetric inverse square root R^(-1/2), with an eigenvalue floor.

    ``eps`` is relative to the largest eigenvalue, so it adapts to the data's
    scale.  Without a floor, near-silent channels (a disconnected electrode)
    produce enormous gains and the whitened signal is pure amplified noise.
    """
    R = np.asarray(R, dtype=np.float64)
    R = 0.5 * (R + R.T)                       # enforce exact symmetry
    w, V = np.linalg.eigh(R)
    floor = eps * max(float(w.max()), 1e-12)
    w = np.maximum(w, floor)
    return (V * w ** -0.5) @ V.T


def mean_covariance(
    segments: Sequence[np.ndarray],
    trim: float = 0.05,
) -> np.ndarray:
    """Trimmed mean of per-segment spatial covariances.

    Parameters
    ----------
    segments : sequence of (C, T) arrays
    trim     : fraction of highest-power segments to drop (0 = plain mean)
    """
    if not len(segments):
        raise ValueError("mean_covariance: no segments")

    keep = range(len(segments))
    if trim > 0.0 and len(segments) > 20:
        power = np.array([segment_power(X) for X in segments])
        cutoff = np.quantile(power, 1.0 - trim)
        keep = np.flatnonzero(power <= cutoff)
        if keep.size < 10:                    # degenerate — fall back
            keep = range(len(segments))

    C = segments[0].shape[0]
    R = np.zeros((C, C), dtype=np.float64)
    n = 0
    for i in keep:
        X = np.asarray(segments[i], dtype=np.float64)
        R += (X @ X.T) / X.shape[1]
        n += 1
    return R / n


# ── aligner ─────────────────────────────────────────────────────────────────

class EuclideanAligner:
    """Per-group whitening transforms, fitted without labels.

    Usage
    -----
        aligner = EuclideanAligner(trim=0.05, eps=1e-6, scale=0.125)
        aligner.fit(groups)          # {group_key: [segment, ...]}
        Xw = aligner.apply(key, X)   # (C, T) -> (C, T)
    """

    def __init__(
        self,
        trim: float = 0.05,
        eps: float = 1e-6,
        scale: float = 1.0,
    ) -> None:
        self.trim = trim
        self.eps = eps
        self.scale = scale
        self.transforms: Dict[Hashable, np.ndarray] = {}

    def fit(self, groups: Dict[Hashable, List[np.ndarray]]) -> "EuclideanAligner":
        for key, segs in groups.items():
            R = mean_covariance(segs, trim=self.trim)
            self.transforms[key] = (
                self.scale * inverse_sqrt(R, eps=self.eps)
            ).astype(np.float32)
        return self

    def apply(self, key: Hashable, X: np.ndarray) -> np.ndarray:
        W = self.transforms.get(key)
        if W is None:
            raise KeyError(f"EuclideanAligner: no transform fitted for group {key!r}")
        return W @ X

    # ── diagnostics ─────────────────────────────────────────────────────────
    def report(self, groups: Dict[Hashable, List[np.ndarray]]) -> List[Tuple]:
        """Per group: ‖R̄_after − I‖_F and the median channel std after EA.

        ``‖R̄ − I‖_F`` near 0 means the group was whitened as intended; the
        median std is what has to land in the backbone's input range.
        """
        rows = []
        for key, segs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            W = self.transforms[key]
            # Sample evenly across the group: the first N segments are the
            # first few film clips only, and EEG covariance drifts over a
            # recording, so a prefix understates how well the fit whitens.
            take = np.linspace(0, len(segs) - 1, min(len(segs), 300)).astype(int)
            after = [W @ np.asarray(segs[i], dtype=np.float32) for i in take]
            R = mean_covariance(after, trim=self.trim)
            I = np.eye(R.shape[0])
            dev = float(np.linalg.norm(R / self.scale ** 2 - I, "fro"))
            med = float(np.median([np.std(X, axis=-1) for X in after]))
            rows.append((key, dev, med))
        return rows
