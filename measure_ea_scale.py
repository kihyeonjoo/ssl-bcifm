"""
What amplitude does the backbone actually see, and is diag EA doing its job?

Two questions that the code comments assert but nothing measures:

  (a) ``ea_scale: 0.2`` is documented as matching "the median channel std of
      the scale100 pipeline (0.131)".  0.2 is not 0.131, and no script in the
      repository produces 0.131, so the claim is unverified in both directions:
      the reference value and the match.

  (b) ``EuclideanAligner.report`` scores whitening with ‖R̄/α² − I‖_F.  For
      mode='diag' that is the wrong target — diag only drives the DIAGONAL to
      1 and deliberately leaves the off-diagonal alone, so a large Frobenius
      deviation there is the intended behaviour, not a failure.  The diagonal
      check is ``diag(R̄_after)/α² ≈ 1``.

This prints, per (subject, session):
    scale100 median channel std        — the reference the backbone was tuned on
    EA median channel std              — what it gets instead, per alpha
    mean |diag(R̄_after)/α² − 1|        — did diag EA whiten the diagonal
    ‖offdiag(R̄_after)/α²‖_F            — what diag EA left behind (full EA removes it)

Read-only: builds datasets and computes statistics, trains nothing.
"""

from __future__ import annotations

import argparse

import numpy as np

from data.alignment import mean_covariance
from data.seed_raw_dataset import SEEDRawDataset
from data.seedv_raw_dataset import SEEDVRawDataset

_LOADERS = {"SEEDRawDataset": SEEDRawDataset, "SEEDVRawDataset": SEEDVRawDataset}

from dataset_config import get as _cfg_root
ROOT = _cfg_root().root


def _sample(ds, n=300):
    """Indices spread evenly over the recording, not a prefix.

    The first N segments are the first few film clips only and EEG covariance
    drifts across a session, so a prefix misstates the fit.

    BUG NOTE (fixed 2026-09-23): this used to return the POSITION inside each
    group (0..len-1) while the caller used the value as a GLOBAL index into
    ``ds._segments``.  Only the first group's positions coincide with its
    global indices, so every later group was scored by applying its own
    whitening matrix to the FIRST subject's data.  That is what produced the
    "diag EA does not whiten" numbers (476-3235) in the stage-0 report.
    """
    idx = {}
    for i, key in enumerate(ds._seg_group):
        idx.setdefault(key, []).append(i)
    # map positions back to global segment indices
    return ({k: [v[t] for t in np.linspace(0, len(v) - 1,
                                           min(len(v), n)).astype(int)]
             for k, v in idx.items()}, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3")
    ap.add_argument("--alphas", default="0.2,0.131")
    ap.add_argument("--mode", default="diag", choices=["diag", "full"])
    ap.add_argument("--n_sample", type=int, default=300)
    ap.add_argument("--dataset", default=None,
                    help="seed | seedv.  생략하면 환경변수 DATASET 또는 seed")
    args = ap.parse_args()
    if args.dataset:
        from dataset_config import set_dataset
        set_dataset(args.dataset)
    cfg = _cfg_root()
    DS = _LOADERS[cfg.loader]
    print(f"[데이터셋] {cfg.name}  root={cfg.root}  클래스 {cfg.n_classes}  "
          f"피험자 {cfg.n_subjects}")
    subjects = [int(x) for x in args.subjects.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    common = dict(root=cfg.root, subjects=subjects,
                  sessions=cfg.sessions,
                  segment_length=800, step=800, patch_size=200)

    # ── (a) the reference: what scale100 actually delivers ────────────────
    print("(a) scale100 파이프라인 (ea=False, eeg/100) 의 채널 std")
    ref = DS(norm="scale100", ea=False, **common)
    per_group = {}
    for i in range(len(ref)):
        subj, sess, _ = ref._seg_meta[i]
        per_group.setdefault((subj, sess), []).append(i)
    ref_med = {}
    for key, ids in sorted(per_group.items()):
        take = np.linspace(0, len(ids) - 1, min(len(ids), args.n_sample)).astype(int)
        stds = [ref[ids[t]]["eeg"].reshape(62, -1).std(-1).numpy() for t in take]
        ref_med[key] = float(np.median(np.concatenate(stds)))
        print(f"    S{key[0]} sess{key[1]}   median std = {ref_med[key]:.4f}")
    allmed = float(np.median(list(ref_med.values())))
    print(f"    -> 전체 중앙값 {allmed:.4f}   (주석의 기준값 0.131과 비교)\n")
    del ref

    # ── (b)(c) each alpha ─────────────────────────────────────────────────
    for a in alphas:
        print(f"(b,c) ea_mode={args.mode}  ea_scale={a}")
        ds = DS(norm="none", ea=True, ea_mode=args.mode,
                            ea_scope="session", ea_scale=a, **common)
        samp, _ = _sample(ds, args.n_sample)
        print(f"    {'group':>12} {'med std':>9} {'mean|diag/a^2-1|':>18} "
              f"{'||offdiag||_F/a^2':>19}")
        for key in sorted(samp):
            W = ds._aligner.transforms[key]
            segs = [W @ np.asarray(ds._segments[i][0], dtype=np.float32)
                    for i in samp[key]]
            R = mean_covariance(segs, trim=0.05) / (a ** 2)
            d = np.diag(R)
            off = R - np.diag(d)
            med = float(np.median([np.std(X, axis=-1) for X in segs]))
            print(f"    S{key[0]:>2} sess{key[1]}     {med:9.4f} "
                  f"{float(np.mean(np.abs(d - 1.0))):18.4f} "
                  f"{float(np.linalg.norm(off, 'fro')):19.2f}")
        print()
        del ds


if __name__ == "__main__":
    main()
