"""
From window-level logits to clip-level decisions — offline and causal.

The model classifies each 4 s window on its own.  SEED's label is constant for
a whole ~4 min film clip, and the DE literature's 80%+ numbers come from
LDS-smoothed features that average over that whole clip.  This script asks how
much of that gap is simply "decide from more time":

  window     : the per-window accuracy the training script already reports
  clip       : average softmax over every window of the clip (non-causal —
               the decision needs the clip to have ended)
  causal-Ns  : at each window, average only the windows in the past N seconds
               and score every window.  This is what a real-time system could
               do with an N-second decision delay.

Inputs are the ``save_logits`` files written by finetune_labram_hemi_aux.py,
one per LOSO fold.

Note the unit changes: a clip-level score has 15 clips x 3 sessions = 45
decisions per subject, so it moves in steps of ~2.2% and its fold-to-fold
variance is larger.  Report window- and clip-level side by side.
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
from scipy import stats


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def balanced(pred, true, n_cls):
    """Mean per-class recall — what the EEG-FM benchmarks report as B-Acc."""
    rec = [float((pred[true == c] == c).mean()) for c in range(n_cls)
           if (true == c).any()]
    return float(np.mean(rec))


def mde_from(d, power=0.80):
    """짝지은 차이 ``d`` 로부터 최소 감지 효과.

    (t_{1-α/2,n-1} + t_{power,n-1}) · sd(d) / √n.  SEED 의 +0.0386 이 이 식에
    sd=0.0496, n=15 를 넣은 값이다 (reports/EA_ANALYSIS.md).  데이터셋마다 sd 가
    다르므로 상수로 박으면 다른 데이터셋에서 틀린 기준이 된다.
    """
    from scipy import stats
    d = np.asarray(d, dtype=float)
    n = len(d)
    df = n - 1
    return float((stats.t.ppf(0.975, df) + stats.t.ppf(power, df))
                 * d.std(ddof=1) / np.sqrt(n))


def per_fold(path, windows_per_step, delays):
    d = np.load(path)
    P = softmax(d["logits"].astype(np.float64))
    y, sess, clip = d["label"], d["session"], d["clip"]
    subj = int(d["subject"][0])

    n_cls = P.shape[1]
    out = {"window": float((P.argmax(1) == y).mean()),
           "window_b": balanced(P.argmax(1), y, n_cls)}

    # clip-level (non-causal) — one decision per clip
    cp, ct = [], []
    for key in sorted(set(zip(sess.tolist(), clip.tolist()))):
        m = (sess == key[0]) & (clip == key[1])
        cp.append(P[m].mean(0).argmax()); ct.append(y[m][0])
    cp, ct = np.array(cp), np.array(ct)
    out["clip"] = float((cp == ct).mean())
    out["clip_b"] = balanced(cp, ct, n_cls)

    # causal moving average over the last N seconds, scored per window
    for sec in delays:
        w = max(1, int(round(sec / windows_per_step)))
        pr, tr = [], []
        for key in sorted(set(zip(sess.tolist(), clip.tolist()))):
            idx = np.flatnonzero((sess == key[0]) & (clip == key[1]))  # time order
            c = np.cumsum(P[idx], axis=0)
            for t in range(len(idx)):
                lo = t - w
                avg = c[t] - (c[lo] if lo >= 0 else 0)
                pr.append(avg.argmax()); tr.append(y[idx[t]])
        pr, tr = np.array(pr), np.array(tr)
        out[f"causal-{sec}s"] = float((pr == tr).mean())
        out[f"causal-{sec}s_b"] = balanced(pr, tr, n_cls)
    return subj, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob for the per-fold .npz logit files")
    ap.add_argument("--step_sec", type=float, default=4.0,
                    help="seconds between consecutive windows (step / 200 Hz)")
    ap.add_argument("--delays", default="8,20,60",
                    help="causal look-back windows in seconds")
    ap.add_argument("--compare", default=None,
                    help="second glob to compare against (paired over subjects)")
    args = ap.parse_args()
    delays = [int(x) for x in args.delays.split(",")]

    def run(pattern):
        res = {}
        for f in sorted(glob.glob(pattern)):
            s, o = per_fold(f, args.step_sec, delays)
            res[s] = o
        return res

    A = run(args.pattern)
    if not A:
        raise SystemExit(f"no files match {args.pattern}")
    keys = list(next(iter(A.values())).keys())
    print(f"{len(A)} folds: {sorted(A)}\n")
    print(f"{'level':14s} {'acc':>8} {'± std':>8} {'B-Acc':>9} {'± std':>8}")
    for k in [x for x in keys if not x.endswith("_b")]:
        v = np.array([A[s][k] for s in sorted(A)])
        b = np.array([A[s][k + "_b"] for s in sorted(A)])
        print(f"{k:14s} {v.mean():8.4f} {v.std(ddof=1) if len(v)>1 else 0:8.4f} "
              f"{b.mean():9.4f} {b.std(ddof=1) if len(b)>1 else 0:8.4f}")

    if args.compare:
        B = run(args.compare)
        common = sorted(set(A) & set(B))
        print(f"\npaired vs {args.compare}  (n={len(common)})")
        for k in keys:
            a = np.array([A[s][k] for s in common]); b = np.array([B[s][k] for s in common])
            p = stats.ttest_rel(a, b)[1] if len(common) > 2 else float("nan")
            print(f"{k:14s} Δ{a.mean()-b.mean():+.4f}  "
                  f"better {int((a>b).sum())}/{len(common)}  p={p:.4f}")


if __name__ == "__main__":
    main()
