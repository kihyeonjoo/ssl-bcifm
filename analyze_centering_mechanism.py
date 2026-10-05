"""
What does domain centering remove, and can a short calibration recover it?

S2 established that subtracting each test session's own unlabelled mean is
worth clip +0.125, and that the nearest-prototype decision rule contributes
nothing on its own.  Three questions follow.

1  **Mechanism.**  If the domain mean carries subject identity, centering is
   removing an identity axis and the gain has a clean story.  If it does not,
   centering is fixing something else and the story has to change.  Measured by
   variance decomposition, by linear probes for subject and emotion before and
   after, by whether the per-subject gain tracks how far that subject's mean
   sits from the training mean, and by whether fine-tuning creates the offset
   or merely inherits it from the pretrained backbone.

2  **Variant.**  Centering removes the first moment.  Normalising each
   dimension's spread removes the second; whitening removes the full
   covariance.  These are the feature-space analogues of diag EA and full EA,
   and diag EA beat full EA on the raw signal — whether that ordering survives
   in feature space is not obvious.

3  **Calibration.**  A 30 s prefix does worse than no centering at all.  The
   prefix of a session is one film clip, hence one emotion, so the mean is
   pulled toward that class — amount and composition are confounded in the S2
   result.  Holding the amount fixed and varying only the composition separates
   them.  A causal cumulative variant then asks what an online system could do.

Reads cache_ft/ (raw fine-tuned features, one file per checkpoint, plus
PRETRAINED.npz).  No GPU.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
# 데이터셋은 환경변수 DATASET 으로 고른다.  기본 경로를 설정에서 끌어오는
# 이유: --out 을 한 번 빠뜨리면 SEED 결과 파일을 덮어쓴다.
from dataset_config import get as _cfg_root
CFG = _cfg_root()

import torch
from scipy import stats

from analyze_centering import (N_CLS, N_SUBJ, FS, SEG, roles_for_fold,
                               load_head, head_probs, clip_index, l2,
                               clip_reduce, bootstrap_ci)


# ── 1. mechanism ────────────────────────────────────────────────────────────

def variance_decomposition(X, meta):
    """How much of the variance is between domains rather than inside them.

    Reported on clip embeddings: the unit everything else uses.  After
    centering the between-domain term is exactly zero by construction, so the
    number that carries information is the BEFORE fraction — it is the share of
    the variance that centering deletes.
    """
    cidx = clip_index(meta)
    keys = sorted(cidx)
    E = clip_reduce(X, cidx, keys)
    dom = np.array([(k[0], k[1]) for k in keys])
    uniq = np.unique(dom, axis=0)

    grand = E.mean(0)
    mus = np.stack([E[(dom == d).all(1)].mean(0) for d in uniq])
    ns = np.array([int(((dom == d).all(1)).sum()) for d in uniq])

    # per-dimension sums of squares
    ss_between = (ns[:, None] * (mus - grand) ** 2).sum(0)
    ss_total = ((E - grand) ** 2).sum(0)
    frac_dim = ss_between / np.maximum(ss_total, 1e-30)

    # how concentrated are the domain differences
    M = mus - grand
    sv = np.linalg.svd(M, compute_uv=False)
    ev = sv ** 2 / max((sv ** 2).sum(), 1e-30)
    cum = np.cumsum(ev)

    return {
        "between_frac": float(ss_between.sum() / max(ss_total.sum(), 1e-30)),
        "between_frac_dim_med": float(np.median(frac_dim)),
        "between_frac_dim_max": float(frac_dim.max()),
        "pca_top1": float(cum[0]), "pca_top3": float(cum[2]),
        "pca_top10": float(cum[9]),
        "n_dims_90pct": int(np.searchsorted(cum, 0.90) + 1),
    }


def probes(X, meta, lab, centered, seed=0, n_splits=5):
    """Subject-ID and emotion accuracy from clip embeddings, cross-validated.

    Clip level, so the 675 samples are not 37,890 correlated windows.  The
    subject probe is what tells us whether the domain mean was carrying
    identity: if centering drops it toward chance, the mean WAS the identity.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    cidx = clip_index(meta)
    keys = sorted(cidx)
    E = clip_reduce(l2(X), cidx, keys)
    dom = np.array([(k[0], k[1]) for k in keys])
    subj = dom[:, 0]
    y = np.array([lab[cidx[k][0]] for k in keys])
    if centered:
        for d in np.unique(dom, axis=0):
            m = (dom == d).all(1)
            E[m] -= E[m].mean(0)

    out = {}
    for name, target in (("subj", subj), ("emo", y)):
        accs = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(E, target):
            sc = StandardScaler().fit(E[tr])
            clf = LogisticRegression(max_iter=1500, random_state=seed)
            clf.fit(sc.transform(E[tr]), target[tr])
            accs.append(float((clf.predict(sc.transform(E[te])) == target[te]).mean()))
        out[name] = float(np.mean(accs))
    return out


def domain_distance(X, meta, test_subj):
    """Distance from the test subject's domain means to the training mean.

    If centering works by cancelling an offset, subjects whose offset is larger
    should gain more.  Both a plain Euclidean distance and a cosine angle are
    reported because the two disagree when norms differ.
    """
    train, _, _ = roles_for_fold(test_subj)
    cidx = clip_index(meta)
    keys = sorted(cidx)
    E = clip_reduce(X, cidx, keys)
    dom = np.array([(k[0], k[1]) for k in keys])
    tr_doms = np.unique(dom[np.isin(dom[:, 0], train)], axis=0)
    te_doms = np.unique(dom[dom[:, 0] == test_subj], axis=0)
    mu_tr = np.stack([E[(dom == d).all(1)].mean(0) for d in tr_doms]).mean(0)
    mu_te = np.stack([E[(dom == d).all(1)].mean(0) for d in te_doms])
    d = mu_te - mu_tr
    cos = (mu_te @ mu_tr) / np.maximum(
        np.linalg.norm(mu_te, axis=1) * np.linalg.norm(mu_tr), 1e-12)
    return {"dist_l2": float(np.linalg.norm(d, axis=1).mean()),
            "dist_cos": float(np.mean(1 - cos))}


# ── 2. variants: first moment, second moment, full covariance ───────────────

def _inv_sqrt_shrunk(S, alpha=0.1, eps=1e-8):
    """Sigma^(-1/2) with Ledoit-Wolf-style shrinkage toward a scaled identity.

    A 200x200 covariance from ~840 windows is estimable but poorly conditioned;
    without shrinkage the smallest eigenvalues are noise and inverting them
    amplifies it.  ``alpha`` mixes in (tr S / d) I.
    """
    d = S.shape[0]
    S = 0.5 * (S + S.T)
    S = (1 - alpha) * S + alpha * (np.trace(S) / d) * np.eye(d)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, eps * max(float(w.max()), 1e-12))
    return (V * w ** -0.5) @ V.T


def domain_transform(X, meta, doms, variant, alpha=0.1, calib_mask=None):
    """Per-domain (mu, T) so that  X' = (X - mu) @ T.T  removes the chosen moments.

    variant 'a' centre only        T = I
            'b' centre + per-dim scale   T = diag(1/sigma)
            'c' centre + whiten          T = Sigma^(-1/2)
    Estimated at WINDOW level — a 200-dim covariance cannot be estimated from a
    domain's clips.
    """
    out = {}
    for d in doms:
        m = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1])
        if calib_mask is not None:
            m = m & calib_mask
        Xd = X[m]
        mu = Xd.mean(0)
        if variant == "a":
            T = None
        elif variant == "b":
            sd = Xd.std(0)
            T = np.diag(1.0 / np.maximum(sd, 1e-6 * max(sd.max(), 1e-12)))
        elif variant == "c":
            T = _inv_sqrt_shrunk(np.cov(Xd - mu, rowvar=False), alpha)
        else:
            raise ValueError(variant)
        out[d] = (mu, T)
    return out


def _apply(X, idx, meta, tf):
    Y = X[idx].copy()
    for d, (mu, T) in tf.items():
        m = (meta[idx, 0] == d[0]) & (meta[idx, 1] == d[1])
        if not m.any():
            continue
        Z = Y[m] - mu
        Y[m] = Z if T is None else Z @ T.T
    return Y


def variants(cls, meta, lab, head, test_subj, alpha=0.1):
    """Conditions 2 and 4 under variants a/b/c, clip and window.

    Everything is done at window level so the three variants differ only in
    which moments they remove.  Variant 'a' here is the window-level twin of
    S2's condition 4; it should land close to it.
    """
    train, _, _ = roles_for_fold(test_subj)
    cidx = clip_index(meta)
    doms = sorted({(int(r[0]), int(r[1])) for r in meta})
    tr_doms = [d for d in doms if d[0] in train]
    te_doms = [d for d in doms if d[0] == test_subj]

    te_keys = sorted(k for k in cidx if k[0] == test_subj)
    tr_keys = sorted(k for k in cidx if k[0] in train)
    te_w = np.concatenate([cidx[k] for k in te_keys])
    tr_w = np.concatenate([cidx[k] for k in tr_keys])
    y_w = lab[te_w]
    y_c = np.array([lab[cidx[k][0]] for k in te_keys])
    y_trc = np.array([lab[cidx[k][0]] for k in tr_keys])
    dom_trc = np.array([(k[0], k[1]) for k in tr_keys])
    local = {k: np.searchsorted(te_w, cidx[k]) for k in te_keys}

    out = {}
    for v in ("a", "b", "c"):
        tf_te = domain_transform(cls, meta, te_doms, v, alpha)
        tf_tr = domain_transform(cls, meta, tr_doms, v, alpha)
        Ate = _apply(cls, te_w, meta, tf_te)
        Atr = _apply(cls, tr_w, meta, tf_tr)

        # head path — put the transformed features back where the head expects
        mu_tr_global = Atr.mean(0)
        H = Ate + mu_tr_global
        P = head_probs(head, H)
        out[f"{v}_c2_win"] = float((P.argmax(1) == y_w).mean())
        out[f"{v}_c2_clip"] = float((np.stack(
            [P[local[k]].mean(0) for k in te_keys]).argmax(1) == y_c).mean())

        # prototype path
        Zt = l2(Ate); Zr = l2(Atr)
        Ec = np.stack([Zt[local[k]].mean(0) for k in te_keys])
        off = 0
        Etr = []
        for k in tr_keys:
            n = len(cidx[k]); Etr.append(Zr[off:off + n].mean(0)); off += n
        Etr = np.stack(Etr)
        P = np.stack([np.nanmean(
            [Etr[(dom_trc == np.asarray(d)).all(1) & (y_trc == c)].mean(0)
             for d in tr_doms], axis=0) for c in range(N_CLS)])
        out[f"{v}_c4_clip"] = float(((l2(Ec) @ l2(P).T).argmax(1) == y_c).mean())
        out[f"{v}_c4_win"] = float(((l2(Zt) @ l2(P).T).argmax(1) == y_w).mean())
    return out


# ── 3. calibration: amount versus composition ───────────────────────────────

def calib_masks(meta, te_doms, sec, kind, rng=None, n_clips=None):
    """Which windows may form the centring mean, for a given budget and layout.

    kind 'prefix'   the first N seconds of the session — what a real session
                    actually starts with, and one film clip, hence one emotion
         'random'   N seconds drawn from anywhere in the recording; an upper
                    bound no deployed system can reach, since it needs the
                    whole session to exist first
         'perclip'  N/n_clips seconds from the start of each clip; not deployable
                    either, but it holds the AMOUNT fixed while making the
                    composition balanced, which is the comparison that
                    separates amount from composition
    """
    _nc = CFG.n_clips_per_session if n_clips is None else n_clips
    n_tot = int(round(sec * FS / SEG))
    keep = np.zeros(len(meta), bool)
    for d in te_doms:
        idx = np.flatnonzero((meta[:, 0] == d[0]) & (meta[:, 1] == d[1]))
        if kind == "prefix":
            keep[idx[:n_tot]] = True
        elif kind == "random":
            take = rng.choice(len(idx), size=min(n_tot, len(idx)), replace=False)
            keep[idx[take]] = True
        elif kind == "perclip":
            per = max(1, n_tot // _nc)
            for c in range(1, _nc + 1):
                ci = np.flatnonzero((meta[:, 0] == d[0]) & (meta[:, 1] == d[1])
                                    & (meta[:, 2] == c))
                keep[ci[:per]] = True
        else:
            raise ValueError(kind)
    return keep


def calibration_study(cls, meta, lab, head, test_subj,
                      seconds=(30, 60, 120, 300), n_rep=20, hold_sec=300):
    """Conditions 2 and 4 with the centring mean built from a limited budget.

    The evaluation set is fixed to everything after ``hold_sec`` for every
    budget and layout, so a shorter calibration is not also scored on more (and
    later, easier) material.
    """
    doms = sorted({(int(r[0]), int(r[1])) for r in meta})
    te_doms = [d for d in doms if d[0] == test_subj]
    tr_doms = [d for d in doms if d[0] in roles_for_fold(test_subj)[0]]

    n_hold = int(round(hold_sec * FS / SEG))
    eval_mask = np.zeros(len(meta), bool)
    for d in te_doms:
        idx = np.flatnonzero((meta[:, 0] == d[0]) & (meta[:, 1] == d[1]))
        eval_mask[idx[n_hold:]] = True

    out = {}
    rng = np.random.default_rng(0)
    for sec in seconds:
        for kind in ("prefix", "random", "perclip"):
            reps = n_rep if kind == "random" else 1
            acc2, acc4 = [], []
            for _ in range(reps):
                keep = calib_masks(meta, te_doms, sec, kind, rng)
                r = _score_with_mask(cls, meta, lab, head, test_subj,
                                     keep, eval_mask, te_doms, tr_doms)
                acc2.append(r[2]); acc4.append(r[4])
            out[f"{kind}{sec}_c2"] = float(np.mean(acc2))
            out[f"{kind}{sec}_c4"] = float(np.mean(acc4))
    # full-session reference on the same evaluation set
    r = _score_with_mask(cls, meta, lab, head, test_subj,
                         np.ones(len(meta), bool), eval_mask, te_doms, tr_doms)
    out["full_c2"], out["full_c4"] = r[2], r[4]
    # no centering at all, same evaluation set
    r0 = _score_with_mask(cls, meta, lab, head, test_subj, None,
                          eval_mask, te_doms, tr_doms)
    out["none_c2"], out["none_c4"] = r0[2], r0[4]
    return out


def _score_with_mask(cls, meta, lab, head, test_subj, calib_mask, eval_mask,
                     te_doms, tr_doms):
    """Clip-level conditions 2 and 4.  ``calib_mask=None`` means no centring."""
    cidx = clip_index(meta)
    te_keys = [k for k in sorted(cidx) if k[0] == test_subj
               and eval_mask[cidx[k]].mean() >= 0.5]
    if not te_keys:
        return {2: float("nan"), 4: float("nan")}
    tr_keys = sorted(k for k in cidx if k[0] in {d[0] for d in tr_doms})
    y_c = np.array([lab[cidx[k][0]] for k in te_keys])

    Z = l2(cls)
    mu_raw, mu_z = {}, {}
    for d in te_doms:
        if calib_mask is None:
            mu_raw[d] = np.zeros(cls.shape[1]); mu_z[d] = np.zeros(cls.shape[1])
            continue
        m = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1]) & calib_mask
        if not m.any():
            return {2: float("nan"), 4: float("nan")}
        mu_raw[d] = cls[m].mean(0); mu_z[d] = Z[m].mean(0)

    tr_w = np.concatenate([cidx[k] for k in tr_keys])
    mu_tr_global = np.mean(
        [cls[(meta[:, 0] == d[0]) & (meta[:, 1] == d[1])].mean(0) for d in tr_doms],
        axis=0) if calib_mask is not None else 0.0

    probs = []
    for k in te_keys:
        w = np.array([i for i in cidx[k] if eval_mask[i]])
        probs.append(head_probs(head, cls[w] - mu_raw[(k[0], k[1])] + mu_tr_global).mean(0))
    c2 = float((np.stack(probs).argmax(1) == y_c).mean())

    A = np.stack([Z[[i for i in cidx[k] if eval_mask[i]]].mean(0) - mu_z[(k[0], k[1])]
                  for k in te_keys])
    B = clip_reduce(Z, cidx, tr_keys)
    y_tr = np.array([lab[cidx[k][0]] for k in tr_keys])
    db = np.array([(k[0], k[1]) for k in tr_keys])
    if calib_mask is not None:
        for d in tr_doms:
            m = (db == np.asarray(d)).all(1); B[m] -= B[m].mean(0)
    P = np.stack([np.nanmean([B[(db == np.asarray(d)).all(1) & (y_tr == c)].mean(0)
                              for d in tr_doms], axis=0) for c in range(N_CLS)])
    c4 = float(((l2(A) @ l2(P).T).argmax(1) == y_c).mean())
    return {2: c2, 4: c4}


def causal_curve(cls, meta, lab, head, test_subj, grid=(1, 2, 5, 10, 20, 40, 80, 160, 320)):
    """Centre each window on the running mean of everything before it.

    This is the only variant a live system could actually run: at window t the
    mean uses windows 1..t of that session and nothing later.  Accuracy is
    reported against elapsed windows so the warm-up cost is visible.
    """
    cidx = clip_index(meta)
    doms = sorted({(int(r[0]), int(r[1])) for r in meta})
    te_doms = [d for d in doms if d[0] == test_subj]
    tr_doms = [d for d in doms if d[0] in roles_for_fold(test_subj)[0]]
    tr_keys = sorted(k for k in cidx if k[0] in {d[0] for d in tr_doms})
    Z = l2(cls)
    B = clip_reduce(Z, cidx, tr_keys)
    y_tr = np.array([lab[cidx[k][0]] for k in tr_keys])
    db = np.array([(k[0], k[1]) for k in tr_keys])
    for d in tr_doms:
        m = (db == np.asarray(d)).all(1); B[m] -= B[m].mean(0)
    P = np.stack([np.nanmean([B[(db == np.asarray(d)).all(1) & (y_tr == c)].mean(0)
                              for d in tr_doms], axis=0) for c in range(N_CLS)])
    mu_tr_global = np.mean(
        [cls[(meta[:, 0] == d[0]) & (meta[:, 1] == d[1])].mean(0) for d in tr_doms], axis=0)

    hits2 = defaultdict(list); hits4 = defaultdict(list)
    for d in te_doms:
        idx = np.flatnonzero((meta[:, 0] == d[0]) & (meta[:, 1] == d[1]))
        run_raw = np.cumsum(cls[idx], 0) / np.arange(1, len(idx) + 1)[:, None]
        run_z = np.cumsum(Z[idx], 0) / np.arange(1, len(idx) + 1)[:, None]
        y = lab[idx]
        for g in grid:
            sel = np.arange(g - 1, len(idx))
            if not len(sel):
                continue
            H = cls[idx[sel]] - run_raw[sel] + mu_tr_global
            hits2[g].append(float((head_probs(head, H).argmax(1) == y[sel]).mean()))
            A = Z[idx[sel]] - run_z[sel]
            hits4[g].append(float(((l2(A) @ l2(P).T).argmax(1) == y[sel]).mean()))
    return ({f"causal{g}_c2": float(np.mean(v)) for g, v in hits2.items()}
            | {f"causal{g}_c4": float(np.mean(v)) for g, v in hits4.items()})


# ── driver ──────────────────────────────────────────────────────────────────

def paired_row(a, b, name):
    d = a - b
    lo, hi = bootstrap_ci(d)
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    print(f"  {name:<40} {a.mean():.4f}  Δ{d.mean():+.4f}  "
          f"CI[{lo:+.4f},{hi:+.4f}]{'  0포함' if lo <= 0 <= hi else '       '}  "
          f"p={p:.4f}  {int((d > 0).sum())}/{len(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prev", default=f"results/{CFG.prefix('centering_analysis')}.npz")
    ap.add_argument("--alpha", type=float, default=0.1, help="whitening shrinkage")
    ap.add_argument("--skip_calib", action="store_true")
    ap.add_argument("--out", default=f"results/{CFG.prefix('centering_mechanism')}.npz")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.cache_dir, "S*_seed*.npz")))
    print(f"[cache] fine-tuned {len(files)}개", flush=True)

    R = defaultdict(lambda: defaultdict(list))

    # ── pretrained reference (fold-independent) ─────────────────────────
    pre_path = os.path.join(args.cache_dir, "PRETRAINED.npz")
    pre = None
    if os.path.exists(pre_path):
        z = np.load(pre_path)
        pre = (z["cls"], z["meta"].astype(int), z["lab"].astype(int))
        print("[cache] 사전학습 raw 있음", flush=True)

    for f in files:
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
        z = np.load(f)
        cls, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
        head = load_head(os.path.join(args.ckpt_dir, f"{CFG.ckpt_prefix}S{ts}_seed{sd}.pt"))

        for k, v in variance_decomposition(cls, meta).items():
            R[f"var_{k}"][ts].append(v)
        for pre_c, tag in ((False, "raw"), (True, "cent")):
            for k, v in probes(cls, meta, lab, pre_c).items():
                R[f"probe_{tag}_{k}"][ts].append(v)
        for k, v in domain_distance(cls, meta, ts).items():
            R[k][ts].append(v)
        for k, v in variants(cls, meta, lab, head, ts, args.alpha).items():
            R[k][ts].append(v)
        if not args.skip_calib:
            for k, v in calibration_study(cls, meta, lab, head, ts).items():
                R[k][ts].append(v)
            for k, v in causal_curve(cls, meta, lab, head, ts).items():
                R[k][ts].append(v)
        print(f"  S{ts} seed{sd}  between={R['var_between_frac'][ts][-1]:.3f}  "
              f"subj={R['probe_raw_subj'][ts][-1]:.3f}->{R['probe_cent_subj'][ts][-1]:.3f}",
              flush=True)

    def arr(k):
        return np.array([np.mean(R[k][s]) for s in sorted(R[k])])

    # ── 1. mechanism ────────────────────────────────────────────────────
    print(f"\n{'='*96}\n1. 기전: 중심화가 제거하는 것\n{'='*96}")
    print(f"  clip 임베딩 전체 분산 중 '도메인 평균 간' 비율   "
          f"{arr('var_between_frac').mean():.4f}")
    print(f"    차원별 중앙값 {arr('var_between_frac_dim_med').mean():.4f}  "
          f"최대 {arr('var_between_frac_dim_max').mean():.4f}")
    print(f"  도메인 평균들의 PCA 누적 설명력  "
          f"1축 {arr('var_pca_top1').mean():.3f}  3축 {arr('var_pca_top3').mean():.3f}  "
          f"10축 {arr('var_pca_top10').mean():.3f}   90% 도달 {arr('var_n_dims_90pct').mean():.0f}축")
    print(f"\n  선형 프로브 (클립 단위, 5-fold CV)")
    print(f"    {'':<14} {'중심화 전':>10} {'중심화 후':>10} {'우연':>8}")
    print(f"    {f'피험자 ({CFG.n_subjects})':<14} {arr('probe_raw_subj').mean():10.4f} "
          f"{arr('probe_cent_subj').mean():10.4f} {1/N_SUBJ:8.4f}")
    print(f"    {'감정 (3)':<14} {arr('probe_raw_emo').mean():10.4f} "
          f"{arr('probe_cent_emo').mean():10.4f} {1/N_CLS:8.4f}")

    if pre is not None:
        pv = variance_decomposition(pre[0], pre[1])
        pp_raw = probes(pre[0], pre[1], pre[2], False)
        pp_cen = probes(pre[0], pre[1], pre[2], True)
        print(f"\n  사전학습 모델 (fine-tuning 없음) — 같은 계산")
        print(f"    도메인 간 분산 비율 {pv['between_frac']:.4f}   "
              f"(fine-tuned {arr('var_between_frac').mean():.4f})")
        print(f"    PCA 1축 {pv['pca_top1']:.3f}  3축 {pv['pca_top3']:.3f}  "
              f"90% 도달 {pv['n_dims_90pct']}축")
        print(f"    피험자 프로브 {pp_raw['subj']:.4f} -> {pp_cen['subj']:.4f}   "
              f"감정 {pp_raw['emo']:.4f} -> {pp_cen['emo']:.4f}")

    if os.path.exists(args.prev):
        z = np.load(args.prev)
        gain = z["c4_clip"] - z["c1_clip"]
        if len(gain) != len(arr("dist_l2")):
            print(f"\n  [건너뜀] 이전 결과 {len(gain)}명 vs 지금 "
                  f"{len(arr('dist_l2'))}명 — 상관은 전체 실행에서만")
            gain = None
        for k, lbl in ([] if gain is None else
                       (("dist_l2", "L2 거리"), ("dist_cos", "코사인 거리"))):
            r, p = stats.spearmanr(arr(k), gain)
            print(f"\n  이득(조건4−조건1) vs 테스트·학습 도메인 평균 {lbl}: "
                  f"Spearman ρ={r:+.3f}  p={p:.4f}  (n={len(gain)})")

    # ── 2. variants ─────────────────────────────────────────────────────
    print(f"\n{'='*96}\n2. 변형: 평균만 뺄 것인가 (shrinkage α={args.alpha})\n{'='*96}")
    prev = np.load(args.prev) if os.path.exists(args.prev) else None
    for unit in ("clip", "win"):
        print(f"\n[{unit} 단위]{'':<32}{'평균':>8}  {'Δ vs 조건1':>11}  "
              f"{'95% CI':>22}  {'p':>8}  이긴수")
        base = (prev[f"c1_{unit}"] if prev is not None
                and len(prev[f"c1_{unit}"]) == len(arr(f"a_c2_{unit}")) else None)
        for path, pname in (("c2", "head"), ("c4", "prototype")):
            for v, vname in (("a", "중심화만"), ("b", "중심화+차원별 std"),
                             ("c", "중심화+공분산 화이트닝")):
                a = arr(f"{v}_{path}_{unit}")
                if base is None:
                    print(f"  {pname}: {vname:<28} {a.mean():.4f}")
                else:
                    paired_row(a, base, f"{pname}: {vname}")

    # ── 3. calibration ──────────────────────────────────────────────────
    if not args.skip_calib:
        print(f"\n{'='*96}\n3. 캘리브레이션: 양인가 구성인가 "
              f"(평가는 300초 이후 고정)\n{'='*96}")
        for path in ("c2", "c4"):
            nm = "head(조건2)" if path == "c2" else "prototype(조건4)"
            print(f"\n[{nm}]  중심화 없음 {arr(f'none_{path}').mean():.4f}   "
                  f"세션 전체 {arr(f'full_{path}').mean():.4f}")
            print(f"  {'초':>6} {'연속 앞부분':>12} {'무작위':>12} {'클립마다 조금':>14}")
            for sec in (30, 60, 120, 300):
                row = [arr(f"{k}{sec}_{path}").mean() for k in
                       ("prefix", "random", "perclip")]
                print(f"  {sec:>6} {row[0]:12.4f} {row[1]:12.4f} {row[2]:14.4f}")

        print(f"\n{'='*96}\n3-b. 인과적 누적 중심화 (실시간 가능한 유일한 형태)\n{'='*96}")
        print(f"  {'경과 창':>8} {'경과 초':>8} {'head(조건2)':>13} {'prototype(조건4)':>17}")
        for g in (1, 2, 5, 10, 20, 40, 80, 160, 320):
            k2, k4 = f"causal{g}_c2", f"causal{g}_c4"
            if k2 not in R:
                continue
            print(f"  {g:>8} {g*SEG/FS:>8.0f} {arr(k2).mean():13.4f} {arr(k4).mean():17.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k: np.array([np.mean(v[s]) for s in sorted(v)])
                          for k, v in R.items()})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
