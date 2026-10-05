"""
Where does the diagnostic's clip 0.755 come from, and is it real?

The stage-1 diagnostic classified held-out subjects at clip 0.755 while the
A arm's own classifier head reached clip 0.634 on the same models.  The
diagnostic used two things the head did not: it re-centred the test domain on
its own unlabelled mean, and it classified by cosine distance to class
prototypes instead of by the trained head.  Either could be doing the work, or
neither — the gap might be an artefact of comparing different decision rules.

Four conditions separate them.  All use the same 45 checkpoints and the same
segments, so the only thing that changes is the decision rule:

    1  head,      no centering   -> must reproduce the A arm (pipeline check)
    2  head,      centered       -> centring alone, head kept
    3  prototype, no centering   -> decision rule alone, no centring
    4  prototype, centered       -> both (this is diagnostic metric 1)

Centring uses only the test session's own unlabelled features, the same
transductive assumption Euclidean Alignment already makes, so conditions 2 and
4 are deployable — but only with a calibration recording.  The robustness
section asks how short that recording can be and what happens when it is not
class-balanced.

Ordering conventions, kept identical to the diagnostic so condition 4
reproduces metric 1 exactly:
  prototype path  L2-normalise windows -> clip mean -> centre clip embeddings
  head path       raw windows -> centre windows -> head -> softmax -> clip mean

Reads the cache written by cache_ft_features.py; no GPU needed.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
# 데이터셋은 환경변수 DATASET 으로 고른다.  기본 출력 경로를 설정에서
# 끌어오는 이유: --out 을 한 번 빠뜨리면 SEED 결과 파일을 덮어쓴다.
from dataset_config import get as _cfg_root
CFG = _cfg_root()

import torch
import torch.nn as nn
from scipy import stats

# 데이터셋별 값은 dataset_config 에서 읽는다.  이 파일들이 임포트되는 시점에
# 한 번 읽으므로, 데이터셋을 바꾸려면 **임포트 전에** set_dataset 을 부른다
# (스크립트 맨 위 또는 환경변수 DATASET).
from dataset_config import get as _cfg

N_CLS = _cfg().n_classes
N_SUBJ = _cfg().n_subjects
N_VAL = _cfg().n_val_subjects
FS, SEG = _cfg().fs, _cfg().seg


# ── roles ───────────────────────────────────────────────────────────────────

def roles_for_fold(test_subj):
    subs = list(range(1, N_SUBJ + 1))
    i = subs.index(test_subj)
    val = [subs[(i + k) % len(subs)] for k in range(1, N_VAL + 1)]
    train = [s for s in subs if s != test_subj and s not in val]
    return train, val, [test_subj]


# ── head ────────────────────────────────────────────────────────────────────

def load_head(ckpt_path, d=200, n_cls=N_CLS):
    """main_head only.  lambda_asym is 0 in the A arm, so the reported
    prediction is this head alone — asym_head is multiplied by zero."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                         nn.Dropout(0.0), nn.Linear(d, n_cls))
    own = {k[len("main_head."):]: v for k, v in sd.items()
           if k.startswith("main_head.")}
    missing, unexpected = head.load_state_dict(own, strict=False)
    if missing:
        raise ValueError(f"{ckpt_path}: main_head keys missing {missing}")
    return head.eval()


@torch.no_grad()
def head_probs(head, X):
    return torch.softmax(head(torch.from_numpy(np.ascontiguousarray(X)).float()),
                         -1).numpy()


# ── aggregation ─────────────────────────────────────────────────────────────

def clip_index(meta):
    """{(subj, sess, clip): [window indices in time order]}"""
    d = defaultdict(list)
    for i, r in enumerate(meta):
        d[(int(r[0]), int(r[1]), int(r[2]))].append(i)
    return d


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def clip_reduce(V, cidx, keys):
    return np.stack([V[cidx[k]].mean(0) for k in keys])


def score(pred_w, pred_c, y_w, y_c):
    return (float((pred_w == y_w).mean()), float((pred_c == y_c).mean()))


# ── centring ────────────────────────────────────────────────────────────────

def domain_means(X, meta, domains, idx_filter=None):
    """Mean window feature per (subject, session).  ``idx_filter`` restricts
    which windows may contribute — used for the calibration-length and
    class-imbalance simulations."""
    out = {}
    for d in domains:
        m = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1])
        if idx_filter is not None:
            m = m & idx_filter
        if m.any():
            out[d] = X[m].mean(0)
    return out


def all_domains(meta):
    return sorted({(int(r[0]), int(r[1])) for r in meta})


# ── the four conditions ─────────────────────────────────────────────────────

def four_conditions(cls, meta, lab, head, test_subj, calib_mask=None):
    """-> {cond: (window_acc, clip_acc)} for one checkpoint.

    ``calib_mask`` restricts which windows may contribute to the TEST domains'
    centring mean (robustness simulations); by default the whole session does.
    It applies to conditions 2 and 4 alike — the raw mean the head path
    subtracts and the normalised clip mean the prototype path subtracts are
    both computed from the same windows, so the two conditions are always fed
    the same calibration material.
    """
    train, val, _ = roles_for_fold(test_subj)
    doms = all_domains(meta)
    tr_doms = [d for d in doms if d[0] in train]
    te_doms = [d for d in doms if d[0] == test_subj]

    cidx = clip_index(meta)
    te_keys = sorted(k for k in cidx if k[0] == test_subj)
    tr_keys = sorted(k for k in cidx if k[0] in train)
    te_w = np.concatenate([cidx[k] for k in te_keys])
    y_w = lab[te_w]
    y_c = np.array([lab[cidx[k][0]] for k in te_keys])

    mu = domain_means(cls, meta, doms)
    mu_tr_global = np.mean([mu[d] for d in tr_doms], axis=0)
    if calib_mask is None:
        calib_mask = np.ones(len(meta), bool)
    # raw mean for the head path, normalised-window mean for the prototype path
    Zall = l2(cls)
    mu_test, mu_test_z = {}, {}
    for d in te_doms:
        m = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1]) & calib_mask
        if not m.any():
            m = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1])
        mu_test[d] = cls[m].mean(0)
        mu_test_z[d] = Zall[m].mean(0)

    out = {}

    # ── 1. head, no centering ───────────────────────────────────────────
    P = head_probs(head, cls[te_w])
    pw = P.argmax(1)
    pc = clip_reduce(P, {k: np.searchsorted(te_w, cidx[k]) for k in te_keys},
                     te_keys).argmax(1)
    out[1] = score(pw, pc, y_w, y_c)

    # ── 2. head, test domain centered then shifted to the train location ──
    adj = cls[te_w].copy()
    for d in te_doms:
        m = (meta[te_w, 0] == d[0]) & (meta[te_w, 1] == d[1])
        adj[m] = adj[m] - mu_test[d] + mu_tr_global
    P = head_probs(head, adj)
    pw = P.argmax(1)
    pc = clip_reduce(P, {k: np.searchsorted(te_w, cidx[k]) for k in te_keys},
                     te_keys).argmax(1)
    out[2] = score(pw, pc, y_w, y_c)

    # ── 3/4. prototype, without / with centering ────────────────────────
    Z = l2(cls)                                   # normalise windows
    Ec_te = clip_reduce(Z, cidx, te_keys)          # clip embeddings
    Ec_tr = clip_reduce(Z, cidx, tr_keys)
    y_tr = np.array([lab[cidx[k][0]] for k in tr_keys])
    dom_te = np.array([(k[0], k[1]) for k in te_keys])
    dom_tr = np.array([(k[0], k[1]) for k in tr_keys])

    for cond, centred in ((3, False), (4, True)):
        A, B = Ec_te.copy(), Ec_tr.copy()
        if centred:
            for d in te_doms:
                m = (dom_te[:, 0] == d[0]) & (dom_te[:, 1] == d[1])
                A[m] -= mu_test_z[d]
            for d in tr_doms:
                m = (dom_tr[:, 0] == d[0]) & (dom_tr[:, 1] == d[1])
                B[m] -= B[m].mean(0)
        P = np.stack([np.nanmean(
            [B[(dom_tr[:, 0] == d[0]) & (dom_tr[:, 1] == d[1]) & (y_tr == k)].mean(0)
             for d in tr_doms], axis=0) for k in range(N_CLS)])
        pc = (l2(A) @ l2(P).T).argmax(1)
        # window-level: same prototypes, per-window normalised feature
        Aw = l2(cls[te_w]).copy()
        if centred:
            for d in te_doms:
                m = (meta[te_w, 0] == d[0]) & (meta[te_w, 1] == d[1])
                Aw[m] -= mu_test_z[d]
        pw = (l2(Aw) @ l2(P).T).argmax(1)
        out[cond] = score(pw, pc, y_w, y_c)

    return out


# ── 보완 1: fairer ceilings ─────────────────────────────────────────────────

def ceilings(cls, meta, lab, test_subj):
    """Three upper bounds on the test subject, all leave-one-clip-out.

    The stage-1 ceiling built each prototype from the 4 remaining clips of one
    (subject, session) — 4 clips against roughly 180 for the cross-subject
    prototypes, which stacks the comparison against the ceiling and is how a
    NEGATIVE gap became possible.  These pool the subject's three sessions
    (each centred on its own mean first, so the session offset is still
    removed) and give the ceiling 14 clips, or a fitted classifier.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cidx = clip_index(meta)
    keys = sorted(k for k in cidx if k[0] == test_subj)
    Z = l2(cls)
    E = clip_reduce(Z, cidx, keys)
    y = np.array([lab[cidx[k][0]] for k in keys])
    sess = np.array([k[1] for k in keys])

    # centre each session on its own mean, then pool the subject's 45 clips
    C = E.copy()
    for s in np.unique(sess):
        C[sess == s] -= C[sess == s].mean(0)

    # (0) stage-1 ceiling: LOCO inside one session only
    hit = n = 0
    for s in np.unique(sess):
        idx = np.flatnonzero(sess == s)
        for i in idx:
            P = np.full((N_CLS, C.shape[1]), np.nan)
            for k in range(N_CLS):
                o = idx[(y[idx] == k) & (idx != i)]
                if len(o):
                    P[k] = C[o].mean(0)
            ok = ~np.isnan(P).any(1)
            if ok.sum() < 2:
                continue
            hit += int(np.flatnonzero(ok)[(l2(C[i:i+1]) @ l2(P[ok]).T).argmax(1)[0]] == y[i])
            n += 1
    m_session = hit / max(n, 1)

    # (a) subject-level LOCO, nearest prototype from the other 44 clips
    hit = 0
    for i in range(len(C)):
        P = np.stack([C[(y == k) & (np.arange(len(C)) != i)].mean(0)
                      for k in range(N_CLS)])
        hit += int((l2(C[i:i+1]) @ l2(P).T).argmax(1)[0] == y[i])
    m_subject = hit / len(C)

    # (b) same split, logistic regression instead of nearest prototype
    hit = 0
    for i in range(len(C)):
        tr = np.arange(len(C)) != i
        sc = StandardScaler().fit(C[tr])
        clf = LogisticRegression(max_iter=2000, random_state=0)
        clf.fit(sc.transform(C[tr]), y[tr])
        hit += int(clf.predict(sc.transform(C[i:i+1]))[0] == y[i])
    m_logreg = hit / len(C)

    return {"ceil_session": m_session, "ceil_subject": m_subject,
            "ceil_logreg": m_logreg}


# ── 보완 2: validation domains as a third reference ─────────────────────────

def cross_for(cls, meta, lab, target_subjects, source_subjects):
    """Centered nearest-prototype accuracy of one subject group against another.
    Clip level, same construction as diagnostic metric 1."""
    cidx = clip_index(meta)
    Z = l2(cls)
    tk = sorted(k for k in cidx if k[0] in target_subjects)
    sk = sorted(k for k in cidx if k[0] in source_subjects)
    if not tk or not sk:
        return float("nan")
    A = clip_reduce(Z, cidx, tk); ya = np.array([lab[cidx[k][0]] for k in tk])
    B = clip_reduce(Z, cidx, sk); yb = np.array([lab[cidx[k][0]] for k in sk])
    da = np.array([(k[0], k[1]) for k in tk]); db = np.array([(k[0], k[1]) for k in sk])
    for d in {tuple(x) for x in da}:
        m = (da[:, 0] == d[0]) & (da[:, 1] == d[1]); A[m] -= A[m].mean(0)
    for d in {tuple(x) for x in db}:
        m = (db[:, 0] == d[0]) & (db[:, 1] == d[1]); B[m] -= B[m].mean(0)
    P = np.stack([np.nanmean([B[(db[:, 0] == d[0]) & (db[:, 1] == d[1]) & (yb == k)].mean(0)
                              for d in {tuple(x) for x in db}], axis=0)
                  for k in range(N_CLS)])
    return float(((l2(A) @ l2(P).T).argmax(1) == ya).mean())


def role_comparison(cls, meta, lab, test_subj):
    """Train / val / test domains scored the same way.

    Validation subjects never receive a gradient but DO select the epoch, so
    they sit between "trained on" and "never seen".  If val looks like train,
    the 0.96 is about gradients; if it looks like test, it is about the epoch
    choice too.
    """
    train, val, _ = roles_for_fold(test_subj)
    out = {}
    # train domain scored against the other training subjects
    acc = [cross_for(cls, meta, lab, [s], [t for t in train if t != s]) for s in train]
    out["role_train"] = float(np.nanmean(acc))
    out["role_val"] = float(np.nanmean(
        [cross_for(cls, meta, lab, [s], train) for s in val]))
    out["role_test"] = cross_for(cls, meta, lab, [test_subj], train)
    return out


# ── 강건성 1: an unbalanced calibration recording ───────────────────────────

DROP_TO = max(1, CFG.clips_per_class - 3)   # SEED 5->2, SEED-V 3->1


def imbalance_sim(cls, meta, lab, head, test_subj, drop_to=DROP_TO):
    """Recompute the centring mean from a class-imbalanced subset.

    Centring is label-free, but it is not class-blind: the mean of a recording
    that happens to contain mostly one emotion is pulled toward that class.  A
    real calibration session has no guarantee of balance, so this drops one
    class from 5 clips to ``drop_to`` per session before taking the mean, and
    scores on ALL clips as usual.
    """
    doms = [d for d in all_domains(meta) if d[0] == test_subj]
    rng = np.random.default_rng(0)
    out = {}
    for k in range(N_CLS):
        keep = np.ones(len(meta), bool)
        for d in doms:
            for c in range(1, CFG.n_clips_per_session + 1):
                m = ((meta[:, 0] == d[0]) & (meta[:, 1] == d[1])
                     & (meta[:, 2] == c))
                if not m.any() or lab[np.flatnonzero(m)[0]] != k:
                    continue
                keep[m] = False          # provisionally drop every clip of class k
        # put back `drop_to` clips of class k per domain
        for d in doms:
            clips = sorted({int(meta[i, 2]) for i in np.flatnonzero(
                (meta[:, 0] == d[0]) & (meta[:, 1] == d[1])) if lab[i] == k})
            for c in rng.permutation(clips)[:drop_to]:
                keep[(meta[:, 0] == d[0]) & (meta[:, 1] == d[1]) & (meta[:, 2] == c)] = True
        r = four_conditions(cls, meta, lab, head, test_subj, calib_mask=keep)
        out[f"imb{k}_c2"] = r[2][1]
        out[f"imb{k}_c4"] = r[4][1]
    return out


# ── 강건성 2: how short can the calibration be ──────────────────────────────

def calibration_sim(cls, meta, lab, head, test_subj, seconds=(30, 60, 300)):
    """Centre on the first N seconds of each test session, score on what
    follows.

    EA already needs a calibration recording; centring can share it.  The
    evaluation set is fixed to everything after the longest N so the lengths
    are compared on identical data — otherwise a shorter calibration would also
    be scored on more (and easier, later) material.
    """
    doms = [d for d in all_domains(meta) if d[0] == test_subj]
    per = {}
    for d in doms:
        idx = np.flatnonzero((meta[:, 0] == d[0]) & (meta[:, 1] == d[1]))
        per[d] = idx                       # already in time order
    n_hold = int(round(max(seconds) * FS / SEG))
    eval_mask = np.zeros(len(meta), bool)
    for d, idx in per.items():
        eval_mask[idx[n_hold:]] = True

    out = {}
    for sec in list(seconds) + ["full"]:
        keep = np.zeros(len(meta), bool)
        for d, idx in per.items():
            if sec == "full":
                keep[idx] = True
            else:
                keep[idx[:int(round(sec * FS / SEG))]] = True
        if any(not (keep & (meta[:, 0] == d[0]) & (meta[:, 1] == d[1])).any()
               for d in doms):
            continue
        r = _conditions_on_subset(cls, meta, lab, head, test_subj, keep, eval_mask)
        out[f"cal{sec}_c2"] = r[2]
        out[f"cal{sec}_c4"] = r[4]
    return out


def _conditions_on_subset(cls, meta, lab, head, test_subj, calib_mask, eval_mask):
    """Conditions 2 and 4 scored on a restricted set of clips (clip level).

    A clip counts only if at least half its windows survive the mask, so a clip
    straddling the calibration boundary is not judged from a handful of frames.
    """
    train, _, _ = roles_for_fold(test_subj)
    doms = all_domains(meta)
    tr_doms = [d for d in doms if d[0] in train]
    te_doms = [d for d in doms if d[0] == test_subj]
    cidx = clip_index(meta)
    te_keys = [k for k in sorted(cidx) if k[0] == test_subj
               and eval_mask[cidx[k]].mean() >= 0.5]
    if not te_keys:
        return {2: float("nan"), 4: float("nan")}
    tr_keys = sorted(k for k in cidx if k[0] in train)
    y_c = np.array([lab[cidx[k][0]] for k in te_keys])
    mu_all = domain_means(cls, meta, doms)
    mu_tr_global = np.mean([mu_all[d] for d in tr_doms], axis=0)
    Zall = l2(cls)
    mu_test, mu_z = {}, {}
    for d in te_doms:
        m = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1]) & calib_mask
        mu_test[d] = cls[m].mean(0)
        mu_z[d] = Zall[m].mean(0)

    # condition 2 — head on shifted features
    probs = []
    for k in te_keys:
        w = np.array([i for i in cidx[k] if eval_mask[i]])
        d = (k[0], k[1])
        probs.append(head_probs(head, cls[w] - mu_test[d] + mu_tr_global).mean(0))
    c2 = float((np.stack(probs).argmax(1) == y_c).mean())

    # condition 4 — prototypes, centred with the same restricted mean
    Z = Zall
    A = np.stack([Z[[i for i in cidx[k] if eval_mask[i]]].mean(0) for k in te_keys])
    for i, k in enumerate(te_keys):
        A[i] -= mu_z[(k[0], k[1])]
    B = clip_reduce(Z, cidx, tr_keys)
    y_tr = np.array([lab[cidx[k][0]] for k in tr_keys])
    db = np.array([(k[0], k[1]) for k in tr_keys])
    for d in tr_doms:
        m = (db[:, 0] == d[0]) & (db[:, 1] == d[1]); B[m] -= B[m].mean(0)
    P = np.stack([np.nanmean([B[(db[:, 0] == d[0]) & (db[:, 1] == d[1]) & (y_tr == kk)].mean(0)
                              for d in tr_doms], axis=0) for kk in range(N_CLS)])
    c4 = float(((l2(A) @ l2(P).T).argmax(1) == y_c).mean())
    return {2: c2, 4: c4}


# ── paired comparison ───────────────────────────────────────────────────────

def bootstrap_ci(d, n=20000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    b = d[idx].mean(1)
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def paired(a, b, name, ref="조건1"):
    d = a - b
    lo, hi = bootstrap_ci(d)
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w = float("nan")
    print(f"  {name:<34} {a.mean():.4f}  Δ{d.mean():+.4f}  "
          f"CI[{lo:+.4f},{hi:+.4f}]{'  0포함' if lo <= 0 <= hi else '       '}  "
          f"p={w:.4f}  {int((d > 0).sum())}/{len(d)}")
    return d.mean(), (lo, hi)


# ── driver ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--arm_csv", default=CFG.arm_csv)
    ap.add_argument("--skip_robust", action="store_true")
    ap.add_argument("--out",
                    default=f"results/{CFG.prefix('centering_analysis')}.npz")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "S*_seed*.npz")))
    if not files:
        raise SystemExit(f"no cache in {args.cache_dir} — run cache_ft_features.py")
    print(f"[cache] {len(files)}개\n", flush=True)

    R = defaultdict(lambda: defaultdict(list))     # metric -> subj -> [seeds]
    for f in files:
        m = re.search(r"S(\d+)_seed(\d+)", os.path.basename(f))
        ts, sd = int(m.group(1)), int(m.group(2))
        z = np.load(f)
        cls, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
        head = load_head(os.path.join(args.ckpt_dir, f"{CFG.ckpt_prefix}S{ts}_seed{sd}.pt"))

        c = four_conditions(cls, meta, lab, head, ts)
        for k, (w, cl) in c.items():
            R[f"c{k}_win"][ts].append(w)
            R[f"c{k}_clip"][ts].append(cl)
        for k, v in ceilings(cls, meta, lab, ts).items():
            R[k][ts].append(v)
        for k, v in role_comparison(cls, meta, lab, ts).items():
            R[k][ts].append(v)
        if not args.skip_robust:
            for k, v in imbalance_sim(cls, meta, lab, head, ts).items():
                R[k][ts].append(v)
            for k, v in calibration_sim(cls, meta, lab, head, ts).items():
                R[k][ts].append(v)
        print(f"  S{ts} seed{sd}  c1={c[1][1]:.4f} c2={c[2][1]:.4f} "
              f"c3={c[3][1]:.4f} c4={c[4][1]:.4f}", flush=True)

    def arr(key):
        subs = sorted(R[key])
        return np.array([np.mean(R[key][s]) for s in subs]), subs

    # ── condition table ─────────────────────────────────────────────────
    print(f"\n{'='*96}\n0.755 대 0.634 의 출처 분리  (피험자별 시드 평균, n={CFG.n_subjects})\n{'='*96}")
    names = {1: "1 head, 중심화 없음 (A팔 재현)", 2: "2 head, 중심화 있음",
             3: "3 prototype, 중심화 없음", 4: "4 prototype, 중심화 (지표1)"}
    for unit in ("clip", "win"):
        print(f"\n[{unit} 단위]   {'평균':>8}  {'Δ vs 조건1':>12}  {'부트스트랩 95% CI':>24}"
              f"  {'Wilcoxon':>10}  이긴수")
        base, subs = arr(f"c1_{unit}")
        for k in (1, 2, 3, 4):
            a, _ = arr(f"c{k}_{unit}")
            if k == 1:
                print(f"  {names[k]:<34} {a.mean():.4f}  {'기준':>12}")
            else:
                paired(a, base, names[k])
    # 조건 1~4 는 같은 체크포인트에서 집계만 바꾸는 **결정 규칙 비교**다.
    # 재학습 비교용 최소 감지 효과(SEED +0.0386)는 여기 적용되지 않는다 —
    # 판정은 위의 짝지은 Wilcoxon 과 부트스트랩 CI 로 한다.
    print(f"\n  판정: 짝지은 Wilcoxon + 부트스트랩 CI (결정 규칙 비교이므로\n  재학습용 최소 감지 효과는 쓰지 않는다).  실행 잡음은 A팔 시드 sd 를 보라.")

    # ── ceilings ────────────────────────────────────────────────────────
    print(f"\n{'='*96}\n보완 1: 공정한 상한 (clip 단위)\n{'='*96}")
    c1, _ = arr("c4_clip")
    for key, lab_ in (("ceil_session", "기존 지표2 (세션 내 LOCO, 클래스당 4클립)"),
                      ("ceil_subject", "(a) 피험자 단위 LOCO (클래스당 14클립)"),
                      ("ceil_logreg",  "(b) 피험자 단위 LOCO + 로지스틱 회귀")):
        a, _ = arr(key)
        print(f"  {lab_:<44} {a.mean():.4f} ± {a.std(ddof=1):.4f}   "
              f"격차(상한−교차) {a.mean()-c1.mean():+.4f}")
    print(f"  {'교차 (조건4)':<44} {c1.mean():.4f} ± {c1.std(ddof=1):.4f}")
    neg = {}
    for key in ("ceil_session", "ceil_subject", "ceil_logreg"):
        a, subs = arr(key)
        neg[key] = int(((a - c1) < 0).sum())
    print(f"\n  음수 격차 피험자 수:  기존 {neg['ceil_session']}/{N_SUBJ}   "
          f"(a) {neg['ceil_subject']}/{N_SUBJ}   (b) {neg['ceil_logreg']}/{N_SUBJ}")

    # ── roles ───────────────────────────────────────────────────────────
    print(f"\n{'='*96}\n보완 2: 학습 / 검증 / 테스트 도메인 (clip, 중심화 최근접)\n{'='*96}")
    for key, lab_ in (("role_train", "학습 도메인 (기울기 받음)"),
                      ("role_val",   "검증 도메인 (기울기 없음, epoch 선택에만 사용)"),
                      ("role_test",  "테스트 도메인 (완전 미사용)")):
        a, _ = arr(key)
        print(f"  {lab_:<44} {a.mean():.4f} ± {a.std(ddof=1):.4f}")

    # ── robustness ──────────────────────────────────────────────────────
    if not args.skip_robust:
        _dt = DROP_TO
        print(f"\n{'='*96}\n강건성 1: 중심화 평균이 클래스 불균형일 때 "
              f"({CFG.clips_per_class}클립 -> {_dt}클립)\n{'='*96}")
        b2, _ = arr("c2_clip"); b4, _ = arr("c4_clip")
        print(f"  {'':<22} {'조건2':>18} {'조건4':>18}")
        print(f"  {'균형 (기준)':<22} {b2.mean():>10.4f}{'':>8} {b4.mean():>10.4f}")
        for k, nm in enumerate(CFG.class_names):
            a2, _ = arr(f"imb{k}_c2"); a4, _ = arr(f"imb{k}_c4")
            print(f"  {nm+f' {CFG.clips_per_class}->{_dt}':<22} "
                  f"{a2.mean():>10.4f} ({a2.mean()-b2.mean():+.4f})"
                  f" {a4.mean():>10.4f} ({a4.mean()-b4.mean():+.4f})")

        print(f"\n{'='*96}\n강건성 2: 짧은 캘리브레이션 (평가 구간은 300초 이후로 고정)\n{'='*96}")
        print(f"  {'중심화 근거':<22} {'조건2':>18} {'조건4':>18}")
        for sec in (30, 60, 300, "full"):
            k2, k4 = f"cal{sec}_c2", f"cal{sec}_c4"
            if k2 not in R:
                continue
            a2, _ = arr(k2); a4, _ = arr(k4)
            nm = "세션 전체" if sec == "full" else f"앞 {sec}초"
            print(f"  {nm:<22} {a2.mean():>10.4f}{'':>8} {a4.mean():>10.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k: np.array([np.mean(v[s]) for s in sorted(v)])
                          for k, v in R.items()})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
