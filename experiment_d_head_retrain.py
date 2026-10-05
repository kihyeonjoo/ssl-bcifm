"""
Experiment D — retrain the head on centered features with the backbone frozen.

The staleness problem exists only because the backbone moves: the domain mean
mu is estimated once per epoch and the features drift away from it.  Freeze the
backbone and mu is exact, recomputed from every window of the domain with the
very weights that produced them.  Staleness is zero by construction.

So this asks the question that decides whether the 37-hour end-to-end run is
worth starting: **given perfectly fresh domain means, does a head trained on
centered features beat the head that was trained uncentered?**

  new head + centering     does training on centered features help at all
  new head, no centering   control 1 — isolates "retraining the head"
  A-arm head + centering   control 2 — the published condition 2 (clip 0.718)
  prototype + centering    condition 4 (clip 0.759), no head at all

If the new centered head cannot beat 0.718, then end-to-end training has to
earn its gain from something other than "the head sees centered inputs" — and
that something would have to survive 40% staleness.  If it clears 0.759 the
case is strong.

Everything runs off cache_ft/ (raw CLS per checkpoint).  No backbone forward,
so a full sweep is minutes rather than tens of hours.

Selection discipline: epochs and learning rate are chosen on the VALIDATION
domains' clip macro-F1, centered the same way; the test subject is scored once
afterwards.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from dataset_config import get as _cfg_root
CFG = _cfg_root()

import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import f1_score

from analyze_centering import (N_CLS, roles_for_fold, load_head, head_probs,
                               clip_index, l2, clip_reduce, bootstrap_ci)

LRS = (3e-3, 1e-3, 3e-4)
MAX_EPOCHS = 40


def make_head(d=200, n_cls=N_CLS, dropout=0.0, seed=0):
    """Same shape as main_head so the comparison is about the input, not the head."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(d, n_cls))


def domain_means(X, meta, doms):
    """Exact per-domain mean over every window.  No labels, no staleness."""
    return {d: X[(meta[:, 0] == d[0]) & (meta[:, 1] == d[1])].mean(0) for d in doms}


def centered(X, idx, meta, mu):
    Y = X[idx].copy()
    for d, m in mu.items():
        sel = (meta[idx, 0] == d[0]) & (meta[idx, 1] == d[1])
        if sel.any():
            Y[sel] -= m
    return Y


def clip_scores(P, keys, cidx, offsets, lab):
    """Clip-level predictions and truth from window softmax."""
    pred = np.stack([P[offsets[k]].mean(0) for k in keys]).argmax(1)
    true = np.array([lab[cidx[k][0]] for k in keys])
    return pred, true


def train_head(Xtr, ytr, Xva, va_keys, va_off, cidx, lab, seed, device,
               batch=256, max_epochs=MAX_EPOCHS, lrs=LRS):
    """Grid over lr; within each lr the epoch is chosen by validation clip macro-F1.

    Returns the head at the best (lr, epoch) and what validation said about it.
    """
    Xt = torch.from_numpy(Xtr).float().to(device)
    yt = torch.from_numpy(ytr).long().to(device)
    Xv = torch.from_numpy(Xva).float().to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = (-1.0, None, None, None)          # val_f1, state, lr, epoch

    for lr in lrs:
        head = make_head(seed=seed).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
        g = torch.Generator(device="cpu"); g.manual_seed(seed)
        for ep in range(1, max_epochs + 1):
            head.train()
            perm = torch.randperm(len(Xt), generator=g).to(device)
            for i in range(0, len(perm), batch):
                j = perm[i:i + batch]
                opt.zero_grad()
                crit(head(Xt[j]), yt[j]).backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                P = torch.softmax(head(Xv), -1).cpu().numpy()
            pred, true = clip_scores(P, va_keys, cidx, va_off, lab)
            f1 = f1_score(true, pred, average="macro")
            if f1 > best[0]:
                best = (f1, {k: v.detach().cpu().clone()
                             for k, v in head.state_dict().items()}, lr, ep)
    head = make_head(seed=seed)
    head.load_state_dict(best[1])
    return head.eval().to(device), best[0], best[2], best[3]


def evaluate(head, X, idx, meta, mu, keys, cidx, lab, device):
    """Window and clip accuracy / macro-F1 on centered test features."""
    Xc = centered(X, idx, meta, mu)
    off = {k: np.searchsorted(idx, cidx[k]) for k in keys}
    with torch.no_grad():
        P = torch.softmax(head(torch.from_numpy(Xc).float().to(device)), -1).cpu().numpy()
    yw = lab[idx]
    pc, tc = clip_scores(P, keys, cidx, off, lab)
    return {"win": float((P.argmax(1) == yw).mean()),
            "win_f1": float(f1_score(yw, P.argmax(1), average="macro")),
            "clip": float((pc == tc).mean()),
            "clip_f1": float(f1_score(tc, pc, average="macro"))}


def run_one(path, ckpt_dir, device):
    ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(path)).groups())
    z = np.load(path)
    X, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
    train, val, _ = roles_for_fold(ts)
    cidx = clip_index(meta)
    doms = sorted({(int(r[0]), int(r[1])) for r in meta})
    tr_d = [d for d in doms if d[0] in train]
    va_d = [d for d in doms if d[0] in val]
    te_d = [d for d in doms if d[0] == ts]

    tr_keys = sorted(k for k in cidx if k[0] in train)
    va_keys = sorted(k for k in cidx if k[0] in val)
    te_keys = sorted(k for k in cidx if k[0] == ts)
    tr_i = np.concatenate([cidx[k] for k in tr_keys])
    va_i = np.concatenate([cidx[k] for k in va_keys])
    te_i = np.concatenate([cidx[k] for k in te_keys])

    mu_tr = domain_means(X, meta, tr_d)
    mu_va = domain_means(X, meta, va_d)
    mu_te = domain_means(X, meta, te_d)
    va_off = {k: np.searchsorted(va_i, cidx[k]) for k in va_keys}
    out = {}

    # ── new head on CENTERED features ───────────────────────────────────
    h, vf1, lr, ep = train_head(
        centered(X, tr_i, meta, mu_tr), lab[tr_i],
        centered(X, va_i, meta, mu_va), va_keys, va_off, cidx, lab, sd, device)
    r = evaluate(h, X, te_i, meta, mu_te, te_keys, cidx, lab, device)
    out.update({f"new_cent_{k}": v for k, v in r.items()})
    out["new_cent_lr"], out["new_cent_ep"], out["new_cent_valf1"] = lr, ep, vf1

    # ── control 1: new head, NO centering ───────────────────────────────
    zero = {d: np.zeros(X.shape[1], np.float32) for d in doms}
    h0, vf0, lr0, ep0 = train_head(
        X[tr_i], lab[tr_i], X[va_i], va_keys, va_off, cidx, lab, sd, device)
    r0 = evaluate(h0, X, te_i, meta, {d: zero[d] for d in te_d},
                  te_keys, cidx, lab, device)
    out.update({f"new_raw_{k}": v for k, v in r0.items()})
    out["new_raw_lr"], out["new_raw_ep"] = lr0, ep0

    # ── control 2: the A-arm head, centered (= published condition 2) ───
    ah = load_head(os.path.join(ckpt_dir, f"{CFG.ckpt_prefix}S{ts}_seed{sd}.pt")).to(device)
    mu_tr_global = np.mean([mu_tr[d] for d in tr_d], axis=0)
    shifted = {d: mu_te[d] - mu_tr_global for d in te_d}
    r2 = evaluate(ah, X, te_i, meta, shifted, te_keys, cidx, lab, device)
    out.update({f"arm_cent_{k}": v for k, v in r2.items()})
    # and the A-arm head with no centering at all (= condition 1)
    r1 = evaluate(ah, X, te_i, meta, {d: zero[d] for d in te_d},
                  te_keys, cidx, lab, device)
    out.update({f"arm_raw_{k}": v for k, v in r1.items()})

    # ── prototype on centered features (= published condition 4) ────────
    Z = l2(X)
    muz_tr = domain_means(Z, meta, tr_d)
    muz_te = domain_means(Z, meta, te_d)
    Etr = clip_reduce(Z, cidx, tr_keys)
    dtr = np.array([(k[0], k[1]) for k in tr_keys])
    ytr = np.array([lab[cidx[k][0]] for k in tr_keys])
    for d in tr_d:
        m = (dtr == np.asarray(d)).all(1); Etr[m] -= Etr[m].mean(0)
    P = np.stack([np.nanmean([Etr[(dtr == np.asarray(d)).all(1) & (ytr == c)].mean(0)
                              for d in tr_d], axis=0) for c in range(N_CLS)])
    Ete = np.stack([Z[cidx[k]].mean(0) - muz_te[(k[0], k[1])] for k in te_keys])
    tc = np.array([lab[cidx[k][0]] for k in te_keys])
    pc = (l2(Ete) @ l2(P).T).argmax(1)
    out["proto_clip"] = float((pc == tc).mean())
    out["proto_clip_f1"] = float(f1_score(tc, pc, average="macro"))
    Zw = centered(Z, te_i, meta, muz_te)
    pw = (l2(Zw) @ l2(P).T).argmax(1)
    out["proto_win"] = float((pw == lab[te_i]).mean())
    return ts, sd, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache_ft")
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/experiment_d.npz")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    files = sorted(glob.glob(os.path.join(args.cache_dir, "S*_seed*.npz")))
    print(f"[cache] {len(files)}개, device={device}\n", flush=True)
    R = defaultdict(lambda: defaultdict(list))
    for f in files:
        ts, sd, o = run_one(f, args.ckpt_dir, device)
        for k, v in o.items():
            R[k][ts].append(v)
        print(f"  S{ts} seed{sd}  new_cent {o['new_cent_clip']:.4f} "
              f"(lr {o['new_cent_lr']:.0e} ep {o['new_cent_ep']})  "
              f"new_raw {o['new_raw_clip']:.4f}  arm_cent {o['arm_cent_clip']:.4f}  "
              f"proto {o['proto_clip']:.4f}", flush=True)

    def arr(k):
        return np.array([np.mean(R[k][s]) for s in sorted(R[k])])

    rows = [("arm_raw", "A팔 head, 중심화 없음 (조건1)"),
            ("arm_cent", "A팔 head + 중심화 (조건2)"),
            ("proto", "prototype + 중심화 (조건4)"),
            ("new_raw", "새 head, 중심화 없음 (대조군1)"),
            ("new_cent", "새 head + 중심화 (실험 D)")]
    print(f"\n{'='*92}\n실험 D: 백본 고정, 정확한 μ (낡음 0)\n{'='*92}")
    for unit in ("clip", "win"):
        print(f"\n[{unit} 단위]  n=15 피험자, 시드 3개 평균")
        for key, nm in rows:
            k = f"{key}_{unit}" if f"{key}_{unit}" in R else f"{key}_{unit}"
            a = arr(k)
            print(f"  {nm:<34} {a.mean():.4f} ± {a.std(ddof=1):.4f}")

    print(f"\n{'='*92}\n짝지은 비교 (피험자 단위)\n{'='*92}")
    def cmp(a_key, b_key, unit, label):
        a, b = arr(f"{a_key}_{unit}"), arr(f"{b_key}_{unit}")
        d = a - b; lo, hi = bootstrap_ci(d)
        try: p = stats.wilcoxon(a, b).pvalue
        except ValueError: p = float("nan")
        print(f"  {label:<44} Δ{d.mean():+.4f}  CI[{lo:+.4f},{hi:+.4f}]"
              f"{'  0포함' if lo <= 0 <= hi else '       '}  p={p:.4f}  "
              f"{int((d > 0).sum())}/{len(d)}")
    for unit in ("clip", "win"):
        print(f"\n[{unit}]")
        cmp("new_cent", "arm_cent", unit, "(1) 새 중심화 head  vs  조건2")
        cmp("new_cent", "proto", unit, "(2) 새 중심화 head  vs  조건4")
        cmp("new_cent", "new_raw", unit, "(3) 새 중심화 head  vs  대조군1 (중심화 효과)")
        cmp("new_raw", "arm_raw", unit, "    대조군1 vs 조건1 (head 재학습 자체)")

    print(f"\n  S1 (파일럿과 같은 fold) 재현 확인")
    for key, nm in rows:
        v = np.mean(R[f"{key}_clip"][1])
        print(f"    {nm:<34} clip {v:.4f}")
    print(f"    파일럿 1 (end-to-end, n=1)          clip 0.8222 / head경로 0.8000")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k: np.array([np.mean(v[s]) for s in sorted(v)])
                          for k, v in R.items()})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
