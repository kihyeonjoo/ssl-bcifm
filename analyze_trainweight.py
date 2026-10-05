"""
분석 F: 학습 창에 시간 가중을 주면 head 가 좋아지는가 (백본 고정, 실험 D 방식).

S11 분석 C 에서 감정 신호가 클립 시작 후 시간이 지나야 올라온다는 것을 쟀다.
그렇다면 학습 때 **클립 앞부분 창의 라벨은 사실상 잡음**이다 — "이 창은 그 감정" 이라고
가르치지만 그 창에는 아직 그 감정이 실려 있지 않다.

백본을 학습하지 않고 이것을 시험하는 방법이 실험 D 와 같다: `cache_ft` 의 A팔 특징을
도메인 중심화한 뒤 head 를 새로 학습하되, **학습 창의 가중만 바꾼다.**

  (a) 균등                    실험 D 재현
  (b) 앞 D초 제외             D = 20, 40, 80초 (가중 0)
  (c) 부드러운 시간 가중       w(t) = 1 − exp(−t/τ), τ = 20, 40, 80초
  (d) 시간 의존 라벨 스무딩     앞부분 창일수록 균등 분포 쪽으로 더 스무딩
                             ε(t) = ε_max·exp(−t/τ) + 0.1, τ = 20, 40, 80초

선택은 **검증 도메인의 clip macro-F1**, 테스트는 선택 후 한 번만 채점한다.
평가 창은 가중하지 않는다 — 분석 E 와 섞지 않기 위해서다.  E 의 최선 가중과 결합한
값은 따로 한 줄로 낸다.

**격자 축소**: 실험 D 는 lr 3개 x epoch 40 이었고 lr 은 3e-3 에 몰렸으며 선택 epoch
중앙값이 14(3사분위 18)였다.  여기서는 변형이 10가지라 lr 2개(3e-3, 1e-3) x epoch 20
으로 줄인다 — 실험 D 가 실제로 고른 범위를 덮는다.
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
import torch.nn as nn
from scipy import stats
from sklearn.metrics import f1_score

from analyze_centering import N_CLS, roles_for_fold, clip_index, l2, bootstrap_ci
from experiment_d_head_retrain import (make_head, domain_means, centered,
                                       clip_scores)
from analyze_evidence import decide_head, decide_proto

torch.set_num_threads(4)          # see S11: tiny tensors, 28 threads is 100x slower

LRS = (3e-3, 1e-3)
MAX_EPOCHS = 20
DROPS = [20.0, 40.0, 80.0]
TAUS = [20.0, 40.0, 80.0]
EPS_MAX = 0.5
# 분석 E 가 검증으로 고른 τ 의 최빈값 (10초 13회 / 80초 22회 중 80초)
E_TAU = 80.0


def window_time(meta, cidx):
    """창마다 그 클립 안에서의 시작 시각(초).  창은 클립 안에서 시간 순이다."""
    t = np.zeros(len(meta), np.float32)
    for k, idx in cidx.items():
        t[idx] = np.arange(len(idx)) * 4.0
    return t


def variant_weights(tpos, kind, param):
    """(창 가중, 라벨 스무딩 ε) — 둘 중 하나만 쓰는 변형도 있다."""
    if kind == "uniform":
        return np.ones_like(tpos), np.full_like(tpos, 0.1)
    if kind == "drop":
        return (tpos >= param).astype(np.float32), np.full_like(tpos, 0.1)
    if kind == "soft":
        return 1.0 - np.exp(-tpos / param), np.full_like(tpos, 0.1)
    if kind == "smooth":
        return np.ones_like(tpos), EPS_MAX * np.exp(-tpos / param) + 0.1
    raise ValueError(kind)


def weighted_loss(logits, y, w, eps):
    """샘플별 가중 + 샘플별 라벨 스무딩 교차 엔트로피."""
    logp = torch.log_softmax(logits, -1)
    onehot = torch.zeros_like(logp).scatter_(1, y[:, None], 1.0)
    soft = onehot * (1 - eps[:, None]) + eps[:, None] / N_CLS
    per = -(soft * logp).sum(1)
    return (per * w).sum() / w.sum().clamp_min(1e-8)


def train_head_w(Xtr, ytr, wtr, etr, Xva, va_keys, va_off, cidx, lab, seed,
                 device, batch=256, max_epochs=MAX_EPOCHS, lrs=LRS):
    """실험 D 의 train_head 에 샘플별 가중과 스무딩을 더한 것."""
    keep = wtr > 0                              # dropped windows cost nothing
    Xt = torch.from_numpy(Xtr[keep]).float().to(device)
    yt = torch.from_numpy(ytr[keep]).long().to(device)
    wt = torch.from_numpy(wtr[keep]).float().to(device)
    et = torch.from_numpy(etr[keep]).float().to(device)
    Xv = torch.from_numpy(Xva).float().to(device)
    best = (-1.0, None, None, None)
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
                weighted_loss(head(Xt[j]), yt[j], wt[j], et[j]).backward()
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


def weighted_prototypes(Zc, y, dom, w):
    """가중 평균 prototype — head 없이 같은 가중을 쓰는 경로."""
    P = np.zeros((N_CLS, Zc.shape[1]), np.float32)
    doms = sorted({tuple(x) for x in dom})
    for c in range(N_CLS):
        per = []
        for d in doms:
            m = (dom[:, 0] == d[0]) & (dom[:, 1] == d[1]) & (y == c)
            if not m.any() or w[m].sum() < 1e-8:
                continue
            per.append((Zc[m] * w[m, None]).sum(0) / w[m].sum())
        if per:
            P[c] = np.mean(per, axis=0)
    return P


# ── one checkpoint ──────────────────────────────────────────────────────────

VARIANTS = ([("uniform", 0.0)] + [("drop", d) for d in DROPS]
            + [("soft", t) for t in TAUS] + [("smooth", t) for t in TAUS])


def vname(kind, param):
    return kind if kind == "uniform" else f"{kind}{param:g}"


def run_one(path, device):
    ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)",
                                os.path.basename(path)).groups())
    z = np.load(path)
    X, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
    train, val, test = roles_for_fold(ts)
    cidx = clip_index(meta)
    tpos = window_time(meta, cidx)
    doms = sorted({(int(a), int(b)) for a, b in meta[:, :2]})
    mu = domain_means(X, meta, doms)

    tr = np.flatnonzero(np.isin(meta[:, 0], train))
    va = np.flatnonzero(np.isin(meta[:, 0], val))
    te = np.flatnonzero(meta[:, 0] == ts)
    Xtr, Xva, Xte = (centered(X, i, meta, mu) for i in (tr, va, te))
    va_keys = [k for k in sorted(cidx) if k[0] in val]
    te_keys = [k for k in sorted(cidx) if k[0] == ts]
    va_off = {k: np.searchsorted(va, cidx[k]) for k in va_keys}
    te_off = {k: np.searchsorted(te, cidx[k]) for k in te_keys}

    # prototype path material, normalised space
    Z = l2(X)
    Ztr = centered(Z, tr, meta, {d: Z[(meta[:, 0] == d[0]) &
                                      (meta[:, 1] == d[1])].mean(0)
                                 for d in doms})
    Zte = centered(Z, te, meta, {d: Z[(meta[:, 0] == d[0]) &
                                      (meta[:, 1] == d[1])].mean(0)
                                 for d in doms})
    dom_tr = meta[tr][:, :2]
    y_te_clip = np.array([lab[cidx[k][0]] for k in te_keys])

    out = {}
    for kind, param in VARIANTS:
        nm = vname(kind, param)
        w, eps = variant_weights(tpos[tr], kind, param)
        head, vf1, lr, ep = train_head_w(Xtr, lab[tr], w, eps, Xva, va_keys,
                                         va_off, cidx, lab, sd, device)
        with torch.no_grad():
            P = torch.softmax(head(torch.from_numpy(Xte).float().to(device)),
                              -1).cpu().numpy()
        pc, tc = clip_scores(P, te_keys, cidx, te_off, lab)
        out[f"{nm}|head_win"] = float((P.argmax(1) == lab[te]).mean())
        out[f"{nm}|head_clip"] = float((pc == tc).mean())
        # 분석 E 의 최선 가중(시간 가중)을 **결정 단계**에 더한 값.  학습 가중과
        # 결정 가중은 서로 다른 단계이므로 섞지 않고 따로 낸다.
        pc_tw = np.array([decide_head(P[te_off[k]], "time", E_TAU, 0.3)
                          for k in te_keys])
        out[f"{nm}|head_clip_Etime"] = float((pc_tw == tc).mean())
        out[f"{nm}|val_f1"] = float(vf1)
        out[f"{nm}|lr"] = float(lr)
        out[f"{nm}|ep"] = float(ep)

        # prototype path with the SAME weights (no head training)
        Pw = weighted_prototypes(Ztr, lab[tr], dom_tr, w)
        Pn = l2(Pw)
        sim = l2(Zte) @ Pn.T
        pcp = np.stack([sim[te_off[k]].mean(0) for k in te_keys]).argmax(1)
        out[f"{nm}|proto_win"] = float((sim.argmax(1) == lab[te]).mean())
        out[f"{nm}|proto_clip"] = float((pcp == y_te_clip).mean())
        Zc = l2(Zte)
        pcp_tw = np.array([decide_proto(Zte[te_off[k]], sim[te_off[k]], Pn,
                                        "time", E_TAU, 0.3) for k in te_keys])
        out[f"{nm}|proto_clip_Etime"] = float((pcp_tw == y_te_clip).mean())

        # diagnostic: does weighting shrink the train/test length mismatch?
        out[f"{nm}|Pnorm"] = float(np.linalg.norm(Pw, axis=1).mean())
    # test-side class deviation length does not depend on the weighting
    dte = np.stack([Zte[lab[te] == c].mean(0) for c in range(N_CLS)])
    out["dnorm"] = float(np.linalg.norm(dte, axis=1).mean())
    return ts, sd, out


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=f"results/{CFG.prefix('trainweight')}.npz")
    args = ap.parse_args()
    device = torch.device(args.device)

    R = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd, out = run_one(f, device)
        for k, v in out.items():
            R[k][ts].append(v)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  " + "  ".join(
            f"{vname(*v)} {out[f'{vname(*v)}|head_clip']:.3f}"
            for v in VARIANTS[:4]), flush=True)

    def arr(k):
        return np.array([np.mean(R[k][s]) for s in sorted(R[k])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    report(arr, R)


def report(arr, R):
    W = 100
    names = {"uniform": "(a) 균등 = 실험 D",
             **{f"drop{d:g}": f"(b) 앞 {d:g}초 제외" for d in DROPS},
             **{f"soft{t:g}": f"(c) 시간 가중 τ={t:g}초" for t in TAUS},
             **{f"smooth{t:g}": f"(d) 시간 라벨스무딩 τ={t:g}초" for t in TAUS}}
    for metric, mn in (("head_clip", "조건2 경로 clip"),
                       ("head_win", "조건2 경로 window"),
                       ("proto_clip", "prototype 경로 clip"),
                       ("proto_win", "prototype 경로 window"),
                       ("head_clip_Etime", "조건2 경로 clip + 분석E 시간 가중"),
                       ("proto_clip_Etime", "prototype 경로 clip + 분석E 시간 가중")):
        base = arr(f"uniform|{metric}")
        print(f"\n{'='*W}\n{mn}   기준 = (a) 균등 (실험 D 재현)\n"
              f"재학습 비교지만 **같은 특징·같은 분할**이라 짝지은 검정으로 판정한다\n{'='*W}")
        for kind, param in VARIANTS:
            nm = vname(kind, param)
            a = arr(f"{nm}|{metric}")
            if kind == "uniform":
                print(f"  {names[nm]:<28}{a.mean():.4f} ± {a.std(ddof=1):.4f}")
                continue
            d = a - base
            lo, hi = bootstrap_ci(d)
            p = stats.wilcoxon(a, base).pvalue
            print(f"  {names[nm]:<28}{a.mean():.4f}  Δ{d.mean():+.4f} "
                  f"CI[{lo:+.4f},{hi:+.4f}]"
                  f"{'  0포함' if lo <= 0 <= hi else '       '} p={p:.4f} "
                  f"{int((d > 0).sum())}/{len(d)}")

    print(f"\n{'='*W}\n선택된 lr / epoch (검증 clip macro-F1)\n{'='*W}")
    for kind, param in VARIANTS:
        nm = vname(kind, param)
        print(f"  {names[nm]:<28} lr 중앙값 {np.median(arr(f'{nm}|lr')):.0e}"
              f"  epoch 중앙값 {np.median(arr(f'{nm}|ep')):.0f}"
              f"  val_f1 {arr(f'{nm}|val_f1').mean():.4f}")

    print(f"\n{'='*W}\n진단: 학습 prototype 길이 / 테스트 클래스 편차 길이\n"
          f"(S10 에서 약 2배였고, 그 불일치가 M1 에 γ 축소를 강제했다)\n{'='*W}")
    dn = arr("dnorm")
    for kind, param in VARIANTS:
        nm = vname(kind, param)
        pn = arr(f"{nm}|Pnorm")
        r = pn / dn
        print(f"  {names[nm]:<28} ‖P‖ {pn.mean():.4f}  ‖δ‖ {dn.mean():.4f}  "
              f"비율 {r.mean():.3f}")

    print(f"\n{'='*W}\n피험자별 이득 (조건2 경로 clip, 최선 변형)\n{'='*W}")
    base = arr("uniform|head_clip")
    best = max((vname(*v) for v in VARIANTS if v[0] != "uniform"),
               key=lambda n: arr(f"{n}|head_clip").mean())
    a = arr(f"{best}|head_clip")
    d = a - base
    print(f"  최선: {names[best]}   Δ{d.mean():+.4f}")
    print("  " + "  ".join(f"S{i+1}{d[i]:+.3f}" for i in range(len(d))))
    print(f"  나빠진 {int((d < 0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
