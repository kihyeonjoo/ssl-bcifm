"""
few-shot 라벨 곡선이 **fine-tuning 덕분인가** — 특징 대조군.

S14 는 A팔 fine-tuned CLS 로만 쟀다.  같은 절차를 다른 특징에 적용해 "교차 피험자
fine-tuning 이 적은 라벨에서의 효율을 높이는가" 를 본다.  **k=1, 2 에서의 차이가
핵심**이다 — 라벨이 많아지면 어떤 표현이든 결국 따라잡을 수 있기 때문이다.

특징
  ft_cent    fine-tuned A팔 CLS + 중심화        (S14 의 현재 설정)
  ft_raw     fine-tuned A팔 CLS, **중심화 없음**
  pre_cls    사전학습 LaBraM CLS + 중심화
  pre_pat    사전학습 LaBraM 패치 평균 + 중심화
  de         로그 밴드 파워(DE 대용) + 중심화     손설계 대조군, 248차원

프로토콜은 S14 와 같다.  방법은 (a) 라벨 없음과 (c) 규제 없는 로지스틱만.
**사전학습·DE 에는 학습 prototype 이 A팔과 다른 공간에 있으므로 각 특징에서 새로
만든다** — 그래야 (a) 기준선이 그 특징의 라벨 없는 최선이 된다.
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
from scipy import stats

from analyze_centering import N_CLS, roles_for_fold, clip_index, l2, bootstrap_ci
from analyze_fewshot import lr_fit, draw_eval_and_pool, pick_k
from analyze_fewshot_sessions import clip_windows, score, STEPS
from calib_common import clips_by_domain_label

# 이 분석의 텐서는 (수백 x 200) 으로 작다.  그 크기에서는 스레드 핸드오프가
# 계산보다 비싸다: 400 Adam 스텝이 4스레드 2.36s, 1스레드 0.44s (5.4배).
torch.set_num_threads(1)
# 라벨 상한은 analyze_centering 이 데이터셋마다 계산한 ceil_logreg 다
# (SEED 0.8464, SEED-V 0.5431).  박아 두면 분모가 틀린다.
CEILING = float(np.asarray(
    np.load(f"results/{CFG.prefix('centering_analysis')}.npz")["ceil_logreg"]
).mean())
# 감정별 클립 k 개를 쓰므로, 평가용 1편을 뺀 나머지가 상한이다
# (SEED 5-1=4, SEED-V 3-1=2).  박아 두면 SEED-V 에서 k=3,4 가 조용히
# k=2 와 같은 집합으로 떨어져 중복 수치가 결과 파일에 들어간다.
KS = list(range(1, CFG.max_k + 1))


def prototypes_for(X, meta, lab, cidx, train):
    """그 특징 공간의 학습 prototype (클립 단위, 도메인 중심화 후 클래스 평균)."""
    Z = l2(X)
    keys = sorted(k for k in cidx if k[0] in train)
    E = np.stack([Z[cidx[k]].mean(0) for k in keys])
    y = np.array([lab[cidx[k][0]] for k in keys])
    dom = np.array([(k[0], k[1]) for k in keys])
    for d in {tuple(x) for x in dom}:
        m = (dom[:, 0] == d[0]) & (dom[:, 1] == d[1])
        E[m] -= E[m].mean(0)
    return np.stack([np.nanmean(
        [E[(dom[:, 0] == d[0]) & (dom[:, 1] == d[1]) & (y == c)].mean(0)
         for d in {tuple(x) for x in dom}], axis=0) for c in range(N_CLS)])


def load_feature(name, ts, sd, cache_ft="cache_ft"):
    """(X, meta, lab, 중심화 여부).  meta/lab 은 모든 특징에서 같아야 한다."""
    if name in ("ft_cent", "ft_raw"):
        z = np.load(f"{cache_ft}/S{ts}_seed{sd}.npz")
        return z["cls"], z["meta"].astype(int), z["lab"].astype(int), \
            (name == "ft_cent")
    if name in ("pre_cls", "pre_pat"):
        z = np.load(f"{cache_ft}/PRETRAINED.npz")
        key = "cls" if name == "pre_cls" else "pat"
        return z[key], z["meta"].astype(int), z["lab"].astype(int), True
    if name == "de":
        z = np.load("cache_de.npz")
        return z["cls"], z["meta"].astype(int), z["lab"].astype(int), True
    raise ValueError(name)


FEATURES = ["ft_cent", "ft_raw", "pre_cls", "pre_pat", "de"]


def run_feature(X, meta, lab, ts, centre, n_rep, steps_grid, same_day_only,
                rng):
    """S14 프로토콜(또는 1-(i) 같은 날만)로 k 곡선을 낸다."""
    Z = l2(X)
    cidx = clip_index(meta)
    train, val, _ = roles_for_fold(ts)
    P_z = prototypes_for(X, meta, lab, cidx, train)
    doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
    by = clips_by_domain_label(cidx, lab, ts)

    # 스텝 수는 검증 피험자로 (k 별)
    steps_of = {}
    vpacks = {k: [] for k in KS}
    for v in val:
        vd = sorted({(kk[0], kk[1]) for kk in cidx if kk[0] == v})
        vby = clips_by_domain_label(cidx, lab, v)
        vr = np.random.default_rng(1000 * v + 7)
        Pv = prototypes_for(X, meta, lab, cidx, train)
        for _ in range(max(2, n_rep // 5)):
            ev, pool = draw_eval_and_pool(vby, vd, vr)
            for k in KS:
                sel = pick_k(pool, vd, k, vr)
                groups = ([{d: sel[d]} for d in vd] if same_day_only
                          else [sel])
                for one in groups:
                    ds_ = sorted(one)
                    ek = sorted(kk for d in ds_ for kk in ev[d])
                    yc = np.array([lab[cidx[kk][0]] for kk in ek])
                    mu = mus_of(Z, cidx, one, centre)
                    vpacks[k].append((ek, yc, mu, one, Pv))
    for k in KS:
        best, bv = steps_grid[0], -1.0
        for st in steps_grid:
            acc = []
            for ek, yc, mu, one, Pv in vpacks[k]:
                W, b = fit(Z, cidx, one, mu, Pv, st)
                acc.append(score(Z, cidx, ek, yc, mu, Pv, W, b)[0])
            m = float(np.mean(acc))
            if m > bv:
                bv, best = m, st
        steps_of[k] = best

    out = defaultdict(list)
    for _ in range(n_rep):
        ev, pool = draw_eval_and_pool(by, doms, rng)
        for k in KS:
            sel = pick_k(pool, doms, k, rng)
            groups = [{d: sel[d]} for d in doms] if same_day_only else [sel]
            for one in groups:
                ds_ = sorted(one)
                ek = sorted(kk for d in ds_ for kk in ev[d])
                yc = np.array([lab[cidx[kk][0]] for kk in ek])
                mu = mus_of(Z, cidx, one, centre)
                w0, c0 = score(Z, cidx, ek, yc, mu, P_z)
                W, b = fit(Z, cidx, one, mu, P_z, steps_of[k])
                w1, c1 = score(Z, cidx, ek, yc, mu, P_z, W, b)
                out[f"k{k}|a_none|win"].append(w0)
                out[f"k{k}|a_none|clip"].append(c0)
                out[f"k{k}|c_lr|win"].append(w1)
                out[f"k{k}|c_lr|clip"].append(c1)
    return {kk: float(np.mean(v)) for kk, v in out.items()}, steps_of


def mus_of(Z, cidx, sel, centre):
    """중심화를 끄면 0 벡터를 준다 — 경로를 하나로 유지한다."""
    out = {}
    for d, per in sel.items():
        idx = clip_windows(cidx, [k for c in per for k in per[c]])
        out[d] = Z[idx].mean(0) if centre else np.zeros(Z.shape[1], np.float32)
    return out


def fit(Z, cidx, sel, mu, P_z, steps):
    X, y = [], []
    for d, per in sel.items():
        for c, keys in per.items():
            for k in keys:
                w = cidx[k]
                X.append(Z[w] - mu[d])
                y.append(np.full(len(w), c))
    return lr_fit(np.concatenate(X), np.concatenate(y), P_z, 0.0, steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--scenario", default="pooled",
                    choices=["pooled", "sameday"],
                    help="pooled = S14 (세 세션 모아 학습), sameday = 1-(i)")
    ap.add_argument("--out",
                    default=f"results/{CFG.prefix('fewshot_features')}.npz")
    args = ap.parse_args()
    same_day = args.scenario == "sameday"

    R = defaultdict(lambda: defaultdict(list))
    ST = defaultdict(list)
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)",
                                    os.path.basename(f)).groups())
        line = []
        for feat in FEATURES:
            X, meta, lab, centre = load_feature(feat, ts, sd, args.cache)
            rng = np.random.default_rng(1000 * ts + sd)
            r, st = run_feature(X, meta, lab, ts, centre, args.n_rep, STEPS,
                                same_day, rng)
            for k, v in r.items():
                R[f"{feat}|{k}"][ts].append(v)
            ST[feat].append(st)
            line.append(f"{feat} {r['k1|a_none|win']:.3f}->{r['k1|c_lr|win']:.3f}")
        print(f"  [{j}/{len(files)}] S{ts} seed{sd} k=1: " + "  ".join(line),
              flush=True)

    def arr(key):
        return np.array([np.mean(R[key][s]) for s in sorted(R[key])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    report(arr, R, ST, args)


def cmp(a, b):
    d = a - b
    lo, hi = bootstrap_ci(d)
    return (f"{d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]"
            f"{' 0포함' if lo <= 0 <= hi else '      '} "
            f"p={stats.wilcoxon(a, b).pvalue:.4f} {int((d > 0).sum())}/{len(d)}")


def report(arr, R, ST, args):
    W = 104
    NM = {"ft_cent": "fine-tuned + 중심화 (S14)",
          "ft_raw": "fine-tuned, 중심화 없음",
          "pre_cls": "사전학습 CLS + 중심화",
          "pre_pat": "사전학습 패치평균 + 중심화",
          "de": "로그 밴드파워(DE) + 중심화"}
    scen = "S14 (세 세션 모아 학습)" if args.scenario == "pooled" \
        else "1-(i) 같은 날만"
    for unit in ("win", "clip"):
        print(f"\n{'='*W}\n특징별 few-shot 곡선 — {unit}   [{scen}]\n{'='*W}")
        print(f"  {'특징':<28}{'방법':<10}" + "".join(f"{'k='+str(k):>15}"
                                                    for k in KS))
        for feat in FEATURES:
            b = [arr(f"{feat}|k{k}|a_none|{unit}") for k in KS]
            print(f"  {NM[feat]:<28}{'라벨없음':<10}" + "".join(
                f"{v.mean():>15.4f}" for v in b))
            cells = []
            for k, bb in zip(KS, b):
                a = arr(f"{feat}|k{k}|c_lr|{unit}")
                d = a - bb
                p = stats.wilcoxon(a, bb).pvalue
                cells.append(f"{a.mean():.4f} {d.mean():+.4f} p{p:.3f}".rjust(15))
            print(f"  {'':<28}{'로지스틱':<10}" + "".join(cells))

    print(f"\n{'='*W}\nk=1, 2 에서 fine-tuned 가 다른 특징보다 나은가 (로지스틱, win)\n"
          f"**이것이 이 분석의 핵심** — 라벨이 많아지면 어떤 표현이든 따라잡는다\n{'='*W}")
    for k in (1, 2):
        base = arr(f"ft_cent|k{k}|c_lr|win")
        print(f"\n  [k={k}]  fine-tuned+중심화 {base.mean():.4f}")
        for feat in FEATURES[1:]:
            a = arr(f"{feat}|k{k}|c_lr|win")
            print(f"    vs {NM[feat]:<28}{a.mean():.4f}  {cmp(base, a)}")

    print(f"\n{'='*W}\n라벨 상한({CEILING:.4f}) 대비 회수율 — clip\n{'='*W}")
    print(f"  {'특징':<28}" + "".join(f"{'k='+str(k):>12}" for k in KS))
    for feat in FEATURES:
        cells = []
        for k in KS:
            b = arr(f"{feat}|k{k}|a_none|clip").mean()
            a = arr(f"{feat}|k{k}|c_lr|clip").mean()
            cells.append(f"{100*(a-b)/(CEILING-b):>11.1f}%")
        print(f"  {NM[feat]:<28}" + "".join(cells))
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
