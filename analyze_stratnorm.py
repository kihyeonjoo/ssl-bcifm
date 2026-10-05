"""실험 1: stratified normalization 기준선 — 현실적 캘리브레이션 예산에서.

리뷰어의 첫 질문은 "이건 Fdez et al. (2021) 의 stratified normalization 이다" 일
것이다.  그 방법은 (참가자, 세션)마다 특징을 평균 0, 분산 1 로 맞추고 SEED 에서
3클래스 LOSO 79.6% 를 보고했는데, **테스트 피험자의 통계를 무엇으로 계산했는지
명시하지 않는다** (세션 15 trial 전체로 보인다 — transductive).

여기서는 중심화와 stratified normalization 을 **같은 코드 경로·같은 클립·같은
창·같은 난수**로 비교한다.  두 변형의 차이는 σ 로 나누느냐 하나뿐이다.

    none    ẑ = z
    center  ẑ = z − μ_g                 (본 연구)
    strat   ẑ = (z − μ_g) / σ_g         (Fdez 2021 의 특징 단계 판, 차원별)

μ_g, σ_g 는 도메인 g = (피험자, 세션) 마다:
    학습 도메인  그 도메인의 모든 창
    테스트 도메인  (a) 캘리브레이션 창만 (현실적: 감정별 T 초)
                  (b) 평가 창 전체 (transductive — Fdez 가 한 것으로 보이는 쪽)

z 는 L2 정규화한 CLS 임베딩 (기존 prototype 경로와 같다).  클립 결정은 정규화된
창들의 평균과 prototype 의 코사인, 창 결정은 창 하나와 prototype 의 코사인.

주의 — 학습 쪽 Fdez 원 방법은 **은닉층마다, 학습 중에** 정규화한다.  여기는
FM 마지막 임베딩에서 테스트 시점에만 한다.  학습 중 정규화는 S5/S6 에서 "학습 중
중심화" 로 따로 쟀고 조건4 를 넘지 못했다 (3/15).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
from dataset_config import get as _cfg_root
from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               all_domains, bootstrap_ci)
from calib_common import pick_from_clip

CFG = _cfg_root()
SEC_PER_WIN = CFG.seg / CFG.fs
EPS = 1e-6          # σ 바닥 — 창이 몇 개뿐이면 어떤 차원은 분산이 0 에 가깝다


def stats_of(Z, idx):
    X = Z[idx]
    mu = X.mean(0)
    sd = X.std(0, ddof=1) if len(idx) > 1 else np.ones(Z.shape[1])
    return mu, np.maximum(sd, EPS * max(float(sd.max()), 1e-12))


def transform(Z, idx, mu, sd, kind):
    if kind == "none":
        return Z[idx]
    if kind == "center":
        return Z[idx] - mu
    return (Z[idx] - mu) / sd


def prototypes(Z, meta, lab, cidx, train, kind):
    """학습 도메인을 각자의 통계로 변환한 뒤, 클래스별 클립 임베딩 평균.
    도메인마다 먼저 평균 내고 도메인 간 평균 — 기존 경로와 같은 가중."""
    doms = [d for d in all_domains(meta) if d[0] in train]
    per_dom = defaultdict(list)
    for d in doms:
        widx = np.flatnonzero((meta[:, 0] == d[0]) & (meta[:, 1] == d[1]))
        mu, sd = stats_of(Z, widx)
        for k in sorted(cidx):
            if (k[0], k[1]) != d:
                continue
            emb = transform(Z, np.asarray(cidx[k]), mu, sd, kind).mean(0)
            per_dom[(d, int(lab[cidx[k][0]]))].append(emb)
    P = np.stack([np.nanmean([np.mean(per_dom[(d, c)], axis=0)
                              for d in doms if per_dom[(d, c)]], axis=0)
                  for c in range(N_CLS)])
    return l2(P)


def score(Z, cidx, eval_keys, y_c, stats_by_dom, P, kind):
    pw, pc, yw = [], [], []
    for i, k in enumerate(eval_keys):
        d = (k[0], k[1])
        mu, sd = stats_by_dom[d] if stats_by_dom else (0.0, 1.0)
        W = transform(Z, np.asarray(cidx[k]), mu, sd, kind)
        pw.append((l2(W) @ P.T).argmax(1))
        pc.append(int((l2(W.mean(0)[None]) @ P.T).argmax(1)[0]))
        yw.append(np.full(len(cidx[k]), int(y_c[i])))
    pw, yw, pc = np.concatenate(pw), np.concatenate(yw), np.array(pc)

    def bal(p, t):
        return float(np.mean([(p[t == c] == c).mean() for c in range(N_CLS)
                              if (t == c).any()]))
    return {"win": float((pw == yw).mean()), "clip": float((pc == y_c).mean()),
            "win_bal": bal(pw, yw), "clip_bal": bal(pc, y_c)}


def run_one(cls, meta, lab, ts, seconds, n_rep, rng):
    train, _, _ = roles_for_fold(ts)
    cidx = clip_index(meta)
    Z = l2(cls)
    te_doms = [d for d in all_domains(meta) if d[0] == ts]
    P = {k: prototypes(Z, meta, lab, cidx, train, k)
         for k in ("none", "center", "strat")}

    by = defaultdict(list)
    for k in sorted(cidx):
        if k[0] == ts:
            by[((k[0], k[1]), int(lab[cidx[k][0]]))].append(k)

    acc = defaultdict(list)
    for _ in range(n_rep):
        # analyze_calib_protocol.run_one 과 **같은 순서로** 난수를 소비한다 —
        # 그래야 같은 시드에서 같은 클립이 캘리브레이션으로 뽑힌다.
        held = {d: [by[(d, c)][rng.integers(len(by[(d, c)]))]
                    for c in range(N_CLS)] for d in te_doms}
        held_set = {k for v in held.values() for k in v}
        eval_keys = [k for k in sorted(cidx) if k[0] == ts and k not in held_set]
        y_c = np.array([lab[cidx[k][0]] for k in eval_keys])

        r = score(Z, cidx, eval_keys, y_c, None, P["none"], "none")
        for m, v in r.items():
            acc[f"none|{m}"].append(v)

        budgets = {f"T{s:g}": s for s in seconds}
        budgets["Tfull"] = "full"
        for tag, sec in budgets.items():
            sbd = {}
            for d in te_doms:
                w = []
                for k in held[d]:
                    ww = cidx[k]
                    n = len(ww) if sec == "full" else max(
                        1, int(np.ceil(sec / SEC_PER_WIN)))
                    w.append(np.asarray(pick_from_clip(ww, n, "prefix")))
                sbd[d] = stats_of(Z, np.concatenate(w))
            for kind in ("center", "strat"):
                r = score(Z, cidx, eval_keys, y_c, sbd, P[kind], kind)
                for m, v in r.items():
                    acc[f"{kind}_{tag}|{m}"].append(v)

        # transductive: 평가 창 전체로 μ, σ
        sbd = {d: stats_of(Z, np.concatenate(
            [cidx[k] for k in eval_keys if (k[0], k[1]) == d])) for d in te_doms}
        for kind in ("center", "strat"):
            r = score(Z, cidx, eval_keys, y_c, sbd, P[kind], kind)
            for m, v in r.items():
                acc[f"{kind}_trans|{m}"].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--seconds", type=float, nargs="+", default=[20, 40])
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--out", default=f"results/{CFG.prefix('stratnorm')}.npz")
    args = ap.parse_args()

    R = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)",
                                    os.path.basename(f)).groups())
        z = np.load(f)
        rng = np.random.default_rng(1000 * ts + sd)   # analyze_calib_protocol 과 동일
        r = run_one(z["cls"], z["meta"].astype(int), z["lab"].astype(int),
                    ts, args.seconds, args.n_rep, rng)
        for k, v in r.items():
            R[k][ts].append(v)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  none {r['none|clip']:.3f}  "
              f"T20 c/s {r['center_T20|clip']:.3f}/{r['strat_T20|clip']:.3f}  "
              f"Tfull c/s {r['center_Tfull|clip']:.3f}/{r['strat_Tfull|clip']:.3f}  "
              f"trans c/s {r['center_trans|clip']:.3f}/{r['strat_trans|clip']:.3f}",
              flush=True)

    def arr(k):
        return np.array([np.mean(R[k][s]) for s in sorted(R[k])])

    tags = [f"T{s:g}" for s in args.seconds] + ["Tfull", "trans"]
    names = {f"T{s:g}": f"감정별 {s:g}초 (총 {N_CLS*s:g}초)" for s in args.seconds}
    names.update(Tfull=f"감정별 1클립 전체", trans="transductive (평가 창 전체)")
    W = 104
    for unit in ("clip", "win", "win_bal"):
        print(f"\n{'='*W}\n{CFG.name}  {unit}   n={CFG.n_subjects}, 시드 평균 x 반복 "
              f"{args.n_rep}\n{'='*W}")
        base = arr(f"none|{unit}")
        print(f"  {'예산':<26}{'center (우리)':>16}{'strat (Fdez)':>16}"
              f"{'strat − center':>34}")
        print(f"  {'적응 없음':<26}{base.mean():>16.4f}")
        for t in tags:
            c, s_ = arr(f"center_{t}|{unit}"), arr(f"strat_{t}|{unit}")
            d = s_ - c
            lo, hi = bootstrap_ci(d)
            try:
                p = stats.wilcoxon(s_, c).pvalue
            except ValueError:
                p = float("nan")
            print(f"  {names[t]:<26}{c.mean():>16.4f}{s_.mean():>16.4f}"
                  f"   Δ{d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} "
                  f"{int((d>0).sum())}/{len(d)}")

    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
