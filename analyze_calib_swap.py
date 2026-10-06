"""교환 캘리브레이션 대조 (2026-10-06) — 중심화 이득이 '본인' 캘리브레이션에서 오는가.

analyze_sla.py 의 center 경로 (같은 캘리브레이션 추출 · 같은 난수 소비 · 같은 prototype · 같은 채점) 에서 빼는 평균 μ 의
출처만 바꾼다.  평가 창 (캘리브레이션 클립을 뺀 본인의 나머지 클립) 과 prototype 은 모든 조건에서 같다.
  own         본인 · 같은 세션의 캘리브레이션 블록 (= center.  회귀 검사: --ref 의 center 와 피험자별로 같아야 한다)
  other_sess  본인 · 다른 세션의 캘리브레이션 블록 (다른 날 · 다른 영상) — 다른 두 세션 결과의 평균
  other_subj  다른 사람 · 같은 세션 · 같은 캘리브레이션 클립 · 같은 창 수 (같은 영상을 본 다른 사람) — 다른 사람 전원 결과의 평균
  pop         학습 피험자 도메인 평균들의 평균 (개인화 없는 고정 오프셋)
다른 사람은 그 fold 모델의 학습 · 검증 피험자다 (모델이 본 사람) — 그래도 '남의 캘리브레이션' 대조로는 유효하다.

    DATASET=seedv python analyze_calib_swap.py --cache cache_ft_seedv_noea_seed0 \
        --ref results/seedv_noea_seed0_sla.npz --out results/seedv_noea_seed0_calib_swap.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from analyze_calib_rotation import draw, BUDGETS, SEC
from analyze_centering import all_domains
from analyze_sla import load_fold, doms_of, calib_windows, score

CONDS = ("own", "other_sess", "other_subj", "pop")


def n_calib(w, sec):
    """calib_windows 와 같은 창 수 규칙."""
    n = len(w) if sec == "full" else max(1, int(np.ceil(sec / SEC)))
    return min(n, len(w))


def run_subject_swap(Z, meta, lab, cidx, P, T, s, doms, train, rng, n_rep):
    by = defaultdict(list)
    for k in sorted(cidx):
        if k[0] == s:
            by[((k[0], k[1]), int(lab[cidx[k][0]]))].append(k)
    tr_doms = [d for d in all_domains(meta) if d[0] in train]
    mu_pop = np.mean([Z[(meta[:, 0] == d[0]) & (meta[:, 1] == d[1])].mean(0) for d in tr_doms], axis=0)
    others = sorted({int(x) for x in meta[:, 0]} - {s})
    Pg = {d: P for d in doms}
    acc = defaultdict(lambda: defaultdict(list))
    for _ in range(n_rep):
        held = draw(by, doms, rng)                    # analyze_sla.run_subject 와 같은 소비 — 한 반복에 한 번
        hs = {k for v in held.values() for k in v}
        ev = [k for k in sorted(cidx) if k[0] == s and k not in hs]
        y_c = np.array([lab[cidx[k][0]] for k in ev])

        def sc(mu_):
            E = [((k[0], k[1]), Z[cidx[k]] - mu_[(k[0], k[1])]) for k in ev]
            return np.array(score(E, y_c, None, Pg))

        for tag, sec in BUDGETS.items():
            mu = {d: Z[calib_windows(held, cidx, lab, d, sec, T)[0]].mean(0) for d in doms}
            acc[tag]["own"].append(sc(mu))
            acc[tag]["other_sess"].append(np.mean(
                [sc({d: mu[doms[(i + r) % len(doms)]] for i, d in enumerate(doms)}) for r in range(1, len(doms))],
                axis=0))
            alt = []
            for g in others:
                mu_g = {}
                for d in doms:
                    idx = []
                    for k in held[d]:
                        n = n_calib(cidx[k], sec)
                        wg = cidx[(g, k[1], k[2])]
                        idx.append(np.asarray(wg[:min(n, len(wg))]))
                    mu_g[d] = Z[np.concatenate(idx)].mean(0)
                alt.append(sc(mu_g))
            acc[tag]["other_subj"].append(np.mean(alt, axis=0))
            acc[tag]["pop"].append(sc({d: mu_pop for d in doms}))
    return {t: {c: np.mean(v, axis=0) for c, v in d.items()} for t, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ref", required=True, help="같은 캐시의 analyze_sla 결과 (center 회귀 기준)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--max_files", type=int, default=0, help="시험 실행용")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    if args.max_files:
        files = files[:args.max_files]
    R = defaultdict(dict)
    for j, f in enumerate(files, 1):
        ts, sd, meta, lab, Z, cidx, train, val, P, T = load_fold(f)
        r = run_subject_swap(Z, meta, lab, cidx, P, T, ts, doms_of(meta, ts), train,
                             np.random.default_rng(1000 * ts + sd), args.n_rep)   # analyze_sla 와 같은 시드
        for t in BUDGETS:
            for c in CONDS:
                R[f"{t}__{c}__clip"][ts], R[f"{t}__{c}__win"][ts] = (float(x) for x in r[t][c])
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  Tfull clip " +
              "  ".join(f"{c} {r['Tfull'][c][0]:.3f}" for c in CONDS), flush=True)

    subs = sorted(R["Tfull__own__clip"])
    np.savez(args.out, subjects=np.array(subs), **{k: np.array([v[s] for s in subs]) for k, v in R.items()})
    print(f"[저장] {args.out}")
    ref = np.load(args.ref)
    pos = [int(np.flatnonzero(ref["subjects"] == s)[0]) for s in subs]
    for t in BUDGETS:
        for m in ("clip", "win"):
            d = np.abs(np.array([R[f"{t}__own__{m}"][s] for s in subs]) - ref[f"{t}__center__{m}"][pos]).max()
            print(f"  [회귀] {t} {m}: own vs center 최대 차이 {d:.2e}")


if __name__ == "__main__":
    main()
