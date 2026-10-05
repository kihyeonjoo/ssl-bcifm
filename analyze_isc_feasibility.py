"""V5 gate 0 (성립성 진단): LaBraM 특징에 '같은 영상의 같은 시점' 신호가 있는가.

방법 실험이 아니다.  제안하려는 방법(같은 캘리브레이션 영상에 대한 시점별 대응으로 새
사용자를 집단 공간에 정렬 — fMRI 의 response hyperalignment 에 해당)이 성립하려면, 보지 못한
피험자의 특징 궤적이 학습 피험자들의 같은 클립·같은 시점 평균과 **시간에 맞춰** 닮아야 한다.

  X_{a,k}  : 테스트 피험자, 세션 a, 클립 k 의 창 특징 (시간 순서, 도메인 중심화)
  T_{a,k}  : 같은 (a,k) 에서 학습 피험자들의 평균 궤적
  동적 ISC : 클립 안에서 시간 평균을 뺀 두 궤적의 Frobenius 코사인
  귀무     : T 를 클립 안에서 원형 이동 (3창 이상) — 자기상관은 보존, 시점 대응만 깸

정적 성분도 함께 본다: 클립 평균 특징이 '같은 클립' 의 학습 평균과, '같은 감정의 다른
클립' 학습 평균 중 어느 쪽에 더 가까운가 (감정 넘어 자극 고유 신호가 있는가).

창은 데이터셋 순서 = 클립 안 시간 순서다 (_slice_trial 이 start=0 부터 순서대로 자르고,
캐시는 shuffle=False).  클립 길이(창 수)는 같은 세션의 모든 피험자에서 같다 (2026-10-04 확인).
중심화는 진단 편의를 위해 도메인 전체 평균 (transductive) 이다 — 방법 평가가 아니다.

    DATASET=seed  python analyze_isc_feasibility.py --cache cache_ft_noea
    DATASET=seedv python analyze_isc_feasibility.py --cache cache_ft_seedv_noea
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
from analyze_centering import roles_for_fold, clip_index, l2, bootstrap_ci

CFG = _cfg_root()


def fcos(A, B):
    a, b = A.ravel(), B.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--min_shift", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    R = defaultdict(lambda: defaultdict(list))   # 지표 -> 테스트 피험자 -> [값]
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
        z = np.load(f)
        Z = l2(z["cls"]).astype(np.float64)
        meta = z["meta"].astype(int); lab = z["lab"].astype(int)
        cidx = clip_index(meta)
        train, _, _ = roles_for_fold(ts)
        # 도메인 중심화
        Zc = np.empty_like(Z)
        for d in {(int(m[0]), int(m[1])) for m in meta}:
            msk = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1])
            Zc[msk] = Z[msk] - Z[msk].mean(0)
        keys = sorted({(k[1], k[2]) for k in cidx if k[0] == ts})
        lab_of = {(a, c): int(lab[cidx[(ts, a, c)][0]]) for (a, c) in keys}
        T = {(a, c): np.mean([Zc[cidx[(s, a, c)]] for s in train], axis=0) for (a, c) in keys}
        Tmean = {k: v.mean(0) for k, v in T.items()}
        dyn, nul, lag1, st_same, st_emo, st_oth = [], [], [], [], [], []
        for (a, c) in keys:
            X = Zc[cidx[(ts, a, c)]]
            n = len(X)
            Xd = X - X.mean(0); Td = T[(a, c)] - T[(a, c)].mean(0)
            dyn.append(fcos(Xd, Td))
            shifts = [s for s in range(args.min_shift, n - args.min_shift + 1)]
            if shifts:
                sel = np.unique(np.linspace(0, len(shifts) - 1, min(10, len(shifts))).astype(int))
                nul.append(np.mean([fcos(Xd, np.roll(Td, shifts[i], axis=0)) for i in sel]))
            # 1창 어긋남 허용 (응답 지연) — 0, ±1 중 최대
            lag1.append(max(fcos(Xd[1:], Td[:-1]), fcos(Xd, Td), fcos(Xd[:-1], Td[1:])))
            # 정적: 같은 클립 / 같은 감정 다른 클립 / 다른 감정 클립
            m = X.mean(0)
            same_emo = [k for k in keys if k[0] == a and k != (a, c) and lab_of[k] == lab_of[(a, c)]]
            oth_emo = [k for k in keys if k[0] == a and lab_of[k] != lab_of[(a, c)]]
            st_same.append(fcos(m, Tmean[(a, c)]))
            if same_emo:
                st_emo.append(np.mean([fcos(m, Tmean[k]) for k in same_emo]))
            st_oth.append(np.mean([fcos(m, Tmean[k]) for k in oth_emo]))
        for k, v in (("dyn", dyn), ("null", nul), ("lag1", lag1), ("st_same", st_same),
                     ("st_emo", st_emo), ("st_oth", st_oth)):
            R[k][ts].append(float(np.mean(v)))
        R["frac_dyn_gt_null"][ts].append(float(np.mean(np.array(dyn) > np.array(nul))))
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  동적 ISC {np.mean(dyn):+.4f} (귀무 {np.mean(nul):+.4f})  "
              f"정적 같은클립 {np.mean(st_same):+.3f} / 같은감정 {np.mean(st_emo) if st_emo else float('nan'):+.3f} "
              f"/ 다른감정 {np.mean(st_oth):+.3f}", flush=True)

    A = {k: np.array([np.mean(v[s]) for s in sorted(v)]) for k, v in R.items()}
    n = len(A["dyn"])
    print(f"\n{'=' * 90}\n{CFG.name if hasattr(CFG, 'name') else ''}  테스트 피험자 {n}명 (시드 평균)\n{'=' * 90}")
    for lbl, a, b in (("동적 ISC vs 귀무 (시점 대응)", A["dyn"], A["null"]),
                      ("±1창 허용 vs 동적 ISC (지연)", A["lag1"], A["dyn"]),
                      ("정적: 같은 클립 vs 같은 감정 다른 클립", A["st_same"], A["st_emo"]),
                      ("정적: 같은 감정 vs 다른 감정", A["st_emo"], A["st_oth"])):
        d = a - b; lo, hi = bootstrap_ci(d)
        p = stats.wilcoxon(a, b).pvalue
        print(f"  {lbl:<34} {a.mean():+.4f} vs {b.mean():+.4f}  Δ {d.mean():+.4f} "
              f"[{lo:+.4f},{hi:+.4f}] p={p:.4g}  {int((d > 0).sum())}/{n}")
    print(f"  (클립 단위) 동적 ISC > 귀무 비율: {A['frac_dyn_gt_null'].mean():.2f}")
    if args.out:
        np.savez(args.out, subjects=np.array(sorted(R["dyn"])), **A)
        print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
