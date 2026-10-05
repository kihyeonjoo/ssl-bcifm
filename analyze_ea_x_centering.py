"""
Input-space EA x feature-space centering, 2x2.

Two things in this pipeline remove per-recording variation.  Euclidean
Alignment whitens the RAW SIGNAL's channel amplitudes before the backbone
sees it; centering subtracts the domain mean of the FEATURES after.  Both are
transductive and label-free, and nobody has asked whether the second makes the
first unnecessary.

The question matters because EA's own evidence is weaker than it looks: the
+0.0357 in NEXT.md compares a three-seed mean (the A arm) against a single
unseeded run (no-EA), so part of that gap could be the same luck that made the
old 0.6161 look better than the true 0.5881.  This runs no-EA at three seeds
and compares like with like.

Comparisons, each at clip and window level:

  (a) EA effect, no centering      no-EA c1  vs  A-arm c1    retrain -> +0.0386
  (b) EA effect, with centering    no-EA c4  vs  A-arm c4    retrain -> +0.0386
  (c) centering effect, no EA      no-EA c4  vs  no-EA c1    decision rule
  (d) does centering replace EA    is (b) smaller than (a)

Plus the between-domain variance share (S3's measure) on no-EA features, to
see whether EA's contribution shows up as a smaller domain offset.

Reads two caches written by cache_ft_features.py.  No GPU.
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
from aggregate_logits import mde_from
from dataset_config import get as _cfg_root
CFG = _cfg_root()

from scipy import stats

from analyze_centering import (load_head, clip_index, l2, clip_reduce,
                               four_conditions, bootstrap_ci)
from analyze_centering_mechanism import variance_decomposition


def collect(cache_dir, ckpt_dir, ckpt_prefix):
    """{metric: {subject: [per-seed values]}} for one arm."""
    R = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(cache_dir, "S*_seed*.npz")))
    if not files:
        raise SystemExit(f"no cache in {cache_dir}")
    for f in files:
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)",
                                    os.path.basename(f)).groups())
        z = np.load(f)
        cls, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
        head = load_head(os.path.join(ckpt_dir, f"{ckpt_prefix}S{ts}_seed{sd}.pt"))
        c = four_conditions(cls, meta, lab, head, ts)
        for k, (w, cl) in c.items():
            R[f"c{k}_win"][ts].append(w)
            R[f"c{k}_clip"][ts].append(cl)
        for k, v in variance_decomposition(cls, meta).items():
            R[f"var_{k}"][ts].append(v)
        print(f"  S{ts} seed{sd}  c1 {c[1][1]:.4f}  c4 {c[4][1]:.4f}", flush=True)
    return R


def arr(R, key):
    return np.array([np.mean(R[key][s]) for s in sorted(R[key])])


def paired(a, b, label, threshold=None):
    d = a - b
    lo, hi = bootstrap_ci(d)
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    mark = ""
    if threshold is not None:
        # The design's minimum detectable effect is a statement about power, not
        # a second significance gate: once p is small the effect *was* detected.
        # So report where the point estimate and the CI lower bound each fall,
        # and never collapse the two into one "pass/fail" word.
        mark = ("  [점추정 한계 넘음" if d.mean() > threshold else "  [점추정 한계 아래")
        mark += ", 하한도 넘음]" if lo > threshold else ", 하한은 못 넘음]"
    print(f"  {label:<40} Δ{d.mean():+.4f}  CI[{lo:+.4f},{hi:+.4f}]"
          f"{'  0포함' if lo <= 0 <= hi else '       '}  p={p:.4f}  "
          f"{int((d > 0).sum())}/{len(d)}{mark}")
    return d.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea_cache", default=CFG.cache_dir)
    ap.add_argument("--noea_cache", default=CFG.cache_dir + "_noea")
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--out", default=f"results/{CFG.prefix('ea_x_centering')}.npz")
    args = ap.parse_args()

    print("[EA 있음] A팔")
    EA = collect(args.ea_cache, args.ckpt_dir, CFG.ckpt_prefix)
    print("\n[EA 없음]")
    NO = collect(args.noea_cache, args.ckpt_dir, CFG.noea_ckpt_prefix)

    print(f"\n{'='*94}\n입력 EA x 특징 중심화 2x2  (n={CFG.n_subjects} 피험자, 시드 평균 평균)\n{'='*94}")
    for unit in ("clip", "win"):
        print(f"\n[{unit} 단위]{'':<18}{'중심화 없음 (조건1)':>22} {'중심화 (조건4)':>20}"
              f" {'중심화 효과':>12}")
        for nm, R in (("EA 없음", NO), ("EA 있음 (A팔)", EA)):
            c1, c4 = arr(R, f"c1_{unit}"), arr(R, f"c4_{unit}")
            print(f"  {nm:<22} {c1.mean():>14.4f} ± {c1.std(ddof=1):.4f} "
                  f"{c4.mean():>10.4f} ± {c4.std(ddof=1):.4f} {c4.mean()-c1.mean():>+12.4f}")
        d1 = arr(EA, f"c1_{unit}").mean() - arr(NO, f"c1_{unit}").mean()
        d4 = arr(EA, f"c4_{unit}").mean() - arr(NO, f"c4_{unit}").mean()
        print(f"  {'EA 효과':<22} {d1:>+14.4f} {'':>8} {d4:>+10.4f}")

    # SEED 는 A팔에서 확정한 0.0386 을 쓴다(이 값으로 보고된 판정이 있다).
    # 다른 데이터셋은 그 비교의 짝지은 차이에서 계산한다 — sd 가 다르다.
    TH = CFG.mde
    print(f"\n{'='*94}\n짝지은 비교\n{'='*94}")
    for unit in ("clip", "win"):
        _th = TH if TH is not None else mde_from(
            arr(EA, f"c1_{unit}") - arr(NO, f"c1_{unit}"))
        print(f"\n[{unit}]  재학습 비교는 감지 한계 {_th:+.4f} 적용"
              f"{'' if TH is not None else ' (이 비교의 짝지은 차이에서 계산)'}")
        a = paired(arr(EA, f"c1_{unit}"), arr(NO, f"c1_{unit}"),
                   "(a) EA 효과, 중심화 없음  [재학습]", _th)
        b = paired(arr(EA, f"c4_{unit}"), arr(NO, f"c4_{unit}"),
                   "(b) EA 효과, 중심화 있음  [재학습]", _th)
        paired(arr(NO, f"c4_{unit}"), arr(NO, f"c1_{unit}"),
               "(c) 중심화 효과, EA 없음  [결정규칙]")
        paired(arr(EA, f"c4_{unit}"), arr(EA, f"c1_{unit}"),
               "    중심화 효과, EA 있음  [결정규칙]")
        # "does centering absorb EA" is an interaction, so test the
        # difference-of-differences per subject instead of eyeballing two means.
        inter = ((arr(EA, f"c1_{unit}") - arr(NO, f"c1_{unit}"))
                 - (arr(EA, f"c4_{unit}") - arr(NO, f"c4_{unit}")))
        ilo, ihi = bootstrap_ci(inter)
        try:
            ip = stats.wilcoxon(inter).pvalue
        except ValueError:
            ip = float("nan")
        verdict = ("구별 안 됨 (가법적)" if ilo <= 0 <= ihi
                   else ("중심화가 EA 이득을 일부 흡수" if inter.mean() > 0
                         else "중심화가 EA 이득을 키움"))
        print(f"  (d) 중심화가 EA 를 대체하는가: EA 효과 {a[0]:+.4f} -> {b[0]:+.4f}, "
              f"상호작용 Δ{inter.mean():+.4f} CI[{ilo:+.4f},{ihi:+.4f}] p={ip:.4f}"
              f"  -> {verdict}")

    print(f"\n{'='*94}\n도메인 간 분산 비율 (S3 방식)\n{'='*94}")
    for nm, R in (("EA 없음", NO), ("EA 있음 (A팔)", EA)):
        print(f"  {nm:<16} {arr(R,'var_between_frac').mean():.4f}   "
              f"PCA 1축 {arr(R,'var_pca_top1').mean():.3f}  "
              f"90% 도달 {arr(R,'var_n_dims_90pct').mean():.0f}축")
    print("  (참고) 사전학습 0.7969, 1축 0.646, 4축")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out,
             **{f"ea_{k}": arr(EA, k) for k in EA},
             **{f"noea_{k}": arr(NO, k) for k in NO})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
