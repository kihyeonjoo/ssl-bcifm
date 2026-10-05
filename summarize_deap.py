"""DEAP 결과 요약 — analyze_deap.py 의 npz 를 분야 관례 (피험자 단위 짝지은 Wilcoxon, 향상 피험자 수) 로 정리.

    DATASET=deap python summarize_deap.py --res results/deap_calib_noea.npz [--mis results/deap_misalignment_noea.npz]

지표: clip_bal (4사분면 클립 균형 정확도, 주), win_bal, clip_vbal / clip_abal (이진 정서가 · 각성도, 같은 예측).
우연 수준: 4사분면 0.25, 이진 0.5.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

BUD = ("T20", "T40", "Tfull")


def cmp(z, a, b):
    x, y = z[a], z[b]; d = x - y
    p = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    return f"{d.mean():+.4f} (p={p:.3f}, {int((d > 0).sum())}/{len(d)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--mis", default=None)
    args = ap.parse_args()
    z = np.load(args.res)
    n = len(z["subjects"])
    print(f"DEAP — 피험자 {n}명 (피험자 하나 빼기, 시드별 평균).  우연: 4사분면 0.25, 이진 0.5\n")
    for met in ("clip_bal", "win_bal", "clip_vbal", "clip_abal", "clip"):
        print(f"== {met}")
        print(f"  적응 없음           {z[f'none__none__{met}'].mean():.4f}")
        for t in BUD:
            print(f"  중심화 {t:5s}        {z[f'{t}__center__{met}'].mean():.4f}   − 적응 없음 "
                  f"{cmp(z, f'{t}__center__{met}', f'none__none__{met}')}")
        print(f"  전달식 상한          {z[f'transductive__center__{met}'].mean():.4f}   − 중심화 끝까지 "
              f"{cmp(z, f'transductive__center__{met}', f'Tfull__center__{met}')}")
        sing = [z[f'single{c}__center__{met}'] for c in range(4)]
        print(f"  한 사분면만 (평균)    {np.mean(sing):.4f}   − 중심화 끝까지 "
              f"{np.mean([x - z[f'Tfull__center__{met}'] for x in sing], axis=1).mean():+.4f}")
        for t in BUD:
            print(f"  [{t}] SLA − 중심화 {cmp(z, f'{t}__sla__{met}', f'{t}__center__{met}')}   "
                  f"L3 − 중심화 {cmp(z, f'{t}__l3__{met}', f'{t}__center__{met}')}   "
                  f"CR − 중심화 {cmp(z, f'{t}__cr__{met}', f'{t}__center__{met}')}")
            print(f"  [{t}] SLA − L3 {cmp(z, f'{t}__sla__{met}', f'{t}__l3__{met}')}   "
                  f"SLA+L3 − L3 {cmp(z, f'{t}__slal3__{met}', f'{t}__l3__{met}')}")
        print()
    for t in BUD:
        k = f"{t}__cr__applied"
        if k in z.files:
            print(f"  라벨 회전이 적용된 비율 {t}: {z[k].mean():.3f}  (본인 평점이 네 사분면을 모두 덮어야 회전)")
    if args.mis:
        c = np.load(args.mis)["c"]
        print(f"\n  어긋남 c = {c.mean():.3f} (범위 {c.min():.3f}~{c.max():.3f})  — SEED 0.754 · SEED-IV 0.428 · SEED-V 0.322")


if __name__ == "__main__":
    main()
