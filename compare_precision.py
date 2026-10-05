"""정밀도 강건성 (2026-10-04): SEED no-EA 팔의 bf16 캐시 vs fp32 캐시.

SEED 모델의 CLS 특징이 bf16 정밀도에 민감하다는 것을 찾았다 (피험자 1 에서 창의 21% 가 fp32 와
코사인 < 0.9).  기존 결과는 학습·평가 때와 같은 bf16 캐시로 냈다.  여기서는 같은 분석을 fp32
캐시로 다시 낸 결과와 나란히 놓고, **결론(부호·유의성)이 바뀌는지**를 본다.  기존 판정을 다시
내리는 것이 아니다 — 사전 등록 판정은 bf16 결과로 이미 확정됐다.

    python compare_precision.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
from analyze_centering import l2, bootstrap_ci

BF, FP = "cache_ft_noea", "cache_ft_noea_fp32"
BUD = ("T20", "T40", "Tfull")


def feature_level():
    meds, f09, f05 = [], [], []
    for f in sorted(glob.glob(os.path.join(FP, "S*_seed*.npz"))):
        A = np.load(os.path.join(BF, os.path.basename(f))); B = np.load(f)
        assert (A["meta"] == B["meta"]).all() and (A["lab"] == B["lab"]).all()
        cos = np.sum(l2(A["cls"]) * l2(B["cls"]), 1)
        meds.append(np.median(cos)); f09.append(np.mean(cos < 0.9)); f05.append(np.mean(cos < 0.5))
    meds, f09, f05 = map(np.array, (meds, f09, f05))
    print(f"[특징] fold 파일 {len(meds)}개 — 창별 코사인(bf16 vs fp32) 중앙값의 범위 "
          f"{meds.min():.3f}~{meds.max():.3f}")
    print(f"       코사인 < 0.9 인 창 비율: 파일 평균 {f09.mean():.3f}, 범위 {f09.min():.3f}~{f09.max():.3f}")
    print(f"       코사인 < 0.5 인 창 비율: 파일 평균 {f05.mean():.3f}, 범위 {f05.min():.3f}~{f05.max():.3f}")
    np.savez("results/precision_feature_level.npz", median_cos=meds, frac_lt09=f09, frac_lt05=f05)


def cmp(a, b):
    d = a - b
    p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return d.mean(), p, int((d > 0).sum()), len(d)


def side(title, pairs):
    """pairs: [(라벨, (bf16 npz, 키 new, 키 ref), (fp32 npz, 키 new, 키 ref))]"""
    print(f"\n[{title}]")
    print(f"  {'':34s}{'bf16 Δ':>9s}{'p':>8s}{'이긴':>7s}   {'fp32 Δ':>9s}{'p':>8s}{'이긴':>7s}   결론 같음?")
    for lbl, (za, na, ra), (zb, nb, rb) in pairs:
        d1, p1, w1, n1 = cmp(za[na], za[ra]); d2, p2, w2, n2 = cmp(zb[nb], zb[rb])
        same = (np.sign(d1) == np.sign(d2)) and ((p1 < 0.05) == (p2 < 0.05))
        print(f"  {lbl:34s}{d1:>+9.4f}{p1:>8.4f}{w1:>4d}/{n1:<3d}  {d2:>+9.4f}{p2:>8.4f}{w2:>4d}/{n2:<3d}  "
              f"{'예' if same else '**아니오**'}")


def load(path):
    return np.load(path) if os.path.exists(path) else None


def main():
    feature_level()
    cp_b, cp_f = load("results/calib_protocol_noea_r10.npz"), load("results/calib_protocol_noea_r10_fp32.npz")
    if cp_b is not None and cp_f is not None:
        pairs = []
        for t in BUD:
            for m in ("proto_clip", "proto_win", "head_clip"):
                pairs.append((f"중심화 {t} − 없음 ({m})", (cp_b, f"{t}__{m}", f"none__{m}"),
                              (cp_f, f"{t}__{m}", f"none__{m}")))
        side("중심화 사다리 (S13 3단계, no-EA)", pairs)
        print("  절대값 (조건4 clip): " + "  ".join(
            f"{k} bf16 {cp_b[k + '__proto_clip'].mean():.4f} / fp32 {cp_f[k + '__proto_clip'].mean():.4f}"
            for k in ("none", "T20", "Tfull")))
    cr_b, cr_f = load("results/calib_rotation_noea.npz"), load("results/calib_rotation_noea_fp32.npz")
    if cr_b is not None and cr_f is not None:
        side("V4 CR − 중심화 (clip)", [(f"CR {t}", (cr_b, f"{t}__cr__clip", f"{t}__center__clip"),
                                         (cr_f, f"{t}__cr__clip", f"{t}__center__clip")) for t in BUD])
    lc_b, lc_f = load("results/labelcalib_noea.npz"), load("results/labelcalib_noea_fp32.npz")
    if lc_b is not None and lc_f is not None:
        pairs = [(f"{m} 클립 전체 − 기준선", (lc_b, f"Tfull__{m}__proto_clip", "Tfull__base__proto_clip"),
                  (lc_f, f"Tfull__{m}__proto_clip", "Tfull__base__proto_clip")) for m in ("L2", "L3")]
        pairs.append(("L3 − L2 클립 전체", (lc_b, "Tfull__L3__proto_clip", "Tfull__L2__proto_clip"),
                      (lc_f, "Tfull__L3__proto_clip", "Tfull__L2__proto_clip")))
        side("V4 부록 A (clip)", pairs)
    s_b, s_f = load("results/sla_noea.npz"), load("results/sla_noea_fp32.npz")
    if s_b is not None and s_f is not None:
        pairs = []
        for t in BUD:
            pairs.append((f"SLA − 중심화 {t}", (s_b, f"{t}__sla__clip", f"{t}__center__clip"),
                          (s_f, f"{t}__sla__clip", f"{t}__center__clip")))
            pairs.append((f"SLA+L3 − L3 {t}", (s_b, f"{t}__slal3__clip", f"{t}__l3__clip"),
                          (s_f, f"{t}__slal3__clip", f"{t}__l3__clip")))
        side("V5 gate 1 SEED (clip)", pairs)


if __name__ == "__main__":
    main()
