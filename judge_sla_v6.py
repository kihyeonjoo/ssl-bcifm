"""V6 판정 — reports/V6_SEEDIV_SLA_PREREG.md (14:18 고정) 의 표를 기계적으로 적용한다.

    DATASET=seediv python judge_sla_v6.py --res results/seediv_sla_noea.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from judge_sla_gate1 import holm, cmp, BUD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--c", type=float, default=None, help="측정한 어긋남 c (예측 대조용)")
    args = ap.parse_args()
    Z = np.load(args.res)
    rows = [(t, cmp(Z[f"{t}__sla__clip"], Z[f"{t}__center__clip"])) for t in BUD]
    for (t, r), h in zip(rows, holm([r["p"] for _, r in rows])):
        r["holm"] = float(h)
    print("[주] SLA − center (clip, Holm 3)")
    for t, r in rows:
        print(f"  {t:6s} center {Z[f'{t}__center__clip'].mean():.4f} → SLA {Z[f'{t}__sla__clip'].mean():.4f}  "
              f"Δ {r['d']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.4f} Holm={r['holm']:.4f} {r['w']}/{r['n']}")
    for nm, new, ref in (("SLA+L3 − L3", "slal3", "l3"), ("SLA − L3", "sla", "l3"), ("L3 − center", "l3", "center")):
        print(f"[기술] {nm}: " + "  ".join(
            f"{t} {cmp(Z[f'{t}__{new}__clip'], Z[f'{t}__{ref}__clip'])['d']:+.4f}"
            f" (p={cmp(Z[f'{t}__{new}__clip'], Z[f'{t}__{ref}__clip'])['p']:.3f})" for t in BUD))
    r = dict(rows)
    full, t40 = r["Tfull"], r["T40"]
    neg = any(x["d"] < 0 and x["holm"] < 0.05 for _, x in rows)
    if full["d"] >= 0.015 and full["holm"] < 0.05 and not neg:
        v = "재현"
    elif (full["d"] > 0 and full["holm"] < 0.05) or (t40["d"] >= 0.015 and t40["holm"] < 0.05 and not neg):
        v = "부분"
    else:
        v = "재현 안 됨"
    print(f"\n판정: {v}   (유의하게 음수인 예산: {'있음' if neg else '없음'})")
    if args.c is not None:
        pred = 0.0065 + (0.754 - args.c) / (0.754 - 0.322) * (0.0610 - 0.0065)
        q = "재현" if args.c < 0.5 else ("재현 안 됨 (Δ<+0.01)" if args.c > 0.7 else "방향만 (Δ>0)")
        print(f"예측 대조: c={args.c:.3f} → 예측 이득 {pred:+.4f}, 질적 예측 '{q}'  |  관측 {full['d']:+.4f}")


if __name__ == "__main__":
    main()
