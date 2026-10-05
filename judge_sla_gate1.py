"""V5 gate 1 판정 — reports/V5_SLA_GATE1_PREREG.md (13:00 고정) 의 규칙을 기계적으로 적용한다.

    python judge_sla_gate1.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
from analyze_centering import bootstrap_ci

BUD = ("T20", "T40", "Tfull")


def holm(ps):
    ps = np.asarray(ps, float); n = len(ps); out = np.empty(n); m = 0.0
    for r, i in enumerate(np.argsort(ps)):
        m = max(m, min(1.0, ps[i] * (n - r))); out[i] = m
    return out


def cmp(a, b):
    d = a - b
    lo, hi = bootstrap_ci(d)
    p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return dict(d=float(d.mean()), lo=lo, hi=hi, p=float(p), w=int((d > 0).sum()), n=len(d))


def family(Z, new, ref, metric="clip"):
    rows = [(ds, t, cmp(Z[ds][f"{t}__{new}__{metric}"], Z[ds][f"{t}__{ref}__{metric}"]))
            for ds in Z for t in BUD]
    for (ds, t, r), h in zip(rows, holm([r["p"] for _, _, r in rows])):
        r["holm"] = float(h)
    return rows


def show(title, rows, Z, new, ref, metric="clip"):
    print(f"\n[{title}]  {new} − {ref}  ({metric})")
    for ds, t, r in rows:
        a, b = Z[ds][f"{t}__{new}__{metric}"].mean(), Z[ds][f"{t}__{ref}__{metric}"].mean()
        hol = f" Holm={r['holm']:.4f}" if "holm" in r else ""
        print(f"  {ds:7s}{t:6s} {ref} {b:.4f} → {new} {a:.4f}  Δ {r['d']:+.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.4f}{hol} {r['w']}/{r['n']}")


def main():
    for arm, files, primary in (("no-EA (주 입력)", {"SEED": "results/sla_noea.npz",
                                                      "SEED-V": "results/seedv_sla_noea.npz"}, True),
                                ("EA (부 입력, 기술)", {"SEED": "results/sla_ea.npz",
                                                       "SEED-V": "results/seedv_sla_ea.npz"}, False)):
        print(f"\n{'#' * 100}\n# {arm}\n{'#' * 100}")
        Z = {ds: np.load(f) for ds, f in files.items()}
        main_f = family(Z, "sla", "center")
        sub_f = family(Z, "slal3", "l3")
        show("주 족 (Holm 6)", main_f, Z, "sla", "center")
        show("부 족 (Holm 6)", sub_f, Z, "slal3", "l3")
        for new, ref in (("l3", "center"), ("sla", "l3"), ("slal3", "center")):
            rows = [(ds, t, cmp(Z[ds][f"{t}__{new}__clip"], Z[ds][f"{t}__{ref}__clip"])) for ds in Z for t in BUD]
            show("기술 (족 밖)", rows, Z, new, ref)
        rows = [(ds, t, cmp(Z[ds][f"{t}__sla__win"], Z[ds][f"{t}__center__win"])) for ds in Z for t in BUD]
        show("기술 (족 밖)", rows, Z, "sla", "center", "win")
        if not primary:
            continue

        get = lambda rows, ds, t: next(r for d_, t_, r in rows if d_ == ds and t_ == t)
        A_ds = {ds: (get(main_f, ds, "Tfull")["d"] >= 0.015 and get(main_f, ds, "Tfull")["holm"] < 0.05)
                for ds in Z}
        A = all(A_ds.values())
        B1 = any(get(sub_f, ds, "Tfull")["d"] > 0 and get(sub_f, ds, "Tfull")["holm"] < 0.05 for ds in Z)
        B2 = all(get(main_f, ds, "T40")["d"] > 0 and get(main_f, ds, "T40")["holm"] < 0.05 for ds in Z)
        B = B1 or B2
        if A and B:
            verdict = "진행"
        elif A or sum(A_ds.values()) == 1:
            verdict = "부분"
        else:
            verdict = "실패"
        sig_neg = [(ds, t, fam) for fam, rows in (("주", main_f), ("부", sub_f))
                   for ds, t, r in rows if r["d"] < 0 and r["holm"] < 0.05]
        order = ["실패", "부분", "진행"]
        final = order[max(0, order.index(verdict) - 1)] if sig_neg else verdict
        stop = all(get(main_f, ds, "Tfull")["d"] < 0.01 for ds in Z)
        print(f"\n{'=' * 100}\n판정 (사전 등록 규칙)")
        print(f"  A (두 데이터셋 클립 전체 SLA−center ≥ +0.015, Holm<0.05): {A}  {A_ds}")
        print(f"  B1 (클립 전체 SLA+L3 − L3 유의 양수, 1개 이상): {B1}")
        print(f"  B2 (T40 두 데이터셋 SLA−center 유의 양수): {B2}")
        print(f"  유의하게 음수인 칸: {sig_neg if sig_neg else '없음'}")
        print(f"  → 판정: {final}" + (f" (강등 전 {verdict})" if final != verdict else "")
              + ("   [중단 조건: 두 데이터셋 모두 클립 전체 Δ < +0.01]" if stop else ""))


if __name__ == "__main__":
    main()
