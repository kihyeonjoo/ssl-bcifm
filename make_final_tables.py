"""
최종 사다리 표를 하나의 CSV 로 굳힌다 — results/final_tables.csv.

S13(EA 현실적)과 EA 없음 팔의 사다리를 한 파일에 모으고, 각 행에 평균·sd·직전 단계
대비 짝지은 이득·CI·Wilcoxon p·이긴 수를 담는다.  이후 인용은 이 파일에서 한다.

단계
  1 none        중심화 없음, 균등 집계
  2 ea          + EA 현실적            (EA 팔에만 존재)
  3 center      + 중심화
  4 timeweight  + 시간 가중 (clip 만; window 는 정의상 불변)
"""

from __future__ import annotations

import csv

import numpy as np
from scipy import stats

from analyze_centering import bootstrap_ci

MDE = 0.0386
BUDGETS = [("T20", "60s"), ("T40", "120s"), ("Tfull", "~11min")]
PATHS = [("proto", "cond4_prototype"), ("head", "cond2_head")]
# 지표 접미사.  ""=정확도, "_f1"=macro-F1, "_bal"=balanced accuracy.
# SEED 는 클립 라벨이 225/225/225 로 완전 균형이라 **clip 의 _bal 은 정확도와 같다**.
# 창 수는 클래스별로 1.058배 차이나므로 window 에서만 다르다.
SUFFIXES = ["", "_f1", "_bal"]


def row(arm, budget, step, path, metric, a, prev, note=""):
    r = {"arm": arm, "budget": budget, "step": step, "path": path,
         "metric": metric, "mean": f"{a.mean():.4f}",
         "sd": f"{a.std(ddof=1):.4f}", "note": note}
    if prev is None:
        r.update(delta="", ci_lo="", ci_hi="", p="", wins="")
    else:
        d = a - prev
        lo, hi = bootstrap_ci(d)
        r.update(delta=f"{d.mean():+.4f}", ci_lo=f"{lo:+.4f}",
                 ci_hi=f"{hi:+.4f}",
                 p=f"{stats.wilcoxon(a, prev).pvalue:.4f}",
                 wins=f"{int((d > 0).sum())}/{len(d)}")
    return r


def main():
    E = np.load("results/final_ladder.npz")
    NO = np.load("results/noea_ladder.npz")
    # 1단계(EA 없음·중심화 없음)는 두 팔에서 같은 값이다.  예전에는
    # calib_protocol_noea_r10.npz 에서 가져왔는데 그 파일에는 macro-F1 / balanced
    # accuracy 가 없다.  noea_ladder.npz 의 같은 값(최대차 2.2e-16)을 쓴다.
    NB = NO
    rows = []

    def has(z, k):
        return k in z.files

    for path, pname in PATHS:
        for unit in ("clip", "win"):
            for suf in SUFFIXES:
                metric = f"{path}_{unit}{suf}"
                # ── EA arm ────────────────────────────────────────────
                if not has(NB, f"none__{metric}"):
                    continue
                base = NB[f"none__{metric}"]
                for tag, blab in BUDGETS:
                    rows.append(row("EA_realistic", blab, "1_none", pname,
                                    metric, base, None))
                    ea = E[f"eaRealNoCen_{tag}__{metric}"]
                    # 1단계는 no-EA 로 학습한 모델, 2단계는 EA 로 학습한 모델이다.
                    # EA 는 입력 분포를 바꾸므로 재학습 없이 "추가" 할 수 없다 —
                    # 즉 이 한 칸은 사다리의 다른 칸과 달리 **재학습 비교**다.
                    # 구조가 같은 final_config 행에만 경고가 붙어 있어서
                    # 여기에도 같은 표시를 단다 (2026-10-03).
                    rows.append(row("EA_realistic", blab, "2_ea", pname,
                                    metric, ea, base,
                                    note=("retraining comparison (step 1 is the "
                                          "no-EA model, step 2 the EA model); "
                                          "not an additive ladder step")))
                    cen = E[f"eaReal_{tag}__{metric}"]
                    rows.append(row("EA_realistic", blab, "3_center", pname,
                                    metric, cen, ea))
                    k = f"eaRealTW_{tag}__{metric}"
                    if unit == "clip" and has(E, k):
                        rows.append(row("EA_realistic", blab, "4_timeweight",
                                        pname, metric, E[k], cen))
                # ── no-EA arm ─────────────────────────────────────────
                nb = NO[f"none__{metric}"]
                for tag, blab in BUDGETS:
                    rows.append(row("no_EA", blab, "1_none", pname, metric,
                                    nb, None))
                    cen = NO[f"cen_{tag}__{metric}"]
                    rows.append(row("no_EA", blab, "3_center", pname, metric,
                                    cen, nb))
                    k = f"tw_{tag}__{metric}"
                    if unit == "clip" and has(NO, k):
                        rows.append(row("no_EA", blab, "4_timeweight", pname,
                                        metric, NO[k], cen))

    # ── final configuration: EA vs no EA (a RETRAINING comparison) ──────
    for path, pname in PATHS:
        for tag, blab in BUDGETS:
            for suf in SUFFIXES:
                for unit, ea_key, no_key in (
                        ("clip", f"eaRealTW_{tag}__{path}_clip{suf}",
                         f"tw_{tag}__{path}_clip{suf}"),
                        ("win", f"eaReal_{tag}__{path}_win{suf}",
                         f"cen_{tag}__{path}_win{suf}")):
                    if not (has(E, ea_key) and has(NO, no_key)):
                        continue
                    a, b = E[ea_key], NO[no_key]
                    r = row("EA_vs_noEA", blab, "final_config", pname,
                            f"{path}_{unit}{suf}", a, b)
                    r["note"] = ("retraining comparison; MDE +0.0386 "
                                 + ("cleared" if (a - b).mean() > MDE
                                    else "not cleared"))
                    rows.append(r)

    cols = ["arm", "budget", "step", "path", "metric", "mean", "sd", "delta",
            "ci_lo", "ci_hi", "p", "wins", "note"]
    with open("results/final_tables.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[저장] results/final_tables.csv  ({len(rows)} 행)")
    print("\n핵심 행 (조건4 clip):")
    for r in rows:
        if (r["path"] == "cond4_prototype" and r["metric"] == "proto_clip"
                and r["arm"] != "EA_vs_noEA"):
            print(f"  {r['arm']:<13}{r['budget']:<8}{r['step']:<14}"
                  f"{r['mean']}  {r['delta']:>8} {r['p']:>8} {r['wins']:>7}")


if __name__ == "__main__":
    main()
