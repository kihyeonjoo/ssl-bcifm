"""
Paired comparison of two conditions, with the run-to-run noise made visible.

The unit of comparison is the SUBJECT, not the run: seeds are averaged within
a subject first, then the 15 subject means are paired across conditions.  Two
reasons.  Seeds of the same fold are not independent observations of anything
the comparison is about, so pooling them as extra rows would inflate n and
shrink every interval.  And the quantity of interest — "does this condition
help this person" — is defined per subject.

Three tests are printed because they fail differently:
  bootstrap CI    — no distributional assumption, shows the effect's range
  Wilcoxon        — rank-based, survives one subject with a wild delta
  paired t-test   — what the literature reports; listed for comparability

Window- and clip-level are both reported.  A clip-level score has 45 decisions
per subject versus ~2500 windows, so it moves in coarser steps and its spread
is wider; a gain that appears only at window level is a gain on correlated
near-duplicate samples.

Also printed, and the reason this script exists: the seed-to-seed spread
within each subject.  Any claimed improvement smaller than that is not
distinguishable from restarting the same run.

Inputs are the ``results_csv`` files written by finetune_labram_hemi_aux.py.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

# Conditions run before seeding existed: one unseeded run per fold, so they
# carry no noise estimate.  Shown for context only, never tested against.
LEGACY = {
    "EA 없음": {1: .5241, 2: .6105, 3: .5934, 4: .5819, 5: .5669, 6: .4794,
                7: .4727, 8: .6793, 9: .4854, 10: .5067, 11: .6370, 12: .4929,
                13: .4691, 14: .5364, 15: .6508},
    "full EA": {1: .6089, 2: .5269, 3: .7094, 4: .4755, 5: .5986, 6: .4968,
                7: .6006, 8: .7375, 9: .6283, 10: .6045, 11: .6730, 12: .5693,
                13: .6211, 14: .5625, 15: .6322},
}


def load(path):
    """-> {subject: {seed: row}}"""
    out = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v == "":
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            out[int(row["subject"])][int(row["seed"])] = row
    return dict(out)


def subject_means(data, key):
    """{subject: mean over seeds}, skipping subjects where `key` is absent."""
    out = {}
    for s, by_seed in data.items():
        vals = [r[key] for r in by_seed.values() if key in r]
        if vals:
            out[s] = float(np.mean(vals))
    return out


def bootstrap_ci(d, n=20000, alpha=0.05, seed=0):
    """Percentile CI of the mean paired difference."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(1)
    return (float(np.quantile(boot, alpha / 2)),
            float(np.quantile(boot, 1 - alpha / 2)))


def compare(a, b, name_a, name_b, key, label):
    ma, mb = subject_means(a, key), subject_means(b, key)
    common = sorted(set(ma) & set(mb))
    if not common:
        print(f"\n[{label}] 공통 피험자 없음 — 건너뜀")
        return
    x = np.array([ma[s] for s in common])
    y = np.array([mb[s] for s in common])
    d = x - y

    print(f"\n[{label}]  n={len(common)} 피험자")
    print(f"  {name_a:<22} {x.mean():.4f} ± {x.std(ddof=1):.4f}")
    print(f"  {name_b:<22} {y.mean():.4f} ± {y.std(ddof=1):.4f}")
    print(f"  차이 (A−B)             {d.mean():+.4f}")
    lo, hi = bootstrap_ci(d)
    print(f"  부트스트랩 95% CI      [{lo:+.4f}, {hi:+.4f}]"
          f"{'   <- 0 포함' if lo <= 0 <= hi else ''}")
    if len(common) >= 6:
        try:
            w = stats.wilcoxon(x, y)
            print(f"  Wilcoxon               W={w.statistic:.1f}  p={w.pvalue:.4f}")
        except ValueError as e:                     # all differences zero
            print(f"  Wilcoxon               계산 불가 ({e})")
    else:
        print("  Wilcoxon               n<6, 생략")
    t = stats.ttest_rel(x, y)
    print(f"  paired t-test          t={t.statistic:+.3f}  p={t.pvalue:.4f}")
    print(f"  A가 나은 피험자        {int((d > 0).sum())}/{len(common)}")


def noise_table(runs, key="accuracy"):
    """Seed-to-seed spread inside each subject — the floor a gain must clear."""
    print(f"\n{'='*70}\n실행 잡음 (같은 피험자, 시드만 다름) — {key}\n{'='*70}")
    names = list(runs)
    print(f"{'subj':>4}" + "".join(f"{n:>22}" for n in names))
    per_cond = {n: [] for n in names}
    subjects = sorted(set().union(*[set(r) for r in runs.values()]))
    for s in subjects:
        line = f"{s:>4}"
        for n in names:
            by_seed = runs[n].get(s, {})
            v = [r[key] for r in by_seed.values() if key in r]
            if len(v) > 1:
                sd = float(np.std(v, ddof=1))
                per_cond[n].append(sd)
                line += f"{np.mean(v):>13.4f} ±{sd:.4f}"
            elif v:
                line += f"{v[0]:>13.4f}  (1seed)"
            else:
                line += f"{'-':>22}"
        print(line)
    print()
    for n in names:
        if per_cond[n]:
            print(f"  {n}: 시드 간 sd 평균 {np.mean(per_cond[n]):.4f}  "
                  f"최대 {np.max(per_cond[n]):.4f}")
    floor = max((np.mean(v) for v in per_cond.values() if v), default=None)
    if floor:
        print(f"\n  -> 이 값보다 작은 개선은 재실행과 구분되지 않는다 "
              f"(기준 {floor:.4f})")


def legacy_table(runs):
    print(f"\n{'='*70}\n참고: 시드 없이 fold당 1회만 돌린 기존 결과 (검정에는 쓰지 않음)"
          f"\n{'='*70}")
    names = list(runs)
    print(f"{'subj':>4}" + "".join(f"{n:>14}" for n in LEGACY)
          + "".join(f"{n:>14}" for n in names))
    subjects = sorted(set().union(*[set(v) for v in LEGACY.values()]))
    for s in subjects:
        line = f"{s:>4}"
        for n in LEGACY:
            line += f"{LEGACY[n].get(s, float('nan')):>14.4f}"
        for n in names:
            m = subject_means(runs[n], "accuracy")
            line += f"{m.get(s, float('nan')):>14.4f}"
        print(line)
    line = f"{'평균':>4}"
    for n in LEGACY:
        line += f"{np.mean(list(LEGACY[n].values())):>14.4f}"
    for n in names:
        m = subject_means(runs[n], "accuracy")
        line += f"{np.mean(list(m.values())):>14.4f}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--name_a", default=None)
    ap.add_argument("--name_b", default=None)
    ap.add_argument("--no_legacy", action="store_true")
    args = ap.parse_args()
    na = args.name_a or args.csv_a
    nb = args.name_b or args.csv_b

    A, B = load(args.csv_a), load(args.csv_b)
    print(f"A = {na}   {len(A)} 피험자")
    print(f"B = {nb}   {len(B)} 피험자")

    compare(A, B, na, nb, "accuracy", "window 단위 accuracy")
    compare(A, B, na, nb, "f1_macro", "window 단위 macro-F1")
    if any("clip_accuracy" in r for by in A.values() for r in by.values()):
        compare(A, B, na, nb, "clip_accuracy", "clip 단위 accuracy")
        compare(A, B, na, nb, "clip_balanced", "clip 단위 balanced acc")

    noise_table({na: A, nb: B}, "accuracy")
    if any("clip_accuracy" in r for by in A.values() for r in by.values()):
        noise_table({na: A, nb: B}, "clip_accuracy")
    if not args.no_legacy:
        legacy_table({na: A, nb: B})


if __name__ == "__main__":
    main()
