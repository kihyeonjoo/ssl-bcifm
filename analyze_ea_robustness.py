"""
Why diag EA fails to whiten, and what a robust replacement would look like.

The stage-0 measurement found ``mean |diag(R̄_after)/α² − 1|`` running to
thousands where it should be near zero.  This script establishes whether that
is a handful of extreme segments poisoning a mean, or the typical segment
genuinely being mis-scaled — and then compares four ways of estimating R̄.

Read-only: loads data, computes statistics, trains nothing and writes no
transform back into the pipeline.

Sections mirror the analysis request:
  1  what ``segment_power`` actually measures, and why it disagrees with the
     median channel std
  2  how common extreme segments are, and whether they cluster
  3  the whitening-quality metric recomputed with a median instead of a mean
  4  four R̄ estimators compared on quality, output scale and extreme handling
  5  the α that lands each estimator's output on the scale100 reference
  6  what the extremes look like at the model's input after EA
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from data.seed_raw_dataset import SEEDRawDataset, SEED_CH_NAMES

from dataset_config import get as _cfg_root
CFG = _cfg_root()
ROOT = _cfg_root().root
REF_STD = 0.1278          # measured scale100 median channel std (stage 0)
EPS = 1e-6


def groups_of(ds):
    g = defaultdict(list)
    for i, (s, se, _c) in enumerate(ds._seg_meta):
        g[(s, se)].append(i)
    return dict(g)


def load(subj):
    return SEEDRawDataset(root=ROOT, subjects=[subj], sessions=[1, 2, 3],
                          segment_length=800, step=800, patch_size=200,
                          norm="none", ea=False)


# ── R̄ estimators ────────────────────────────────────────────────────────────
# Each returns the DIAGONAL only (diag EA never uses the off-diagonal), as a
# vector of per-channel second moments.

def diag_trim(segs, trim):
    """Current method: mean covariance after dropping the highest-power
    segments, where 'power' is summed over channels."""
    pw = np.array([(X ** 2).sum() / X.shape[1] for X in segs])
    keep = np.arange(len(segs))
    if trim > 0 and len(segs) > 20:
        cut = np.quantile(pw, 1.0 - trim)
        keep = np.flatnonzero(pw <= cut)
        if keep.size < 10:
            keep = np.arange(len(segs))
    return np.mean([(segs[i] ** 2).mean(1) for i in keep], axis=0)


def diag_trace_norm(segs):
    """Average the SHAPE of each segment's covariance, then restore a scale.

    Normalising by the trace makes every segment contribute equally regardless
    of its amplitude, so one saturated segment can no longer set the transform.
    The scale comes back from the median trace, which the extremes cannot move.
    """
    d = np.array([(X ** 2).mean(1) for X in segs])        # (n, C)
    tr = d.sum(1, keepdims=True)
    tr = np.maximum(tr, 1e-30)
    shape = (d / tr).mean(0)
    return shape * np.median(tr)


def diag_median(segs):
    """Per-channel median of the segment second moments.

    The most direct answer to a heavy tail: a channel's scale is what that
    channel usually does, and half the segments would have to be extreme
    before the estimate moves.
    """
    return np.median(np.array([(X ** 2).mean(1) for X in segs]), axis=0)


ESTIMATORS = {
    "(i)  trim=0.05 (현재)": lambda S: diag_trim(S, 0.05),
    "(ii) trim=0.20":        lambda S: diag_trim(S, 0.20),
    "(iii) trace 정규화":     diag_trace_norm,
    "(iv) 채널별 중앙값":      diag_median,
}


def whiten_diag(d, alpha):
    """W = alpha * diag(d)^(-1/2) with the pipeline's relative eigenvalue floor."""
    d = np.maximum(d, EPS * max(float(d.max()), 1e-12))
    return alpha * d ** -0.5, (d <= EPS * max(float(d.max()), 1e-12) * 1.0000001).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--stride", type=int, default=4,
                    help="subsample segments for the covariance work")
    args = ap.parse_args()
    subjects = (CFG.subjects if args.subjects == "all"
                else [int(x) for x in args.subjects.split(",")])

    # ══ 1. what segment_power measures ═══════════════════════════════════
    print("=" * 84)
    print("1. segment_power 의 정의와 채널 std 와의 불일치")
    print("=" * 84)
    print("segment_power(X) = tr(X Xᵀ)/T = 모든 채널의 2차 모멘트 '합'.")
    print("  - 평균을 빼지 않는다 (분산이 아니라 2차 모멘트)")
    print("  - 채널에 대해 '합'이므로 가장 나쁜 채널 하나가 지배할 수 있다")
    print("  - 반면 '채널 std 중앙값'은 그 채널을 무시한다  -> 둘이 어긋난다\n")
    print(f"{'group':>8} {'power중앙':>11} {'ch std중앙':>11} {'최악ch':>7} "
          f"{'그 std':>9} {'파워점유':>8} {'DC비중':>7}")
    worst = []
    for subj in subjects:
        ds = load(subj)
        for key, ids in sorted(groups_of(ds).items()):
            take = ids[::args.stride]
            X0 = [np.asarray(ds._segments[i][0], dtype=np.float64) for i in take]
            pw = np.median([(X ** 2).sum() / X.shape[1] for X in X0])
            sd = np.median(np.array([X.std(1) for X in X0]), axis=0)
            dc = np.median([(X.mean(1) ** 2).sum() / ((X ** 2).mean(1).sum()) for X in X0])
            j = int(np.argmax(sd))
            share = sd[j] ** 2 / (sd ** 2).sum()
            worst.append((key, SEED_CH_NAMES[j], sd[j], share))
            print(f"{str(key):>8} {pw:11.3e} {np.median(sd):11.2f} "
                  f"{SEED_CH_NAMES[j]:>7} {sd[j]:9.1f} {share:7.1%} {dc:7.1%}")
        del ds
    bad = [w for w in worst if w[3] > 0.5]
    print(f"\n  한 채널이 세션 파워의 50%↑를 차지하는 경우: {len(bad)}/{len(worst)} 세션")
    for key, ch, s, sh in sorted(bad, key=lambda x: -x[3])[:10]:
        print(f"    {key}  {ch:>5}  std={s:8.1f}  점유 {sh:.1%}")


    # ══ 2. how common are extreme segments, and do they cluster ══════════
    print("\n" + "=" * 84)
    print("2. 극단 세그먼트 실태 (세션 내 파워 중앙값 대비)")
    print("=" * 84)
    print(f"{'group':>8} {'n':>5} {'>10배':>8} {'>100배':>8} {'>1000배':>9} "
          f"{'최대배율':>10} {'최다클립':>9}")
    tot = np.zeros(3); n_all = 0
    clip_hits = defaultdict(int)
    for subj in subjects:
        ds = load(subj)
        for key, ids in sorted(groups_of(ds).items()):
            pw = np.array([(np.asarray(ds._segments[i][0], dtype=np.float64) ** 2).sum()
                           / ds._segments[i][0].shape[1] for i in ids])
            med = np.median(pw)
            r = pw / max(med, 1e-30)
            c = [(r > 10).sum(), (r > 100).sum(), (r > 1000).sum()]
            tot += c; n_all += len(ids)
            cl = defaultdict(int)
            for i, rr in zip(ids, r):
                if rr > 10:
                    cl[ds._seg_meta[i][2]] += 1
                    clip_hits[ds._seg_meta[i][2]] += 1
            top = max(cl.items(), key=lambda kv: kv[1])[0] if cl else "-"
            print(f"{str(key):>8} {len(ids):>5} {c[0]:>4}({c[0]/len(ids):4.1%}) "
                  f"{c[1]:>4}({c[1]/len(ids):4.1%}) {c[2]:>4}({c[2]/len(ids):5.1%}) "
                  f"{r.max():10.0f} {str(top):>9}")
        del ds
    print(f"\n  전체 {n_all}개 세그먼트 중  >10배 {tot[0]:.0f}({tot[0]/n_all:.2%})  "
          f">100배 {tot[1]:.0f}({tot[1]/n_all:.2%})  >1000배 {tot[2]:.0f}({tot[2]/n_all:.3%})")
    if clip_hits:
        top5 = sorted(clip_hits.items(), key=lambda kv: -kv[1])[:5]
        print("  >10배가 몰린 클립: " + ", ".join(f"clip{c}={n}" for c, n in top5))
        print(f"  (클립은 1..15, 균등하면 각 {sum(clip_hits.values())/15:.0f}개)")

    # ══ 3-6. estimators, quality, scale, extremes ════════════════════════
    print("\n" + "=" * 84)
    print("3-6. R̄ 추정 방식 비교")
    print("=" * 84)
    print("지표 설명")
    print("  평균기반 (c) : mean |diag(R̄_after)/α² − 1|   <- 0단계에서 쓴 것")
    print("  중앙값기반   : 세그먼트별 diag 공분산/α² 의 '중앙값'이 1에서 벗어난 정도")
    print("                 둘이 크게 다르면 보통 세그먼트는 맞고 극단값이 지표를 망친 것\n")

    acc = {k: defaultdict(list) for k in ESTIMATORS}
    for subj in subjects:
        ds = load(subj)
        for key, ids in sorted(groups_of(ds).items()):
            take = ids[::args.stride]
            S = [np.asarray(ds._segments[i][0], dtype=np.float64) for i in take]
            D = np.array([(X ** 2).mean(1) for X in S])        # (n, C) 2차 모멘트
            for name, fn in ESTIMATORS.items():
                d = fn(S)
                w, n_floor = whiten_diag(d, 1.0)               # α=1 로 계산 후 스케일링
                Dw = D * (w ** 2)                              # 화이트닝 후 채널별 2차 모멘트
                acc[name]["mean_dev"].append(float(np.abs(Dw.mean(0) - 1).mean()))
                acc[name]["med_dev"].append(float(np.abs(np.median(Dw, 0) - 1).mean()))
                # 출력 채널 std (α=1 기준) — 실제 α 는 곱셈이므로 나중에 스케일
                sd = np.array([X.std(1) for X in S]) * w
                acc[name]["med_std"].append(float(np.median(sd)))
                acc[name]["max_std"].append(float(sd.max()))
                acc[name]["p99_std"].append(float(np.quantile(sd, 0.99)))
                acc[name]["floored"].append(int(n_floor))
        del ds

    print(f"{'추정방식':>20} {'평균기반(c)':>13} {'중앙값기반':>11} "
          f"{'출력std/α':>10} {'맞추는α':>9} {'최대/중앙':>10} {'p99/중앙':>9} {'floor채널':>9}")
    rec = {}
    for name in ESTIMATORS:
        a = acc[name]
        mean_dev = np.median(a["mean_dev"]); med_dev = np.median(a["med_dev"])
        med_std = np.median(a["med_std"])
        alpha = REF_STD / med_std
        ratio_max = np.median(np.array(a["max_std"]) / np.array(a["med_std"]))
        ratio_p99 = np.median(np.array(a["p99_std"]) / np.array(a["med_std"]))
        fl = np.mean(a["floored"])
        rec[name] = dict(mean_dev=mean_dev, med_dev=med_dev, alpha=alpha,
                         ratio_max=ratio_max, ratio_p99=ratio_p99, floored=fl)
        print(f"{name:>20} {mean_dev:13.2f} {med_dev:11.4f} {med_std:10.4f} "
              f"{alpha:9.4f} {ratio_max:10.1f} {ratio_p99:9.2f} {fl:9.1f}")

    print("\n  '맞추는 α' = 출력 채널 std 중앙값이 scale100 기준 "
          f"{REF_STD} 이 되는 값")
    print("  '최대/중앙' = 그 세션에서 가장 큰 세그먼트 채널 std 가 중앙값의 몇 배인가")
    print("               (EA 후에도 이 배율이 크면 극단 세그먼트가 거대한 입력으로 남는다)")

    # ══ 6. clipping judgement ════════════════════════════════════════════
    print("\n" + "=" * 84)
    print("6. 진폭 클리핑이 필요한가 (구현하지 않음, 판단 근거만)")
    print("=" * 84)
    for name, r in rec.items():
        amp = REF_STD * r["ratio_max"]
        p99 = REF_STD * r["ratio_p99"]
        print(f"  {name:>20}  EA 후 최대 채널 std ≈ {amp:8.2f}  "
              f"(기준의 {r['ratio_max']:6.0f}배)   상위1% ≈ {p99:.3f}")
    print(f"\n  LaBraM 이 사전학습된 입력 범위는 채널 std ≈ {REF_STD} 근처다.")
    print("  최대값이 그 수백~수천 배로 남는다면, 그 세그먼트는 사전학습 분포 밖의")
    print("  입력이 되어 backbone 이 본 적 없는 영역에서 동작한다.")


if __name__ == "__main__":
    main()
