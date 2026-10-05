"""
분석 C: 클립 안에서 감정 신호가 언제 올라오는가.

S10 에서 "같은 창 개수를 클립 전체에 퍼뜨리면 앞부분만 쓰는 것보다 낫다" 가 나왔다.
그 해석은 "정서 반응이 영상 시작 직후가 아니라 시간이 지나야 올라온다" 인데, 그것을
직접 재지는 않았다.  여기서 잰다.

측정 (세션 전체 중심화 = transductive 기준.  시간대 사이의 **상대 비교**가 목적이고,
중심화 방식이 구간마다 달라지면 그 비교가 흐려지기 때문이다):

  구간별 window 정확도        조건4 prototype, 테스트 피험자
  구간 평균 특징의 클래스 코사인  그 구간 평균이 자기 클래스 prototype 쪽으로
                             얼마나 기울어 있는가 (정답 클래스의 코사인에서
                             나머지 두 클래스 평균 코사인을 뺀 여백)

그리고 캘리브레이션 설계에 직접 쓰는 것:

  "앞 T초를 보되 처음 D초는 버리고 쓰기"  D = 0, 10, 20초, 시청 시간 T = 40/60/120초.
  시청 시간은 T 그대로이고 버리는 만큼 쓸 수 있는 창이 줄어든다 — **사용자가 앉아
  있어야 하는 시간이 비용**이므로 이것이 공정한 비교다.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
# 데이터셋은 환경변수 DATASET 으로 고른다.  기본 출력 경로를 설정에서
# 끌어오는 이유: --out 을 한 번 빠뜨리면 SEED 결과 파일을 덮어쓴다.
from dataset_config import get as _cfg_root
CFG = _cfg_root()

from scipy import stats

from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               load_head, bootstrap_ci)
from analyze_calib_protocol import _train_side, _score
from analyze_m1 import score_with_mu
from calib_common import (SEC_PER_WIN, clips_by_domain_label, draw_holdout,
                          n_windows)

BINS = [(0, 20), (20, 40), (40, 80), (80, 160), (160, 10 ** 6)]


def bin_label(lo, hi):
    return f"{lo:g}-{'끝' if hi > 1e5 else f'{hi:g}'}초"


def timecourse(cls, Z, meta, lab, cidx, ts, P_z):
    """구간별 window 정확도와 클래스 여백.  세션 전체 중심화."""
    doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
    mu = {}
    for d in doms:
        m = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1])
        mu[d] = Z[m].mean(0)
    Pn = l2(P_z)
    hit, tot, marg = defaultdict(int), defaultdict(int), defaultdict(list)
    for k in sorted(cidx):
        if k[0] != ts:
            continue
        w = cidx[k]
        y = int(lab[w[0]])
        A = Z[w] - mu[(k[0], k[1])]
        sim = l2(A) @ Pn.T
        pred = sim.argmax(1)
        for lo, hi in BINS:
            i0, i1 = int(lo / SEC_PER_WIN), min(len(w), int(hi / SEC_PER_WIN))
            if i1 <= i0:
                continue
            b = bin_label(lo, hi)
            hit[b] += int((pred[i0:i1] == y).sum())
            tot[b] += i1 - i0
            s = l2((A[i0:i1].mean(0))[None]) @ Pn.T          # bin mean feature
            marg[b].append(float(s[0, y] - np.delete(s[0], y).mean()))
    return ({b: hit[b] / tot[b] for b in tot},
            {b: float(np.mean(v)) for b, v in marg.items()},
            {b: tot[b] for b in tot})


def drop_head_sim(cls, Z, meta, lab, cidx, head, ts, P_z, mu_tr_global,
                  view_secs, drops, n_rep, rng):
    """'앞 T초를 보되 처음 D초는 버린다' — 시청 시간 T 는 그대로다."""
    doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
    by = clips_by_domain_label(cidx, lab, ts)
    out = defaultdict(list)
    for _ in range(n_rep):
        held, hs = draw_holdout(by, doms, rng)
        eval_keys = [k for k in sorted(cidx) if k[0] == ts and k not in hs]
        y_c = np.array([lab[cidx[k][0]] for k in eval_keys])
        for T in view_secs:
            nT = n_windows(T)
            for D in drops:
                nD = int(D / SEC_PER_WIN)
                if nT - nD < 1:
                    continue                 # nothing left to use
                calib = {}
                for d in doms:
                    idx = []
                    for k in held[d]:
                        idx.extend(cidx[k][nD:nT])
                    calib[d] = np.asarray(idx)
                raw = {d: cls[i].mean(0) for d, i in calib.items()}
                zz = {d: Z[i].mean(0) for d, i in calib.items()}
                # prototype path only: the head path needs a forward per window
                # and would make this sweep 10x slower for a secondary metric.
                r = score_with_mu(cls, Z, head, cidx, eval_keys, y_c, raw, zz,
                                  P_z, mu_tr_global, proto_only=True)
                for m, v in r.items():
                    out[f"T{T:g}D{D:g}|{m}"].append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prefix", default=CFG.ckpt_prefix)
    ap.add_argument("--view_secs", type=float, nargs="+", default=[40, 60, 120])
    ap.add_argument("--drops", type=float, nargs="+", default=[0, 10, 20])
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--out", default=f"results/{CFG.prefix('timecourse')}.npz")
    args = ap.parse_args()

    ACC, MAR, NW = defaultdict(list), defaultdict(list), defaultdict(list)
    DS = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)",
                                    os.path.basename(f)).groups())
        z = np.load(f)
        cls, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
        head = load_head(os.path.join(args.ckpt_dir,
                                      f"{args.prefix}S{ts}_seed{sd}.pt"))
        Z = l2(cls)
        cidx = clip_index(meta)
        train, _, _ = roles_for_fold(ts)
        mu_tr_global, P_z = _train_side(cls, meta, lab, train)
        acc, mar, nw = timecourse(cls, Z, meta, lab, cidx, ts, P_z)
        for b in acc:
            ACC[b].append(acc[b]); MAR[b].append(mar[b]); NW[b].append(nw[b])
        d = drop_head_sim(cls, Z, meta, lab, cidx, head, ts, P_z, mu_tr_global,
                          args.view_secs, args.drops, args.n_rep,
                          np.random.default_rng(1000 * ts + sd))
        for k, v in d.items():
            DS[k][ts].append(v)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  " +
              "  ".join(f"{b} {acc[b]:.3f}" for b in
                        [bin_label(*x) for x in BINS] if b in acc), flush=True)

    W = 96
    print(f"\n{'='*W}\n클립 내 시간대별 (세션 전체 중심화, 조건4, n={CFG.n_subjects} x 시드 평균)\n{'='*W}")
    print(f"  {'구간':<12}{'window 정확도':>16}{'클래스 여백':>14}{'창 수/클립':>12}")
    for lo, hi in BINS:
        b = bin_label(lo, hi)
        if b not in ACC:
            continue
        a = np.array(ACC[b]).reshape(CFG.n_subjects, -1).mean(1)
        m = np.array(MAR[b]).reshape(CFG.n_subjects, -1).mean(1)
        print(f"  {b:<12}{a.mean():>10.4f} ±{a.std(ddof=1):.3f}"
              f"{m.mean():>12.4f}{np.mean(NW[b])/45:>12.1f}")
    b0 = bin_label(*BINS[0])
    print(f"\n  첫 구간 대비 (짝지은 검정):")
    a0 = np.array(ACC[b0]).reshape(CFG.n_subjects, -1).mean(1)
    for lo, hi in BINS[1:]:
        b = bin_label(lo, hi)
        if b not in ACC:
            continue
        a = np.array(ACC[b]).reshape(CFG.n_subjects, -1).mean(1)
        d = a - a0
        lo_, hi_ = bootstrap_ci(d)
        print(f"    {b:<12} Δ{d.mean():+.4f} CI[{lo_:+.4f},{hi_:+.4f}]"
              f"{'  0포함' if lo_ <= 0 <= hi_ else '       '} "
              f"p={stats.wilcoxon(a, a0).pvalue:.4f} {int((d>0).sum())}/{len(d)}")

    def arr(k):
        return np.array([np.mean(DS[k][s]) for s in sorted(DS[k])])

    print(f"\n{'='*W}\n앞 T초를 보되 처음 D초 버리기 (시청 시간 = T, 반복 "
          f"{args.n_rep})\n{'='*W}")
    for metric in ("proto_win", "proto_clip"):
        print(f"\n  [{metric}]  {'시청 T':>8}" +
              "".join(f"{f'D={d:g}s':>22}" for d in args.drops))
        for T in args.view_secs:
            cells = []
            base = None
            for D in args.drops:
                k = f"T{T:g}D{D:g}|{metric}"
                if k not in DS:
                    cells.append(f"{'-':>22}")
                    continue
                a = arr(k)
                if D == args.drops[0]:
                    base = a
                    cells.append(f"{a.mean():>22.4f}")
                else:
                    d = a - base
                    p = stats.wilcoxon(a, base).pvalue
                    cells.append(f"{a.mean():.4f} {d.mean():+.4f} p{p:.3f}".rjust(22))
            print(f"  {'':>10}{T:>6.0f}초" + "".join(cells))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out,
             **{f"acc__{b}": np.array(ACC[b]).reshape(CFG.n_subjects, -1).mean(1) for b in ACC},
             **{f"marg__{b}": np.array(MAR[b]).reshape(CFG.n_subjects, -1).mean(1) for b in MAR},
             **{f"drop__{k}": arr(k) for k in DS})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
