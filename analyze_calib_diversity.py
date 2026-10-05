"""
분석 A: 감정 다양성인가, 영상 다양성인가?

S8 분석 2 에서 캘리브레이션용 클립(감정별 1개)은 총효과의 75% 밖에 회수하지 못했다.
가설: 약한 이유는 **영상 수가 적어서**다 — 감정은 세 가지 다 들어 있었으므로, 남은
설명은 클립(영상) 다양성이다.

설계 — S8 분석 2 와 역할을 뒤집는다:
  각 테스트 세션에서 **평가용으로 감정별 1클립씩** 떼고(SEED 3, SEED-V 5),
  남은 클립을 **캘리브레이션 풀**로 쓴다.  평가 세트가 작으므로
  window 단위를 주 지표로 삼고 clip 은 참고로 둔다.

  총 예산 B (24, 60, 120초) 안에서 감정당 클립 수 k (1, 2, 3, 4) 를 바꾸고,
  각 클립에서 B/(3k) 초씩 뽑는다.  k 가 커지면 클립당 시간은 줄고 클립 수는 늘어
  **총량은 같다** — 차이는 오직 다양성이다.
  창이 4초 단위이므로 B/(3k) 는 올림되고 실제 총량을 함께 보고한다.

  추가로 최대 k 에서 각 클립 전체 사용 (예산 제한 없음) = 풀 전체.

떼어낼 평가 클립 조합 20회 무작위 반복.  A팔 캐시만, GPU 없음.
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
from dataset_config import get as _cfg_root
CFG = _cfg_root()

from scipy import stats

from analyze_centering import (N_CLS, FS, SEG, roles_for_fold, clip_index, l2,
                               clip_reduce, all_domains, domain_means,
                               head_probs, load_head, bootstrap_ci)
from analyze_calib_protocol import _train_side, _score

SEC_PER_WIN = SEG / FS                      # 4.0 s


def run_one(cls, meta, lab, head, test_subj, budgets, ks, n_rep, rng):
    train, _, _ = roles_for_fold(test_subj)
    mu_tr_global, P = _train_side(cls, meta, lab, train)
    cidx = clip_index(meta)
    Zall = l2(cls)
    te_doms = [d for d in all_domains(meta) if d[0] == test_subj]

    by = defaultdict(list)
    for k in sorted(cidx):
        if k[0] == test_subj:
            by[((k[0], k[1]), int(lab[cidx[k][0]]))].append(k)

    acc, used = defaultdict(list), defaultdict(list)
    for _ in range(n_rep):
        # one EVALUATION clip per emotion per session; the other 12 calibrate
        ev, pool = {}, {}
        for d in te_doms:
            ev[d] = [by[(d, c)][rng.integers(len(by[(d, c)]))]
                     for c in range(N_CLS)]
            pool[d] = {c: [k for k in by[(d, c)] if k not in ev[d]]
                       for c in range(N_CLS)}
        eval_keys = sorted(k for v in ev.values() for k in v)
        y_c = np.array([lab[cidx[k][0]] for k in eval_keys])

        def means_from(picker):
            raw, zz = {}, {}
            tot = 0
            for d in te_doms:
                w = picker(d)
                raw[d] = cls[w].mean(0)
                zz[d] = Zall[w].mean(0)
                tot += len(w)
            return raw, zz, tot / len(te_doms) * SEC_PER_WIN

        # no centering
        r = _score(cls, Zall, head, cidx, eval_keys, y_c, None, None, P,
                   mu_tr_global, centred=False)
        for m, v in r.items():
            acc[f"none|{m}"].append(v)

        for B in budgets:
            for k in ks:
                if any(len(pool[d][c]) < k for d in te_doms
                       for c in range(N_CLS)):
                    continue
                # Hold the budget EXACTLY: spend n_total windows, spread as
                # evenly as possible over the 3k clips.  Rounding B/(3k) up per
                # clip instead would let a larger k spend more total time, and
                # the comparison would no longer isolate diversity.
                n_total = max(1, int(round(B / SEC_PER_WIN)))
                if n_total < N_CLS * k:
                    continue                # budget cannot reach k clips
                base, rem = divmod(n_total, N_CLS * k)

                def pick(d, k=k, base=base, rem=rem):
                    out, j = [], 0
                    for c in range(N_CLS):
                        # first k clips of that emotion, deterministic given ev
                        for key in sorted(pool[d][c])[:k]:
                            n = base + (1 if j < rem else 0)
                            out.append(cidx[key][:n])
                            j += 1
                    return np.concatenate(out)
                raw, zz, sec = means_from(pick)
                r = _score(cls, Zall, head, cidx, eval_keys, y_c, raw, zz, P,
                           mu_tr_global, centred=True)
                for m, v in r.items():
                    acc[f"B{B:g}k{k}|{m}"].append(v)
                used[f"B{B:g}k{k}"].append(sec)

        # max k, whole clips: the entire calibration pool
        raw, zz, sec = means_from(lambda d: np.concatenate(
            [cidx[key] for c in range(N_CLS) for key in pool[d][c]]))
        r = _score(cls, Zall, head, cidx, eval_keys, y_c, raw, zz, P,
                   mu_tr_global, centred=True)
        for m, v in r.items():
            acc[f"full12|{m}"].append(v)
        used["full12"].append(sec)

    return ({k: float(np.mean(v)) for k, v in acc.items()},
            {k: float(np.mean(v)) for k, v in used.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prefix", default=CFG.ckpt_prefix)
    ap.add_argument("--budgets", type=float, nargs="+", default=[24, 60, 120])
    # 감정별 클립 수의 상한은 평가용 1편을 뺀 나머지 (SEED 4, SEED-V 2)
    ap.add_argument("--ks", type=int, nargs="+",
                    default=list(range(1, CFG.max_k + 1)))
    ap.add_argument("--n_rep", type=int, default=20)
    ap.add_argument("--out", default=f"results/{CFG.prefix('calib_diversity')}.npz")
    args = ap.parse_args()

    R = defaultdict(lambda: defaultdict(list))
    U = defaultdict(list)
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)",
                                    os.path.basename(f)).groups())
        z = np.load(f)
        cls, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
        head = load_head(os.path.join(args.ckpt_dir,
                                      f"{args.prefix}S{ts}_seed{sd}.pt"))
        r, u = run_one(cls, meta, lab, head, ts, args.budgets, args.ks,
                       args.n_rep, np.random.default_rng(1000 * ts + sd))
        for k, v in r.items():
            R[k][ts].append(v)
        for k, v in u.items():
            U[k].append(v)
        # a budget cannot always reach every k, so report the widest pair
        # that actually ran at the largest budget
        Bmax = f"B{args.budgets[-1]:g}"
        feas = [k for k in args.ks if f"{Bmax}k{k}|proto_win" in r]
        lo, hi = (feas[0], feas[-1]) if feas else (None, None)
        extra = (f"{Bmax}k{lo} {r[f'{Bmax}k{lo}|proto_win']:.3f}  "
                 f"{Bmax}k{hi} {r[f'{Bmax}k{hi}|proto_win']:.3f}  "
                 if feas else "")
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  없음 "
              f"{r['none|proto_win']:.3f}  {extra}"
              f"full12 {r['full12|proto_win']:.3f}", flush=True)

    def arr(key):
        return np.array([np.mean(R[key][s]) for s in sorted(R[key])])

    W = 104
    for metric, nm in (("proto_win", "조건4 prototype  window  (주 지표)"),
                       ("head_win", "조건2 head  window  (주 지표)"),
                       ("proto_clip", f"조건4 prototype  clip  (참고, 세션당 {N_CLS}클립)"),
                       ("head_clip", "조건2 head  clip  (참고)")):
        base = arr(f"none|{metric}")
        top = arr(f"full12|{metric}")
        print(f"\n{'='*W}\n{nm}   n={CFG.n_subjects}, 시드 평균 x 반복 {args.n_rep}\n"
              f"중심화 없음 {base.mean():.4f}   풀 전체 {top.mean():.4f}\n{'='*W}")
        print(f"  {'예산':>8} {'감정당 클립 k':>12} {'실제 총량':>10} "
              f"{'정확도':>9} {'없음 대비':>10} {'k=1 대비':>10} {'p(vs k=1)':>11} 이긴 수")
        for B in args.budgets:
            k1 = f"B{B:g}k1"
            if f"{k1}|{metric}" not in R:
                continue
            a1 = arr(f"{k1}|{metric}")
            for k in args.ks:
                key = f"B{B:g}k{k}"
                if f"{key}|{metric}" not in R:
                    continue
                a = arr(f"{key}|{metric}")
                d1 = a - a1
                try:
                    p = (stats.wilcoxon(a, a1).pvalue if k != 1
                         else float("nan"))
                except ValueError:
                    p = float("nan")
                print(f"  {B:>7.0f}s {k:>12} {np.mean(U[key]):>9.0f}s "
                      f"{a.mean():>9.4f} {a.mean()-base.mean():>+10.4f} "
                      f"{d1.mean():>+10.4f} {p:>11.4f}"
                      f"  {int((d1>0).sum()) if k != 1 else '-'}/{len(d1)}")
        a = arr(f"full12|{metric}")
        print(f"  {'제한없음':>8} {'4 (전체)':>12} {np.mean(U['full12']):>9.0f}s "
              f"{a.mean():>9.4f} {a.mean()-base.mean():>+10.4f}")

    print(f"\n{'='*W}\n짝지은 비교 (조건4 window)\n{'='*W}")

    def cmp(ka, kb, label):
        a, b = arr(f"{ka}|proto_win"), arr(f"{kb}|proto_win")
        d = a - b
        lo, hi = bootstrap_ci(d)
        try:
            p = stats.wilcoxon(a, b).pvalue
        except ValueError:
            p = float("nan")
        print(f"  {label:<54} Δ{d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]"
              f"{'  0포함' if lo <= 0 <= hi else '       '} p={p:.4f} "
              f"{int((d > 0).sum())}/{len(d)}")

    kmax = max(args.ks)
    for B in args.budgets:
        feas = [k for k in args.ks if f"B{B:g}k{k}|proto_win" in R]
        if len(feas) >= 2:
            cmp(f"B{B:g}k{feas[-1]}", f"B{B:g}k{feas[0]}",
                f"예산 {B:g}s 안에서  감정당 {feas[-1]}클립 vs {feas[0]}클립 (총량 동일)")
    for B in args.budgets:
        feas = [k for k in args.ks if f"B{B:g}k{k}|proto_win" in R]
        if feas:
            cmp("full12", f"B{B:g}k{feas[-1]}",
                f"12클립 전체  vs 예산 {B:g}s 감정당 {feas[-1]}클립")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R},
             **{f"usedsec__{k}": np.mean(v) for k, v in U.items()})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
