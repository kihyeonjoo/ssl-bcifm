"""
EA 없음 파이프라인의 최종 사다리 — S13 과 같은 프로토콜, 캐시만.

S13 은 EA 를 현실적으로(캘리브레이션 창에서만 R̄ 추정) 쓴 사다리를 냈다.  여기서는
**EA 를 아예 안 쓴** 팔에 같은 프로토콜을 적용한다.  EA 가 꺼져 있으면 캘리브레이션이
백본 입력을 바꾸지 않으므로 GPU 재인코딩이 필요 없다 — `cache_ft_noea` 로 충분하다.

단계: EA 없음·중심화 없음·균등 -> + 중심화 -> + 시간 가중.
τ 는 각 fold 의 검증 피험자 2명으로 고른다 (S13 과 같은 방식).
클립 추출은 `1000*ts+sd` 시드로 S8/S9/S13 과 **같은 순서**를 쓴다.

마지막에 최종 구성끼리 비교한다:
  [EA 현실적 + 중심화 + 시간 가중]  vs  [EA 없음 + 중심화 + 시간 가중]
이것은 **다른 학습 실행끼리**의 비교(재학습 비교)이므로 짝지은 검정과 함께
최소 감지 효과를 병기한다 (SEED +0.0386, 그 외는 비교마다 계산).
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
from aggregate_logits import mde_from
from dataset_config import get as _cfg_root
CFG = _cfg_root()

import torch
from scipy import stats

from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               head_probs, load_head, bootstrap_ci)
from analyze_calib_protocol import _train_side, _score, extra_metrics
from analyze_evidence import decide_head, decide_proto, choose as choose_tau
from calib_common import calib_indices, clips_by_domain_label, draw_holdout

torch.set_num_threads(4)
# SEED 는 A팔 확정값(0.0386), 다른 데이터셋은 비교마다 짝지은 차이에서 계산한다.
MDE = CFG.mde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir + "_noea")
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prefix", default=CFG.noea_ckpt_prefix)
    ap.add_argument("--seconds", type=float, nargs="+", default=[20, 40])
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--out", default=f"results/{CFG.prefix('noea_ladder')}.npz")
    args = ap.parse_args()

    R = defaultdict(lambda: defaultdict(list))
    TAU = {}
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    conds = [f"T{s:g}" for s in args.seconds] + ["Tfull"]
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
        mu_tr_global, P = _train_side(cls, meta, lab, train)
        Pn = l2(P)
        hp, _, _ = choose_tau(cls, Z, meta, lab, cidx, head, ts,
                              args.seconds, "proto")
        tau = hp["tau"]
        TAU[(ts, sd)] = tau

        doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
        by = clips_by_domain_label(cidx, lab, ts)
        rng = np.random.default_rng(1000 * ts + sd)
        for _ in range(args.n_rep):
            held, hs = draw_holdout(by, doms, rng)
            ev = [k for k in sorted(cidx) if k[0] == ts and k not in hs]
            y = np.array([lab[cidx[k][0]] for k in ev])

            # 1단계: 중심화 없음, 균등 집계 (조건1/3)
            zero = {d: np.zeros(cls.shape[1], np.float32) for d in doms}
            r = _score(cls, Z, head, cidx, ev, y, zero, zero, P,
                       mu_tr_global, centred=False)
            for m, v in r.items():
                R[f"none|{m}"][ts].append(v)

            for sec, cond in zip(list(args.seconds) + ["full"], conds):
                cal = calib_indices(held, cidx, doms, sec, "prefix")
                mu_raw = {d: cls[np.asarray(i)].mean(0) for d, i in cal.items()}
                mu_z = {d: Z[np.asarray(i)].mean(0) for d, i in cal.items()}
                # 2단계: + 중심화, 균등 집계
                r = _score(cls, Z, head, cidx, ev, y, mu_raw, mu_z, P,
                           mu_tr_global, centred=True)
                for m, v in r.items():
                    R[f"cen_{cond}|{m}"][ts].append(v)
                # 3단계: + 시간 가중 (클립 집계만 바뀐다)
                ph_p, pp_p = [], []
                for i2, k2 in enumerate(ev):
                    w2 = cidx[k2]
                    d2 = (k2[0], k2[1])
                    ph = head_probs(head, cls[w2] - mu_raw[d2] + mu_tr_global)
                    A2 = Z[w2] - mu_z[d2]
                    sim2 = l2(A2) @ Pn.T
                    ph_p.append(decide_head(ph, "time", tau, 0.3))
                    pp_p.append(decide_proto(A2, sim2, Pn, "time", tau, 0.3))
                ph_p, pp_p = np.array(ph_p), np.array(pp_p)
                R[f"tw_{cond}|head_clip"][ts].append(float((ph_p == y).mean()))
                R[f"tw_{cond}|proto_clip"][ts].append(float((pp_p == y).mean()))
                for k3, v3 in extra_metrics(ph_p, y, "head_clip").items():
                    R[f"tw_{cond}|{k3}"][ts].append(v3)
                for k3, v3 in extra_metrics(pp_p, y, "proto_clip").items():
                    R[f"tw_{cond}|{k3}"][ts].append(v3)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd} tau={tau:g}  " + "  ".join(
            f"{c} {np.mean(R[f'cen_{c}|proto_clip'][ts]):.3f}"
            f"/{np.mean(R[f'tw_{c}|proto_clip'][ts]):.3f}" for c in conds),
            flush=True)

    def arr(k):
        return np.array([np.mean(R[k][s]) for s in sorted(R[k])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    report(arr, conds, TAU, args)


def cmp(a, b):
    d = a - b
    lo, hi = bootstrap_ci(d)
    return (f"{d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]"
            f"{' 0포함' if lo <= 0 <= hi else '      '} "
            f"p={stats.wilcoxon(a, b).pvalue:.4f} {int((d > 0).sum())}/{len(d)}")


def report(arr, conds, TAU, args):
    W = 100
    ts_ = [v for v in TAU.values()]
    print(f"\n{'='*W}\n선택된 τ: " + "  ".join(
        f"{t:g}:{ts_.count(t)}" for t in sorted(set(ts_))) + f"  (총 {len(ts_)})")
    lbl = {f"T{s:g}": f"시청 {3*s:g}초" for s in args.seconds}
    lbl["Tfull"] = "시청 ~11분"
    for path in ("proto", "head"):
        pn = "조건4 prototype" if path == "proto" else "조건2 head"
        print(f"\n{'='*W}\nEA 없음 사다리 — {pn}\n{'='*W}")
        for mt in (f"{path}_clip", f"{path}_win"):
            print(f"\n  [{mt}]")
            s1 = arr(f"none|{mt}")
            print(f"    1 중심화 없음·균등        {s1.mean():.4f} ± {s1.std(ddof=1):.4f}")
            for c in conds:
                s2 = arr(f"cen_{c}|{mt}")
                line = f"    2 + 중심화 ({lbl[c]:<10}) {s2.mean():.4f}   {cmp(s2, s1)}"
                print(line)
                if mt.endswith("clip"):
                    s3 = arr(f"tw_{c}|{mt}")
                    print(f"    3 + 시간 가중              {s3.mean():.4f}   {cmp(s3, s2)}")
            if mt.endswith("win"):
                print("    (시간 가중은 클립 집계만 바꾸므로 window 에는 영향 없음)")

    try:
        E = np.load("results/final_ladder.npz")
    except FileNotFoundError:
        print("\n  results/final_ladder.npz 없음 — EA 비교 생략")
        return
    print(f"\n{'='*W}\n최종 구성끼리: [EA 현실적 + 중심화 + 시간가중] vs "
          f"[EA 없음 + 중심화 + 시간가중]\n**재학습 비교** — 최소 감지 효과 "
          + (f"{MDE:+.4f} 병기" if MDE is not None
             else "비교마다 짝지은 차이에서 계산") + f"\n{'='*W}")
    for path in ("proto", "head"):
        for c in conds:
            a = E[f"eaRealTW_{c}__{path}_clip"]
            b = arr(f"tw_{c}|{path}_clip")
            d = a - b
            _mde = MDE if MDE is not None else mde_from(d)
            mark = (f"점추정 한계({_mde:+.4f}) "
                    + ("넘음" if d.mean() > _mde else "아래"))
            print(f"  {path:<6} {lbl[c]:<12} EA {a.mean():.4f}  EA없음 {b.mean():.4f}"
                  f"   {cmp(a, b)}  [{mark}]")
    print(f"\n{'='*W}\nwindow 도 함께 (시간 가중 전 단계 기준)\n{'='*W}")
    for path in ("proto", "head"):
        for c in conds:
            a = E[f"eaReal_{c}__{path}_win"]
            b = arr(f"cen_{c}|{path}_win")
            d = a - b
            _mde = MDE if MDE is not None else mde_from(d)
            mark = (f"점추정 한계({_mde:+.4f}) "
                    + ("넘음" if d.mean() > _mde else "아래"))
            print(f"  {path:<6} {lbl[c]:<12} EA {a.mean():.4f}  EA없음 {b.mean():.4f}"
                  f"   {cmp(a, b)}  [{mark}]")
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
