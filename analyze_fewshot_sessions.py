"""
S14 의 few-shot 라벨 곡선을 **현실적인 세션 구조**에서 다시 잰다.

S14 는 세 세션(서로 다른 날)의 라벨 클립을 모아 학습했다.  실전에서는 그럴 수 없다.
두 시나리오를 따로 잰다.

  (i)  같은 날만       평가 세션의 라벨 클립만으로 학습.  첫 방문 날의 시나리오.
  (ii) 등록 후 재사용   한 세션(등록일)의 라벨 클립으로 분류기를 만들고, 다른 날에는
                      **라벨 없이** 그 세션의 캘리브레이션 창으로 중심화만 한 뒤
                      같은 분류기를 적용.  등록 순서를 바꿔 3회 반복.

방법은 (a) 라벨 없음과 (c) 규제 없는 로지스틱만 본다 — S14 에서 (b)(d) 는 (c) 의
절반 이하였다.  초매개변수(로지스틱 스텝 수)는 k 별로 검증 피험자 2명으로 고른다.

시청 시간은 두 종류로 나눠 적는다:
  라벨 캘리브레이션 (1회)        분류기를 만드는 데 드는 시간
  매 세션 라벨 없는 캘리브레이션   그날그날 중심화에 드는 시간
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

import torch
from scipy import stats

from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               bootstrap_ci)
from analyze_calib_protocol import _train_side
from analyze_fewshot import lr_fit, draw_eval_and_pool, pick_k
from calib_common import clips_by_domain_label, calib_indices, n_windows

# 이 분석의 텐서는 (수백 x 200) 으로 작다.  그 크기에서는 스레드 핸드오프가
# 계산보다 비싸다: 400 Adam 스텝이 4스레드 2.36s, 1스레드 0.44s (5.4배).
torch.set_num_threads(1)
# 라벨 상한은 analyze_centering 이 데이터셋마다 계산한 ceil_logreg 다
# (SEED 0.8464, SEED-V 0.5431).  박아 두면 분모가 틀린다.
CEILING = float(np.asarray(
    np.load(f"results/{CFG.prefix('centering_analysis')}.npz")["ceil_logreg"]
).mean())
# 감정별 클립 k 개를 쓰므로, 평가용 1편을 뺀 나머지가 상한이다
# (SEED 5-1=4, SEED-V 3-1=2).  박아 두면 SEED-V 에서 k=3,4 가 조용히
# k=2 와 같은 집합으로 떨어져 중복 수치가 결과 파일에 들어간다.
KS = list(range(1, CFG.max_k + 1))
# 스텝 수가 곧 규제다.  같은 날 시나리오는 한 세션의 클립만 쓰므로 표본이 S14 의
# 1/3 이고, 400 스텝이면 k=1 에서 무너진다 (0.681 -> 0.502 를 실제로 봤다).
# 아래쪽을 넓혀 검증이 "적게 학습" 을 고를 수 있게 한다.
STEPS = [10, 25, 50, 100, 200, 400]
SEC_PER_WIN = 4.0


def clip_windows(cidx, keys):
    return np.concatenate([cidx[k] for k in keys])


def fit_from(Z, cidx, sel_by_dom, mu, P_z, steps):
    """뽑힌 클립의 중심화된 창으로 선형 분류기.  규제 없음."""
    X, y = [], []
    for d, per in sel_by_dom.items():
        for c, keys in per.items():
            for k in keys:
                w = cidx[k]
                X.append(Z[w] - mu[d])
                y.append(np.full(len(w), c))
    return lr_fit(np.concatenate(X), np.concatenate(y), P_z, 0.0, steps=steps)


def score(Z, cidx, ev_keys, y_c, mu, P_z, W=None, b=None):
    """W 가 없으면 학습 prototype (라벨 없음), 있으면 그 분류기."""
    Pn = l2(P_z)
    pw, pc, yw = [], [], []
    for i, k in enumerate(ev_keys):
        w = cidx[k]
        A = Z[w] - mu[(k[0], k[1])]
        if W is None:
            s = l2(A) @ Pn.T
        else:
            s = A @ W.T + b
        pw.append(s.argmax(1))
        pc.append(int(s.mean(0).argmax()))
        yw.append(np.full(len(w), int(y_c[i])))
    yw = np.concatenate(yw)
    return (float((np.concatenate(pw) == yw).mean()),
            float((np.array(pc) == y_c).mean()))


# ── (i) 같은 날만 ───────────────────────────────────────────────────────────

def same_day(Z, cidx, lab, ts, P_z, steps_of, n_rep, rng):
    """평가 세션의 라벨 클립만으로 학습하고 그 세션에서 평가."""
    doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
    by = clips_by_domain_label(cidx, lab, ts)
    out = defaultdict(list)
    for _ in range(n_rep):
        ev, pool = draw_eval_and_pool(by, doms, rng)
        for k in KS:
            sel = pick_k(pool, doms, k, rng)
            for d in doms:                      # 세션마다 따로 학습·평가
                ev_keys = sorted(ev[d])
                y_c = np.array([lab[cidx[kk][0]] for kk in ev_keys])
                one = {d: sel[d]}
                idx = clip_windows(cidx, [kk for c in one[d] for kk in one[d][c]])
                mu = {d: Z[idx].mean(0)}
                w_, c_ = score(Z, cidx, ev_keys, y_c, mu, P_z)
                out[f"k{k}|a_none|win"].append(w_)
                out[f"k{k}|a_none|clip"].append(c_)
                W, b = fit_from(Z, cidx, one, mu, P_z, steps_of[k])
                w_, c_ = score(Z, cidx, ev_keys, y_c, mu, P_z, W, b)
                out[f"k{k}|c_lr|win"].append(w_)
                out[f"k{k}|c_lr|clip"].append(c_)
                out[f"k{k}|nwin"].append(float(len(idx)))
    return {kk: float(np.mean(v)) for kk, v in out.items()}


# ── (ii) 등록 후 재사용 ─────────────────────────────────────────────────────

CAL_SECS = [20.0, 40.0, "full"]


def enrol_reuse(Z, cidx, lab, ts, P_z, steps_of, n_rep, rng):
    """한 세션에서 분류기를 만들고, 다른 날에는 라벨 없이 중심화만 해서 적용."""
    doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
    by = clips_by_domain_label(cidx, lab, ts)
    out = defaultdict(list)
    for _ in range(n_rep):
        ev, pool = draw_eval_and_pool(by, doms, rng)
        for enrol in doms:                       # 등록일을 바꿔가며
            others = [d for d in doms if d != enrol]
            for k in KS:
                sel = pick_k(pool, doms, k, rng)
                one = {enrol: sel[enrol]}
                idx = clip_windows(cidx, [kk for c in one[enrol]
                                          for kk in one[enrol][c]])
                mu_e = {enrol: Z[idx].mean(0)}
                W, b = fit_from(Z, cidx, one, mu_e, P_z, steps_of[k])
                out[f"k{k}|enrol_nwin"].append(float(len(idx)))
                for sec in CAL_SECS:
                    tag = "full" if sec == "full" else f"{sec:g}"
                    for d in others:
                        ev_keys = sorted(ev[d])
                        y_c = np.array([lab[cidx[kk][0]] for kk in ev_keys])
                        # 그날의 라벨 없는 캘리브레이션: 떼어낸 N_CLS 클립의 앞 T초
                        cal = calib_indices({d: ev[d]}, cidx, [d], sec,
                                            "prefix")
                        mu = {d: Z[np.asarray(cal[d])].mean(0)}
                        w0, c0 = score(Z, cidx, ev_keys, y_c, mu, P_z)
                        out[f"k{k}|{tag}|a_none|win"].append(w0)
                        out[f"k{k}|{tag}|a_none|clip"].append(c0)
                        w1, c1 = score(Z, cidx, ev_keys, y_c, mu, P_z, W, b)
                        out[f"k{k}|{tag}|c_lr|win"].append(w1)
                        out[f"k{k}|{tag}|c_lr|clip"].append(c1)
                        out[f"k{k}|{tag}|cal_nwin"].append(
                            float(len(cal[d])))
    return {kk: float(np.mean(v)) for kk, v in out.items()}


# ── 초매개변수: k 별 스텝 수를 검증 피험자로 ────────────────────────────────

def choose_steps(Z, cidx, lab, ts, P_z, n_rep):
    train, val, _ = roles_for_fold(ts)
    out = {}
    packs = {k: [] for k in KS}
    for v in val:
        doms = sorted({(kk[0], kk[1]) for kk in cidx if kk[0] == v})
        by = clips_by_domain_label(cidx, lab, v)
        rng = np.random.default_rng(1000 * v + 7)
        for _ in range(max(2, n_rep // 5)):
            ev, pool = draw_eval_and_pool(by, doms, rng)
            for k in KS:
                sel = pick_k(pool, doms, k, rng)
                for d in doms:
                    ek = sorted(ev[d])
                    yc = np.array([lab[cidx[kk][0]] for kk in ek])
                    one = {d: sel[d]}
                    idx = clip_windows(cidx, [kk for c in one[d]
                                              for kk in one[d][c]])
                    packs[k].append((ek, yc, {d: Z[idx].mean(0)}, one))
    for k in KS:
        best, bv = STEPS[0], -1.0
        for st in STEPS:
            acc = []
            for ek, yc, mu, one in packs[k]:
                W, b = fit_from(Z, cidx, one, mu, P_z, st)
                acc.append(score(Z, cidx, ek, yc, mu, P_z, W, b)[0])
            m = float(np.mean(acc))
            if m > bv:
                bv, best = m, st
        out[k] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--n_rep", type=int, default=15)
    ap.add_argument("--out",
                    default=f"results/{CFG.prefix('fewshot_sessions')}.npz")
    args = ap.parse_args()

    R = defaultdict(lambda: defaultdict(list))
    ST = {}
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)",
                                    os.path.basename(f)).groups())
        z = np.load(f)
        cls, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
        Z = l2(cls)
        cidx = clip_index(meta)
        train, _, _ = roles_for_fold(ts)
        _, P_z = _train_side(cls, meta, lab, train)
        steps_of = choose_steps(Z, cidx, lab, ts, P_z, args.n_rep)
        ST[(ts, sd)] = steps_of
        rng = np.random.default_rng(1000 * ts + sd)
        for k, v in same_day(Z, cidx, lab, ts, P_z, steps_of, args.n_rep,
                             rng).items():
            R[f"i|{k}"][ts].append(v)
        rng = np.random.default_rng(2000 * ts + sd)
        for k, v in enrol_reuse(Z, cidx, lab, ts, P_z, steps_of, args.n_rep,
                                rng).items():
            R[f"ii|{k}"][ts].append(v)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd} steps={list(steps_of.values())}"
              f"  || (i) k{KS[0]} "
              f"{np.mean(R[f'i|k{KS[0]}|a_none|win'][ts]):.3f}->"
              f"{np.mean(R['i|k1|c_lr|win'][ts]):.3f}"
              f"  k{KS[-1]} "
              f"{np.mean(R[f'i|k{KS[-1]}|a_none|win'][ts]):.3f}->"
              f"{np.mean(R[f'i|k{KS[-1]}|c_lr|win'][ts]):.3f}", flush=True)

    def arr(key):
        return np.array([np.mean(R[key][s]) for s in sorted(R[key])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    report(arr, R, ST, args)


def cmp(a, b):
    d = a - b
    lo, hi = bootstrap_ci(d)
    return (f"{d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]"
            f"{' 0포함' if lo <= 0 <= hi else '      '} "
            f"p={stats.wilcoxon(a, b).pvalue:.4f} {int((d > 0).sum())}/{len(d)}")


def report(arr, R, ST, args):
    W = 100
    print(f"\n{'='*W}\n선택된 로지스틱 스텝 수 (검증 피험자 2명)\n{'='*W}")
    for k in KS:
        c = [s[k] for s in ST.values()]
        print(f"  k={k}  " + "  ".join(f"{v}:{c.count(v)}" for v in STEPS))

    print(f"\n{'='*W}\n(i) 같은 날만 — 평가 세션의 라벨 클립만으로 학습\n"
          f"라벨 캘리브레이션과 중심화가 **같은 클립**이다 (추가 시청 없음)\n{'='*W}")
    print(f"  {'k':<4}{'라벨없음':>10}{'로지스틱':>10}{'시청(분/세션)':>14}  이득")
    for k in KS:
        a, b = arr(f"i|k{k}|c_lr|win"), arr(f"i|k{k}|a_none|win")
        mins = arr(f"i|k{k}|nwin").mean() * SEC_PER_WIN / 60
        print(f"  {k:<4}{b.mean():>10.4f}{a.mean():>10.4f}{mins:>14.1f}  {cmp(a, b)}")
    print(f"\n  [clip]")
    for k in KS:
        a, b = arr(f"i|k{k}|c_lr|clip"), arr(f"i|k{k}|a_none|clip")
        print(f"  {k:<4}{b.mean():>10.4f}{a.mean():>10.4f}{'':>14}  {cmp(a, b)}")

    print(f"\n{'='*W}\n(ii) 등록 후 재사용 — 한 세션의 라벨로 만든 분류기를 다른 날에\n"
          f"그날은 **라벨 없이** 캘리브레이션 창으로 중심화만 한다\n{'='*W}")
    for tag, lb in (("20", "감정별 20초"), ("40", "감정별 40초"),
                    ("full", f"{N_CLS}클립 전체")):
        print(f"\n  [그날의 라벨 없는 캘리브레이션: {lb}]")
        print(f"  {'k':<4}{'라벨없음':>10}{'로지스틱':>10}"
              f"{'등록(분)':>10}{'매일(분)':>10}  이득")
        for k in KS:
            a = arr(f"ii|k{k}|{tag}|c_lr|win")
            b = arr(f"ii|k{k}|{tag}|a_none|win")
            em = arr(f"ii|k{k}|enrol_nwin").mean() * SEC_PER_WIN / 60
            cm = arr(f"ii|k{k}|{tag}|cal_nwin").mean() * SEC_PER_WIN / 60
            print(f"  {k:<4}{b.mean():>10.4f}{a.mean():>10.4f}"
                  f"{em:>10.1f}{cm:>10.1f}  {cmp(a, b)}")

    print(f"\n{'='*W}\n라벨 상한({CEILING:.4f}) 대비 회수율 — clip\n{'='*W}")
    print(f"  {'시나리오':<28}" + "".join(f"{'k='+str(k):>12}" for k in KS))
    rows = [("(i) 같은 날만", lambda k: (arr(f"i|k{k}|c_lr|clip").mean(),
                                      arr(f"i|k{k}|a_none|clip").mean()))]
    for tag, lb in (("20", "20초"), ("40", "40초"), ("full", f"{N_CLS}클립")):
        rows.append((f"(ii) 등록+재사용 [{lb}]",
                     lambda k, t=tag: (arr(f"ii|k{k}|{t}|c_lr|clip").mean(),
                                       arr(f"ii|k{k}|{t}|a_none|clip").mean())))
    for nm, fn in rows:
        cells = []
        for k in KS:
            a, b = fn(k)
            cells.append(f"{100*(a-b)/(CEILING-b):>11.1f}%")
        print(f"  {nm:<28}" + "".join(cells))
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
