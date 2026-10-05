"""
분석 E: 결정 단계의 증거 가중 — 라벨도 캘리브레이션 추가도 필요 없다.

S11 분석 C 에서 감정 신호가 클립 시작 후 시간이 지나야 올라온다는 것을 쟀다
(0–20초 window 0.515, 160초+ 0.687; 클래스 여백 0.475 → 0.935).  그런데 지금까지
클립 결정은 **모든 창을 똑같이 평균**냈다.  신호가 약한 앞부분 창이 뒤쪽 창과 같은
표를 던지고 있다.

여기서 바꾸는 것은 **결정 단계뿐**이다 — 중심화도 head 도 그대로다.

  (a) 균등 평균            지금까지의 방식
  (b) 시간 가중  w(t) = 1 − exp(−t/τ)      영상 시작 시점을 **알아야 한다**
  (c) 확신도 가중 w ∝ exp(여백/T)          시작 시점을 **몰라도 된다**
  (d) 시간 × 확신도

τ 와 온도는 각 fold 의 검증 피험자 2명으로 고른다.

**window 단위 정확도는 이 분석에서 정의상 바뀌지 않는다** — 창별 예측은 그대로이고
클립으로 모으는 방식만 바꾸기 때문이다.  한 번 확인만 하고 clip 으로 판정한다.

인과적 판정: 20/60/120초 시점까지 본 창만으로 클립을 결정할 때 (a) 와 (c) 비교.
(b)(d) 는 인과 설정에서 t 의 범위가 시점마다 달라 비교가 흐려지므로 제외한다.
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

import torch
from scipy import stats

from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               head_probs, load_head, bootstrap_ci)
from analyze_calib_protocol import _train_side
from calib_common import (SEC_PER_WIN, calib_indices, clips_by_domain_label,
                          draw_holdout)

torch.set_num_threads(4)          # see S11: tiny tensors, 28 threads is 100x slower

TAUS = [10.0, 20.0, 40.0, 80.0]
TEMPS = [0.02, 0.05, 0.1, 0.3]
CAUSAL_T = [20.0, 60.0, 120.0]


def time_weights(n, tau):
    """w(t) = 1 − exp(−t/τ), t = 창 시작 시각 (초)."""
    t = np.arange(n) * SEC_PER_WIN
    return 1.0 - np.exp(-t / tau)


def conf_weights(prob, temp):
    """여백(1등−2등) 기반.  prob 은 창별 확률 또는 코사인 유사도."""
    s = np.sort(prob, axis=1)
    m = s[:, -1] - s[:, -2]
    w = np.exp((m - m.max()) / max(temp, 1e-8))
    return w


def make_weights(n, score, mode, tau, temp):
    """창 가중.  ``score`` 는 여백을 재는 데 쓰는 창별 점수 (확률 또는 코사인)."""
    w = np.ones(n)
    if mode in ("time", "both"):
        w = w * time_weights(n, tau)
    if mode in ("conf", "both"):
        w = w * conf_weights(score, temp)
    if w.sum() < 1e-12:
        w = np.ones(n)
    return w / w.sum()


def decide_head(prob, mode, tau, temp):
    """조건2 경로: 창별 **확률**의 가중 평균.  (a) 는 조건2 그대로다."""
    w = make_weights(len(prob), prob, mode, tau, temp)
    return int((prob * w[:, None]).sum(0).argmax())


def decide_proto(A, sim, P_n, mode, tau, temp):
    """조건4 경로: 중심화된 창 **임베딩**의 가중 평균을 prototype 과 비교.

    조건4 는 확률을 평균하는 것이 아니라 클립 임베딩(중심화된 창의 평균)을 만들어
    코사인을 잰다.  가중은 그 평균에 걸어야 (a) 가 조건4 와 **정확히** 같아진다 —
    확률을 평균하면 가중이 없어도 조건4 와 다른 값이 나온다 (Tfull 0.710 대 0.730).
    """
    w = make_weights(len(A), sim, mode, tau, temp)
    e = (A * w[:, None]).sum(0)
    return int((l2(e[None]) @ P_n.T).argmax(1)[0])


def window_evidence(cls, Z, head, cidx, key, mu_raw, mu_z, P_z, mu_tr_global):
    """한 클립의 창별 증거.

    ph  (n, 3) head 확률            조건2 가 평균하는 것
    A   (n, d) 중심화된 정규화 임베딩  조건4 가 평균하는 것
    sim (n, 3) 창별 코사인            prototype 경로의 여백을 재는 데 쓴다
    """
    w = cidx[key]
    d = (key[0], key[1])
    ph = head_probs(head, cls[w] - mu_raw[d] + mu_tr_global)
    A = Z[w] - mu_z[d]
    sim = l2(A) @ l2(P_z).T
    return ph, A, sim


def run_condition(cls, Z, head, cidx, eval_keys, y_c, mu_raw, mu_z, P_z,
                  mu_tr_global, hp):
    """{(경로, 방식): clip 정확도}  +  window 정확도(불변 확인용)."""
    got = defaultdict(list)
    wacc = {"head": [], "proto": []}
    P_n = l2(P_z)
    for i, k in enumerate(eval_keys):
        ph, A, sim = window_evidence(cls, Z, head, cidx, k, mu_raw, mu_z, P_z,
                                     mu_tr_global)
        y = int(y_c[i])
        wacc["head"].append((ph.argmax(1) == y).mean())
        wacc["proto"].append((sim.argmax(1) == y).mean())
        for mode in ("uniform", "time", "conf", "both"):
            got[("head", mode)].append(
                int(decide_head(ph, mode, hp["tau"], hp["temp"]) == y))
            got[("proto", mode)].append(
                int(decide_proto(A, sim, P_n, mode, hp["tau"], hp["temp"]) == y))
    out = {f"{path}|{mode}": float(np.mean(v))
           for (path, mode), v in got.items()}
    out["head|win"] = float(np.mean(wacc["head"]))
    out["proto|win"] = float(np.mean(wacc["proto"]))
    return out


def run_causal(cls, Z, head, cidx, eval_keys, y_c, mu_raw, mu_z, P_z,
               mu_tr_global, hp):
    """지금까지 본 창만으로 클립을 결정할 때 균등 대 확신도."""
    out = defaultdict(list)
    P_n = l2(P_z)
    for i, k in enumerate(eval_keys):
        ph, A, sim = window_evidence(cls, Z, head, cidx, k, mu_raw, mu_z, P_z,
                                     mu_tr_global)
        y = int(y_c[i])
        for T in CAUSAL_T:
            n = max(1, int(round(T / SEC_PER_WIN)))
            for mode in ("uniform", "conf"):
                out[f"causal{T:g}|head|{mode}"].append(
                    int(decide_head(ph[:n], mode, hp["tau"], hp["temp"]) == y))
                out[f"causal{T:g}|proto|{mode}"].append(
                    int(decide_proto(A[:n], sim[:n], P_n, mode, hp["tau"],
                                     hp["temp"]) == y))
    return {k: float(np.mean(v)) for k, v in out.items()}


# ── centring conditions ─────────────────────────────────────────────────────

def make_mus(cls, Z, cidx, lab, ts, doms, rng, secs):
    """{조건: (mu_raw, mu_z, eval_keys, y_c)} — S10 기준선과 같은 추출."""
    by = clips_by_domain_label(cidx, lab, ts)
    held, hs = draw_holdout(by, doms, rng)
    ev = [k for k in sorted(cidx) if k[0] == ts and k not in hs]
    y = np.array([lab[cidx[k][0]] for k in ev])
    out = {}
    for sec in list(secs) + ["full"]:
        tag = "Tfull" if sec == "full" else f"T{sec:g}"
        cal = calib_indices(held, cidx, doms, sec, "prefix")
        out[tag] = ({d: cls[np.asarray(i)].mean(0) for d, i in cal.items()},
                    {d: Z[np.asarray(i)].mean(0) for d, i in cal.items()}, ev, y)
    # transductive reference: the evaluation clips themselves
    raw, zz = {}, {}
    for d in doms:
        idx = np.concatenate([cidx[k] for k in ev if (k[0], k[1]) == d])
        raw[d], zz[d] = cls[idx].mean(0), Z[idx].mean(0)
    out["trans"] = (raw, zz, ev, y)
    return out


def choose(cls, Z, meta, lab, cidx, head, ts, secs, sel_path):
    """검증 피험자 2명으로 τ 와 온도를 고른다 (방식별로 따로)."""
    train, val, _ = roles_for_fold(ts)
    mu_tr_global, P_z = _train_side(cls, meta, lab, train)
    packs = []
    for v in val:
        doms = sorted({(k[0], k[1]) for k in cidx if k[0] == v})
        conds = make_mus(cls, Z, cidx, lab, v, doms,
                         np.random.default_rng(1000 * v + 7), secs)
        for tag, (raw, zz, ev, y) in conds.items():
            if tag == "trans":
                continue                  # select on the realistic conditions
            packs.append((raw, zz, ev, y))

    # cache the per-clip window probabilities once — they do not depend on the
    # weighting, so recomputing them inside the grid would dominate the cost
    P_n = l2(P_z)
    cache = []
    for raw, zz, ev, y in packs:
        rows = []
        for i, k in enumerate(ev):
            ph, A, sim = window_evidence(cls, Z, head, cidx, k, raw, zz, P_z,
                                         mu_tr_global)
            rows.append((ph, A, sim, int(y[i])))
        cache.append(rows)

    def acc(mode, tau, temp):
        vals = []
        for rows in cache:
            hits = []
            for ph, A, sim, y in rows:
                if sel_path == "head":
                    hits.append(decide_head(ph, mode, tau, temp) == y)
                else:
                    hits.append(decide_proto(A, sim, P_n, mode, tau, temp) == y)
            vals.append(np.mean(hits))
        return float(np.mean(vals))

    hp = {"tau": TAUS[0], "temp": TEMPS[0]}
    hp["tau"] = max(TAUS, key=lambda t: acc("time", t, 0.1))
    hp["temp"] = max(TEMPS, key=lambda t: acc("conf", hp["tau"], t))
    return hp, P_z, mu_tr_global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prefix", default=CFG.ckpt_prefix)
    ap.add_argument("--secs", type=float, nargs="+", default=[20, 40])
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--sel_path", default="proto")
    ap.add_argument("--out", default=f"results/{CFG.prefix('evidence')}.npz")
    args = ap.parse_args()

    R = defaultdict(lambda: defaultdict(list))
    HP = {}
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
        hp, P_z, mg = choose(cls, Z, meta, lab, cidx, head, ts, args.secs,
                             args.sel_path)
        HP[(ts, sd)] = dict(hp)
        doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
        rng = np.random.default_rng(1000 * ts + sd)
        for _ in range(args.n_rep):
            conds = make_mus(cls, Z, cidx, lab, ts, doms, rng, args.secs)
            for tag, (raw, zz, ev, y) in conds.items():
                r = run_condition(cls, Z, head, cidx, ev, y, raw, zz, P_z,
                                  mg, hp)
                for k, v in r.items():
                    R[f"{tag}|{k}"][ts].append(v)
                if tag == "Tfull":
                    for k, v in run_causal(cls, Z, head, cidx, ev, y, raw, zz,
                                           P_z, mg, hp).items():
                        R[k][ts].append(v)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  τ={hp['tau']:g} "
              f"T={hp['temp']:g}  ||  Tfull proto 균등 "
              f"{np.mean(R['Tfull|proto|uniform'][ts]):.3f} 시간 "
              f"{np.mean(R['Tfull|proto|time'][ts]):.3f} 확신도 "
              f"{np.mean(R['Tfull|proto|conf'][ts]):.3f}", flush=True)

    def arr(k):
        return np.array([np.mean(R[k][s]) for s in sorted(R[k])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    report(arr, R, HP, args)


def report(arr, R, HP, args):
    W = 106
    print(f"\n{'='*W}\n선택된 초매개변수 (검증 피험자 2명, {args.sel_path} 경로)\n{'='*W}")
    for key, vals in (("tau", TAUS), ("temp", TEMPS)):
        c = [h[key] for h in HP.values()]
        print(f"  {key:<6}" + "  ".join(f"{v:g}:{c.count(v)}" for v in vals))

    tags = [f"T{s:g}" for s in args.secs] + ["Tfull", "trans"]
    names = {"uniform": "(a) 균등 평균", "time": "(b) 시간 가중 [시작 시점 필요]",
             "conf": "(c) 확신도 가중 [시점 불필요]", "both": "(d) 시간 x 확신도"}
    for path, pn in (("proto", "조건4 prototype"), ("head", "조건2 head")):
        print(f"\n{'='*W}\n{pn}  clip 정확도   결정 규칙 비교 -> 짝지은 검정\n{'='*W}")
        print(f"  {'방식':<30}" + "".join(
            f"{('시청 '+t[1:]+'초' if t.startswith('T') and t != 'Tfull' else {'Tfull':'클립 전체','trans':'transductive'}.get(t,t)):>21}"
            for t in tags))
        for mode in ("uniform", "time", "conf", "both"):
            cells = []
            for t in tags:
                a, b = arr(f"{t}|{path}|{mode}"), arr(f"{t}|{path}|uniform")
                if mode == "uniform":
                    cells.append(f"{a.mean():>21.4f}")
                else:
                    d = a - b
                    p = stats.wilcoxon(a, b).pvalue
                    cells.append(f"{a.mean():.4f} {d.mean():+.4f} p{p:.3f}".rjust(21))
            print(f"  {names[mode]:<30}" + "".join(cells))
        w = arr(f"Tfull|{path}|win")
        print(f"  {'(참고) window 정확도':<30}{w.mean():>21.4f}"
              "   — 가중은 클립 집계만 바꾸므로 정의상 불변")

    print(f"\n{'='*W}\n인과적 판정 (지금까지 본 창만으로 클립 결정, 클립 전체 중심화)\n{'='*W}")
    for path in ("proto", "head"):
        print(f"\n  [{path}]  {'시점':>8}{'(a) 균등':>14}{'(c) 확신도':>14}{'Δ':>10}"
              f"{'p':>9}  이긴 수")
        for T in CAUSAL_T:
            a = arr(f"causal{T:g}|{path}|conf")
            b = arr(f"causal{T:g}|{path}|uniform")
            d = a - b
            print(f"  {'':>8}{T:>6.0f}초{b.mean():>14.4f}{a.mean():>14.4f}"
                  f"{d.mean():>+10.4f}{stats.wilcoxon(a,b).pvalue:>9.4f}"
                  f"  {int((d>0).sum())}/{len(d)}")

    print(f"\n{'='*W}\n피험자별 이득 분포 (클립 전체, prototype, 최선 방식)\n{'='*W}")
    b = arr("Tfull|proto|uniform")
    for mode in ("time", "conf", "both"):
        a = arr(f"Tfull|proto|{mode}")
        d = a - b
        lo, hi = bootstrap_ci(d)
        bad = [f"S{i+1}({d[i]:+.3f})" for i in range(len(d)) if d[i] < 0]
        print(f"  {names[mode]:<30} Δ{d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]"
              f"  나빠진 {len(bad)}/{len(b)}  " + (", ".join(bad[:5]) or ""))
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
