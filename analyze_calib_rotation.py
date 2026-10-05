"""V4: 캘리브레이션 자극 라벨로 세션별 회전 (CR).  사전 등록 reports/V4_CALIB_ROTATION_PREREG.md.

중심화에 쓰는 바로 그 캘리브레이션 창(감정별 한 편, 앞 T 초)의 **자극 라벨**로, prototype
부분공간 안의 Procrustes 회전을 추정한다.  adapt_methods.m2_rotation 을 그대로 쓰고 라벨만
바꾼다 (M2: 평가 클립 의사라벨 → CR: 캘리브레이션 자극 라벨).  추가 시청 시간 0.

테스트 피험자의 난수 소비는 analyze_calib_protocol / analyze_stratnorm 과 같다 — 기준선
(중심화만)이 그쪽 수치를 재현하는지가 회귀 검사다.
"""
from __future__ import annotations

import argparse, glob, os, re, sys
from collections import defaultdict
from itertools import product

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
from dataset_config import get as _cfg_root
from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               all_domains, bootstrap_ci)
from calib_common import pick_from_clip
from adapt_methods import m2_rotation
import analyze_stratnorm as SN

CFG = _cfg_root()
SEC = CFG.seg / CFG.fs
KEXTRA = [0, 2, 5]
BETAS = [0.25, 0.5, 1.0]
BUDGETS = {"T20": 20, "T40": 40, "Tfull": "full"}


def draw(by, doms, rng):
    """감정별 한 클립을 캘리브레이션으로 — calib_protocol 과 같은 소비 순서."""
    return {d: [by[(d, c)][rng.integers(len(by[(d, c)]))] for c in range(N_CLS)]
            for d in doms}


def calib_windows(held, cidx, lab, d, sec):
    idx, y = [], []
    for k in held[d]:
        w = cidx[k]
        n = len(w) if sec == "full" else max(1, int(np.ceil(sec / SEC)))
        sel = np.asarray(pick_from_clip(w, n, "prefix"))
        idx.append(sel); y.append(np.full(len(sel), int(lab[w[0]])))
    return np.concatenate(idx), np.concatenate(y)


def score(Z, cidx, eval_keys, y_c, mu, W, P):
    pw, pc, yw = [], [], []
    for i, k in enumerate(eval_keys):
        d = (k[0], k[1])
        X = (Z[cidx[k]] - mu[d]) @ W[d]
        pw.append((l2(X) @ P.T).argmax(1))
        pc.append(int((l2(X.mean(0)[None]) @ P.T).argmax(1)[0]))
        yw.append(np.full(len(cidx[k]), int(y_c[i])))
    pw, yw, pc = np.concatenate(pw), np.concatenate(yw), np.array(pc)
    return float((pc == y_c).mean()), float((pw == yw).mean())


def run_subject(Z, meta, lab, cidx, P, s, rng, n_rep, hps):
    """피험자 s 를 테스트처럼: {budget: {method: [clip, win] 평균}}."""
    doms = [d for d in all_domains(meta) if d[0] == s]
    by = defaultdict(list)
    for k in sorted(cidx):
        if k[0] == s:
            by[((k[0], k[1]), int(lab[cidx[k][0]]))].append(k)
    I = np.eye(Z.shape[1], dtype=np.float32)
    acc = defaultdict(lambda: defaultdict(list))
    for _ in range(n_rep):
        held = draw(by, doms, rng)
        hs = {k for v in held.values() for k in v}
        ev = [k for k in sorted(cidx) if k[0] == s and k not in hs]
        y_c = np.array([lab[cidx[k][0]] for k in ev])
        for tag, sec in BUDGETS.items():
            mu, cal = {}, {}
            for d in doms:
                ci, cy = calib_windows(held, cidx, lab, d, sec)
                mu[d] = Z[ci].mean(0)
                cal[d] = (Z[ci] - mu[d], cy)
            res = {"center": score(Z, cidx, ev, y_c, mu, {d: I for d in doms}, P)}
            for (ke, b) in hps.get(tag, []):
                W = {}
                for d in doms:
                    A, cy = cal[d]
                    W[d], _ = m2_rotation(A, cy, P, extra=A, k_extra=ke, beta=b)
                res[f"cr_{ke}_{b}"] = score(Z, cidx, ev, y_c, mu, W, P)
            if "oracle" in hps:
                W = {}
                for d in doms:
                    ew = np.concatenate([cidx[k] for k in ev if (k[0], k[1]) == d])
                    ey = np.concatenate([np.full(len(cidx[k]), int(lab[cidx[k][0]]))
                                         for k in ev if (k[0], k[1]) == d])
                    A = Z[ew] - mu[d]
                    W[d], _ = m2_rotation(A, ey, P, extra=A, k_extra=5, beta=1.0)
                res["oracle"] = score(Z, cidx, ev, y_c, mu, W, P)
            for m, v in res.items():
                acc[tag][m].append(v)
    return {t: {m: np.mean(v, axis=0) for m, v in d.items()} for t, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--n_rep_val", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grid = list(product(KEXTRA, BETAS))
    R = defaultdict(lambda: defaultdict(list))
    HP = defaultdict(list)
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
        z = np.load(f)
        meta = z["meta"].astype(int); lab = z["lab"].astype(int)
        Z = l2(z["cls"]).astype(np.float32); cidx = clip_index(meta)
        train, val, _ = roles_for_fold(ts)
        P = SN.prototypes(Z, meta, lab, cidx, train, "center")

        # 검증 피험자로 초매개변수 선택 (예산마다)
        vscore = defaultdict(lambda: defaultdict(list))
        for v in val:
            rv = run_subject(Z, meta, lab, cidx, P, v,
                             np.random.default_rng(100000 + 1000 * v + sd),
                             args.n_rep_val, {t: grid for t in BUDGETS})
            for t in BUDGETS:
                for (ke, b) in grid:
                    vscore[t][(ke, b)].append(rv[t][f"cr_{ke}_{b}"][0])
        chosen = {t: max(grid, key=lambda g: np.mean(vscore[t][g])) for t in BUDGETS}

        # 테스트 피험자 — 난수 소비는 calib_protocol 과 동일
        rt = run_subject(Z, meta, lab, cidx, P, ts, np.random.default_rng(1000 * ts + sd),
                         args.n_rep, {**{t: [chosen[t]] for t in BUDGETS}, "oracle": True})
        for t in BUDGETS:
            ke, b = chosen[t]
            for nm, key in (("center", "center"), ("cr", f"cr_{ke}_{b}"), ("oracle", "oracle")):
                R[f"{t}|{nm}|clip"][ts].append(rt[t][key][0])
                R[f"{t}|{nm}|win"][ts].append(rt[t][key][1])
            HP[t].append(chosen[t])
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  " + "  ".join(
            f"{t} c/cr/or {rt[t]['center'][0]:.3f}/{rt[t][f'cr_{chosen[t][0]}_{chosen[t][1]}'][0]:.3f}"
            f"/{rt[t]['oracle'][0]:.3f}" for t in BUDGETS), flush=True)

    arr = lambda k: np.array([np.mean(R[k][s]) for s in sorted(R[k])])
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    print(f"\n[저장] {args.out}")
    for t in BUDGETS:
        from collections import Counter
        print(f"  선택된 (k_extra, β) {t}: {dict(Counter(HP[t]))}")


if __name__ == "__main__":
    main()
