"""V5 gate 1: SLA (자극 시점 정렬) — 사전 등록 reports/V5_SLA_GATE1_PREREG.md (13:00 고정).

새 사용자가 캘리브레이션으로 본 클립은 학습 피험자들도 본 같은 영상이다.  같은 클립·같은 시점
(클립 안 창 번호)에서 학습 피험자들의 중심화 특징을 평균한 템플릿을 대응점으로 삼아, 세션마다
직교 회전을 추정한다 (fMRI 의 response hyperalignment 를 배포 시점 캘리브레이션으로).

  center  (z − μ_d) → P                                   — V4 의 center 와 같은 경로
  L3      P'_c = l2(λ·l2(M_c) + (1−λ)·P_c)                — 감정 라벨 (부록 A 의 L3)
  SLA     (z − μ_d) W,  W = (1−β)I + β(I − UUᵀ + U R Uᵀ)   — 감정 라벨 없음
  SLA+L3  SLA 의 W 를 적용한 공간에서 L3

추출(난수·소비 순서), 학습 prototype, 중심화, 채점은 analyze_calib_rotation (V4) 와 같다 —
center 가 V4 를 피험자별로 재현하는지가 회귀 검사다.  자극 정체성은 캘리브레이션 클립에만 쓰고,
평가 클립의 정체성·시점은 쓰지 않는다.

    DATASET=seed  python analyze_sla.py --cache cache_ft_noea --out results/sla_noea.npz
    DATASET=seed  python analyze_sla.py --cache cache_ft_noea --smoke   # 회귀 + 유한성만
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import product

import numpy as np

sys.path.insert(0, os.getcwd())
from dataset_config import get as _cfg_root
from analyze_centering import N_CLS, roles_for_fold, clip_index, l2, all_domains
from analyze_calib_rotation import draw, BUDGETS, SEC
import analyze_stratnorm as SN

CFG = _cfg_root()
KS = [10, 20, 50]
BETAS = [0.25, 0.5, 0.75]
LAMS = [0.25, 0.5, 0.75, 1.0]
SLA_GRID = list(product(KS, BETAS))
# SEED 는 세 세션이 같은 15개 영상(길이·라벨 전원 동일, 2026-10-04 확인) — 템플릿을 세션 합쳐 만든다.
POOL_SESSIONS = CFG.name == "seed"


# ── 템플릿 ───────────────────────────────────────────────────────────────────

def templates(Z, meta, cidx, train):
    """{(session, clip) 또는 (clip,): (창 수, d)} — 학습 피험자의 도메인 중심화 특징, 시점별 평균."""
    mu = {}
    for d in all_domains(meta):
        if d[0] in train:
            w = np.flatnonzero((meta[:, 0] == d[0]) & (meta[:, 1] == d[1]))
            mu[d] = Z[w].mean(0)
    acc = defaultdict(list)
    for (s, a, c) in sorted(cidx):
        if s in train:
            acc[(c,) if POOL_SESSIONS else (a, c)].append(Z[cidx[(s, a, c)]] - mu[(s, a)])
    out = {}
    for key, lst in acc.items():
        lens = {len(x) for x in lst}
        assert len(lens) == 1, f"같은 클립의 창 수가 다르다 {key}: {lens} — 시점 대응이 깨진다"
        out[key] = np.mean(lst, axis=0)
    return out


def tkey(k):
    return (k[2],) if POOL_SESSIONS else (k[1], k[2])


# ── 캘리브레이션 창 (V4 와 같은 선택 + 시점) ─────────────────────────────────

def calib_windows(held, cidx, lab, d, sec, T):
    idx, y, tgt = [], [], []
    for k in held[d]:
        w = cidx[k]
        n = len(w) if sec == "full" else max(1, int(np.ceil(sec / SEC)))
        n = min(n, len(w))
        sel = np.asarray(w[:n])                       # pick_from_clip(w, n, "prefix") 와 같다
        idx.append(sel)
        y.append(np.full(n, int(lab[w[0]])))
        tgt.append(T[tkey(k)][:n])                    # 같은 클립의 시점 0..n−1 템플릿
    return np.concatenate(idx), np.concatenate(y), np.concatenate(tgt)


# ── 변환 ─────────────────────────────────────────────────────────────────────

def sla_rotation(X, Y, k, beta):
    """[X; Y] 의 상위 k 우특이벡터 부분공간에서 Procrustes(X→Y), 단위행렬 쪽 수축."""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    d = X.shape[1]
    I = np.eye(d)
    S = np.concatenate([X, Y])
    _, sv, Vt = np.linalg.svd(S, full_matrices=False)
    rank = int((sv > sv.max() * 1e-8).sum())
    U = Vt[:min(k, rank)].T                           # (d, k')
    u, _, vt = np.linalg.svd((X @ U).T @ (Y @ U))
    R = u @ vt
    if np.linalg.det(R) < 0:                          # 회전으로 고정 (m2_rotation 과 같다)
        u[:, -1] *= -1
        R = u @ vt
    W = I - U @ U.T + U @ R @ U.T
    return ((1 - beta) * I + beta * W).astype(np.float32)


def l3_protos(A, cy, P, lam):
    out = P.copy()
    for c in range(N_CLS):
        m = A[cy == c].mean(0)
        n = np.linalg.norm(m)
        if n < 1e-9:
            continue
        out[c] = lam * (m / n * np.linalg.norm(P[c])) + (1 - lam) * P[c]
    return l2(out)


def score(E, y_c, W, Pd):
    """E: [(domain, (n_k, d) 중심화된 평가 창)] — clip, window accuracy."""
    pc, ok_w, n_w = [], 0, 0
    for i, (d, X) in enumerate(E):
        Xw = X @ W[d] if W is not None else X
        P = Pd[d]
        ok_w += int(((l2(Xw) @ P.T).argmax(1) == y_c[i]).sum()); n_w += len(Xw)
        pc.append(int((l2(Xw.mean(0)[None]) @ P.T).argmax(1)[0]))
    return float((np.array(pc) == y_c).mean()), ok_w / n_w


# ── 한 피험자 ────────────────────────────────────────────────────────────────

def run_subject(Z, lab, cidx, P, T, s, doms, rng, n_rep, plan):
    """plan: {budget: {"l3": [λ..], "sla": [(k,β)..], "slal3": [(k,β,λ)..]}} → {budget: {method: [clip, win]}}"""
    by = defaultdict(list)
    for k in sorted(cidx):
        if k[0] == s:
            by[((k[0], k[1]), int(lab[cidx[k][0]]))].append(k)
    acc = defaultdict(lambda: defaultdict(list))
    for _ in range(n_rep):
        held = draw(by, doms, rng)
        hs = {k for v in held.values() for k in v}
        ev = [k for k in sorted(cidx) if k[0] == s and k not in hs]
        y_c = np.array([lab[cidx[k][0]] for k in ev])
        for tag, sec in BUDGETS.items():
            pl = plan.get(tag)
            if pl is None:
                continue
            mu, cal = {}, {}
            for d in doms:
                ci, cy, Y = calib_windows(held, cidx, lab, d, sec, T)
                mu[d] = Z[ci].mean(0)
                cal[d] = (Z[ci] - mu[d], cy, Y - Y.mean(0))
            E = [((k[0], k[1]), Z[cidx[k]] - mu[(k[0], k[1])]) for k in ev]
            Pg = {d: P for d in doms}
            res = {"center": score(E, y_c, None, Pg)}
            for lam in pl.get("l3", []):
                Pd = {d: l3_protos(cal[d][0], cal[d][1], P, lam) for d in doms}
                res[f"l3_{lam}"] = score(E, y_c, None, Pd)
            Wc = {}
            need = set(pl.get("sla", [])) | {(k_, b_) for (k_, b_, _) in pl.get("slal3", [])}
            for (k_, b_) in need:
                Wc[(k_, b_)] = {d: sla_rotation(cal[d][0], cal[d][2], k_, b_) for d in doms}
            Ew = {kb: [(d, X @ W[d]) for (d, X) in E] for kb, W in Wc.items()}   # 회전은 한 번만
            for (k_, b_) in pl.get("sla", []):
                res[f"sla_{k_}_{b_}"] = score(Ew[(k_, b_)], y_c, None, Pg)
            for (k_, b_, lam) in pl.get("slal3", []):
                W = Wc[(k_, b_)]
                Pd = {d: l3_protos(cal[d][0] @ W[d], cal[d][1], P, lam) for d in doms}
                res[f"slal3_{k_}_{b_}_{lam}"] = score(Ew[(k_, b_)], y_c, None, Pd)
            for m, v in res.items():
                acc[tag][m].append(v)
    return {t: {m: np.mean(v, axis=0) for m, v in d.items()} for t, d in acc.items()}


def load_fold(f):
    ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
    z = np.load(f)
    meta = z["meta"].astype(int); lab = z["lab"].astype(int)
    Z = l2(z["cls"]).astype(np.float32)
    cidx = clip_index(meta)
    train, val, _ = roles_for_fold(ts)
    P = SN.prototypes(Z, meta, lab, cidx, train, "center")
    T = templates(Z, meta, cidx, train)
    return ts, sd, meta, lab, Z, cidx, train, val, P, T


def doms_of(meta, s):
    return [d for d in all_domains(meta) if d[0] == s]


def smoke(args):
    """회귀(center = V4, 피험자 1 의 세 시드 평균)와 유한성만.  SLA·L3 정확도는 출력하지 않는다."""
    v4 = np.load(args.v4)
    files = sorted(glob.glob(os.path.join(args.cache, "S1_seed*.npz")))
    cen = defaultdict(list)
    for f in files:
        ts, sd, meta, lab, Z, cidx, train, val, P, T = load_fold(f)
        r = run_subject(Z, lab, cidx, P, T, ts, doms_of(meta, ts),
                        np.random.default_rng(1000 * ts + sd), args.n_rep, {t: {} for t in BUDGETS})
        for t in BUDGETS:
            cen[t].append(r[t]["center"])
    for i, t in enumerate(BUDGETS):
        c = np.mean(cen[t], axis=0)
        ref = (v4[f"{t}__center__clip"][0], v4[f"{t}__center__win"][0])
        print(f"  [회귀] S1 {t} center clip {c[0]:.6f} vs V4 {ref[0]:.6f}  win {c[1]:.6f} vs V4 {ref[1]:.6f}  "
              f"{'일치' if np.allclose(c, ref, atol=1e-9) else '불일치!'}")
    ts, sd, meta, lab, Z, cidx, train, val, P, T = load_fold(files[0])
    plan = {t: {"l3": LAMS, "sla": SLA_GRID, "slal3": [(10, 0.5, 0.5)]} for t in BUDGETS}
    r = run_subject(Z, lab, cidx, P, T, ts, doms_of(meta, ts), np.random.default_rng(1), 1, plan)
    fin = all(np.all(np.isfinite(v)) for d in r.values() for v in d.values())
    print(f"  [유한성] 방법 {len(next(iter(r.values())))}개 × 예산 {len(r)}개 — 모두 유한: {fin}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--n_rep_val", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--v4", help="회귀 기준 V4 npz (기본: 입력 팔에 맞춰 자동)")
    args = ap.parse_args()
    if args.v4 is None:
        arm = "noea" if "noea" in args.cache else "ea"
        args.v4 = f"results/{CFG.prefix(f'calib_rotation_{arm}')}.npz"
    if args.smoke:
        smoke(args)
        return
    assert args.out, "--out 필요"

    R = defaultdict(lambda: defaultdict(list))
    HP = []
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    full_val = {t: {"l3": LAMS, "sla": SLA_GRID,
                    "slal3": [(k_, b_, lam) for (k_, b_) in SLA_GRID for lam in LAMS]} for t in BUDGETS}
    for j, f in enumerate(files, 1):
        ts, sd, meta, lab, Z, cidx, train, val, P, T = load_fold(f)
        # 검증 피험자로 선택 (예산마다, window accuracy)
        vs = defaultdict(lambda: defaultdict(list))
        for v in val:
            rv = run_subject(Z, lab, cidx, P, T, v, doms_of(meta, v),
                             np.random.default_rng(100000 + 1000 * v + sd), args.n_rep_val, full_val)
            for t in BUDGETS:
                for m, val_ in rv[t].items():
                    vs[t][m].append(val_[1])
        chosen = {}
        for t in BUDGETS:
            lam = max(LAMS, key=lambda x: np.mean(vs[t][f"l3_{x}"]))
            kb = max(SLA_GRID, key=lambda g: np.mean(vs[t][f"sla_{g[0]}_{g[1]}"]))
            lam2 = max(LAMS, key=lambda x: np.mean(vs[t][f"slal3_{kb[0]}_{kb[1]}_{x}"]))
            chosen[t] = {"l3": lam, "sla": kb, "slal3_lam": lam2}
        HP.append({"ts": ts, "sd": sd, **{t: {"l3": c["l3"], "sla": list(c["sla"]), "slal3_lam": c["slal3_lam"]}
                                         for t, c in chosen.items()}})
        plan = {t: {"l3": [c["l3"]], "sla": [c["sla"]], "slal3": [(*c["sla"], c["slal3_lam"])]}
                for t, c in chosen.items()}
        rt = run_subject(Z, lab, cidx, P, T, ts, doms_of(meta, ts),
                         np.random.default_rng(1000 * ts + sd), args.n_rep, plan)
        for t, c in chosen.items():
            k_, b_ = c["sla"]
            for nm, key in (("center", "center"), ("l3", f"l3_{c['l3']}"), ("sla", f"sla_{k_}_{b_}"),
                            ("slal3", f"slal3_{k_}_{b_}_{c['slal3_lam']}")):
                R[f"{t}|{nm}|clip"][ts].append(rt[t][key][0])
                R[f"{t}|{nm}|win"][ts].append(rt[t][key][1])
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  선택 " + "  ".join(
            f"{t}: λ{c['l3']} k{c['sla'][0]}β{c['sla'][1]} λ'{c['slal3_lam']}" for t, c in chosen.items()),
            flush=True)

    arr = lambda k: np.array([np.mean(R[k][s]) for s in sorted(R[k])])
    np.savez(args.out, subjects=np.array(sorted(R["T20|center|clip"])),
             **{k.replace("|", "__"): arr(k) for k in R})
    with open(args.out.replace(".npz", "_hp.json"), "w") as fh:
        json.dump(HP, fh)
    print(f"\n[저장] {args.out}")
    for t in BUDGETS:
        print(f"  선택 {t}: L3 λ {dict(Counter(h[t]['l3'] for h in HP))}  "
              f"SLA (k,β) {dict(Counter(tuple(h[t]['sla']) for h in HP))}  "
              f"SLA+L3 λ {dict(Counter(h[t]['slal3_lam'] for h in HP))}")
    # 회귀 검사: center 가 V4 와 피험자별로 같은가
    v4 = np.load(args.v4)
    for t in BUDGETS:
        for m in ("clip", "win"):
            a, b = arr(f"{t}|center|{m}"), v4[f"{t}__center__{m}"]
            print(f"  [회귀] {t} {m}: center vs V4 최대 차이 {np.abs(a - b).max():.2e}")


if __name__ == "__main__":
    main()
