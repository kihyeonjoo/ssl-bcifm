"""DEAP 캘리브레이션 분석 (2026-10-05) — SEED 계열 분석 (calib_protocol · calib_rotation · sla) 의 DEAP 판.

SEED 계열과 다른 점은 둘뿐이다.

1. **캘리브레이션 영상 고르기.**  SEED 계열은 영상 = 감정이라 감정별 한 편을 테스트 피험자 라벨로 골랐다.  DEAP 의
   라벨은 본인 평점이라 실험자가 미리 알 수 없다 → **학습 피험자들의 평균 평점으로 정한 영상별 사분면 (설계 라벨)**
   에서 한 편씩, 4편을 고른다.  라벨을 쓰는 방법 (라벨 회전 · prototype 혼합) 은 사용자가 그 4편을 보고 매긴
   **본인 평점** 을 쓴다 (보고 나서 평점을 매기는 것은 실사용에서 가능하다).  본인 평점으로는 빠진 사분면이 있을 수
   있다 — 라벨 회전은 그때 회전하지 않고 (m2_rotation 의 단위행렬), prototype 혼합은 있는 사분면만 옮긴다.
2. **지표.**  클래스가 사람마다 고르지 않아 4사분면 **균형 정확도** 를 주 지표로 하고, 같은 예측에서 이진 정서가
   (q // 2) · 각성도 (q % 2) 의 균형 정확도를 함께 낸다 (사용자 결정: 이진 V/A 는 같은 모델로 보고).

나머지 — 학습 prototype (analyze_stratnorm.prototypes 'center'), 중심화, 창 선택 (클립 앞 T 초), SLA 템플릿 · 회전,
L3, 반복 난수 (테스트 1000·ts+seed, 검증 100000+1000·v+seed) — 는 SEED 계열 코드를 그대로 쓴다.  초매개변수는
검증 피험자 2명의 **창 균형 정확도** 로 고른다 (SEED 계열은 창 정확도 — 클래스 불균형 때문에 바꿨다).

    DATASET=deap python analyze_deap.py --cache cache_ft_deap_noea --out results/deap_calib_noea.npz
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
import scipy.io as sio

sys.path.insert(0, os.getcwd())
from dataset_config import get as _cfg_root

CFG = _cfg_root()
assert CFG.name == "deap", "DATASET=deap 로 실행한다"

from analyze_centering import N_CLS, roles_for_fold, clip_index, l2, all_domains
from analyze_sla import templates, calib_windows, sla_rotation, KS, BETAS as SLA_BETAS, LAMS
from analyze_calib_rotation import KEXTRA, BETAS as CR_BETAS
from adapt_methods import m2_rotation
import analyze_stratnorm as SN

BUDGETS = {"T20": 20, "T40": 40, "Tfull": "full"}
SLA_GRID = list(product(KS, SLA_BETAS))
CR_GRID = list(product(KEXTRA, CR_BETAS))


# ── 평점과 설계 라벨 ─────────────────────────────────────────────────────────

def load_ratings(path="results/deap_ratings.npz"):
    """(32, 40, 2) 본인 평점 (정서가, 각성도) — 원본 .mat 의 labels 앞 두 열."""
    if os.path.exists(path):
        return np.load(path)["r"]
    R = np.stack([sio.loadmat(os.path.join(CFG.root, f"s{s:02d}.mat"), variable_names=["labels"])["labels"][:, :2]
                  for s in range(1, CFG.n_subjects + 1)])
    np.savez(path, r=R)
    return R


def design_classes(R, train):
    """영상 1..40 → 학습 피험자 평균 평점의 사분면 (테스트 · 검증 피험자 평점은 쓰지 않는다)."""
    mr = R[np.asarray(train) - 1].mean(0)
    return {v + 1: 2 * int(mr[v, 0] > 5) + int(mr[v, 1] > 5) for v in range(mr.shape[0])}


# ── 지표 ─────────────────────────────────────────────────────────────────────

def _bal(p, t, classes):
    rec = [(p[t == c] == c).mean() for c in classes if (t == c).any()]
    return float(np.mean(rec))


def metrics(pc, yc, pw, yw):
    out = {}
    for lvl, p, t in (("clip", pc, yc), ("win", pw, yw)):
        out[lvl] = float((p == t).mean())
        out[f"{lvl}_bal"] = _bal(p, t, range(N_CLS))
        out[f"{lvl}_vbal"] = _bal(p // 2, t // 2, (0, 1))
        out[f"{lvl}_abal"] = _bal(p % 2, t % 2, (0, 1))
    return out


def score(E, y_c, Pd):
    """E: [(domain, (n_k, d) 평가 창 특징)] → metrics.  창은 각자, 클립은 창 평균을 prototype 과 코사인."""
    pw, yw, pc = [], [], []
    for i, (d, X) in enumerate(E):
        P = Pd[d]
        pw.append((l2(X) @ P.T).argmax(1)); yw.append(np.full(len(X), int(y_c[i])))
        pc.append(int((l2(X.mean(0)[None]) @ P.T).argmax(1)[0]))
    return metrics(np.array(pc), y_c, np.concatenate(pw), np.concatenate(yw))


def l3_safe(A, cy, P, lam):
    """analyze_sla.l3_protos 와 같은 식.  캘리브레이션에 없는 사분면은 학습 prototype 그대로."""
    out = P.copy()
    for c in range(N_CLS):
        if not (cy == c).any():
            continue
        m = A[cy == c].mean(0); n = np.linalg.norm(m)
        if n < 1e-9:
            continue
        out[c] = lam * (m / n * np.linalg.norm(P[c])) + (1 - lam) * P[c]
    return l2(out)


# ── 한 피험자 ────────────────────────────────────────────────────────────────

def run_subject(Z, lab, cidx, P, T, dcls, s, doms, rng, n_rep, plan, extras=False):
    """plan: {budget: {"l3": [λ], "cr": [(ke, b)], "sla": [(k, β)], "slal3": [(k, β, λ)]}}
    → {cond: {method: {metric: 평균}}}.  extras=True 면 적응 없음 · 전달식 상한 · 한 사분면만도 낸다."""
    by = defaultdict(list)
    for k in sorted(cidx):
        if k[0] == s:
            by[((k[0], k[1]), dcls[k[2]])].append(k)          # 설계 라벨로 묶는다
    acc = defaultdict(lambda: defaultdict(list))
    for _ in range(n_rep):
        held = {d: [by[(d, c)][rng.integers(len(by[(d, c)]))] for c in range(N_CLS)] for d in doms}
        hs = {k for v in held.values() for k in v}
        ev = [k for k in sorted(cidx) if k[0] == s and k not in hs]
        y_c = np.array([lab[cidx[k][0]] for k in ev])         # 본인 평점 사분면
        Pg = {d: P for d in doms}
        if extras:
            acc["none"]["none"].append(score([((k[0], k[1]), Z[cidx[k]]) for k in ev], y_c, Pg))
            mu_tr = {d: np.concatenate([Z[cidx[k]] for k in ev if (k[0], k[1]) == d]).mean(0) for d in doms}
            acc["transductive"]["center"].append(
                score([((k[0], k[1]), Z[cidx[k]] - mu_tr[(k[0], k[1])]) for k in ev], y_c, Pg))
            for c in range(N_CLS):
                mu1 = {d: Z[cidx[held[d][c]]].mean(0) for d in doms}
                acc[f"single{c}"]["center"].append(
                    score([((k[0], k[1]), Z[cidx[k]] - mu1[(k[0], k[1])]) for k in ev], y_c, Pg))
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
            res = {"center": score(E, y_c, Pg)}
            for lam in pl.get("l3", []):
                res[f"l3_{lam}"] = score(E, y_c, {d: l3_safe(cal[d][0], cal[d][1], P, lam) for d in doms})
            for (ke, b) in pl.get("cr", []):
                fit = {d: m2_rotation(cal[d][0], cal[d][1], P, extra=cal[d][0], k_extra=ke, beta=b) for d in doms}
                r_ = score([(d, X @ fit[d][0]) for (d, X) in E], y_c, Pg)
                r_["applied"] = float(np.mean([fit[d][1] for d in doms]))   # 본인 평점이 네 사분면을 덮어 회전했나
                res[f"cr_{ke}_{b}"] = r_
            need = set(pl.get("sla", [])) | {(k_, b_) for (k_, b_, _) in pl.get("slal3", [])}
            Wc = {kb: {d: sla_rotation(cal[d][0], cal[d][2], *kb) for d in doms} for kb in need}
            Ew = {kb: [(d, X @ W[d]) for (d, X) in E] for kb, W in Wc.items()}
            for kb in pl.get("sla", []):
                res[f"sla_{kb[0]}_{kb[1]}"] = score(Ew[kb], y_c, Pg)
            for (k_, b_, lam) in pl.get("slal3", []):
                W = Wc[(k_, b_)]
                Pd = {d: l3_safe(cal[d][0] @ W[d], cal[d][1], P, lam) for d in doms}
                res[f"slal3_{k_}_{b_}_{lam}"] = score(Ew[(k_, b_)], y_c, Pd)
            for m_, v in res.items():
                acc[tag][m_].append(v)
    return {c: {m_: {k: float(np.mean([r[k] for r in lst])) for k in lst[0]} for m_, lst in d.items()}
            for c, d in acc.items()}


# ── 메인 ─────────────────────────────────────────────────────────────────────

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir + "_noea")
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--n_rep_val", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_files", type=int, default=0, help="시험용: 앞 몇 fold 만")
    args = ap.parse_args()

    R = load_ratings()
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")),
                   key=lambda p: tuple(map(int, re.search(r"S(\d+)_seed(\d+)", p).groups())))
    if args.max_files:
        files = files[:args.max_files]
    full_val = {t: {"l3": LAMS, "cr": CR_GRID, "sla": SLA_GRID,
                    "slal3": [(k_, b_, lam) for (k_, b_) in SLA_GRID for lam in LAMS]} for t in BUDGETS}
    OUT = defaultdict(lambda: defaultdict(list))         # key → {subject: [값 (시드별)]}
    HP = []
    for j, f in enumerate(files, 1):
        ts, sd, meta, lab, Z, cidx, train, val, P, T = load_fold(f)
        dcls = design_classes(R, train)
        # 검증 피험자로 초매개변수 선택 (예산마다, 창 균형 정확도)
        vs = defaultdict(lambda: defaultdict(list))
        for v in val:
            rv = run_subject(Z, lab, cidx, P, T, dcls, v, doms_of(meta, v),
                             np.random.default_rng(100000 + 1000 * v + sd), args.n_rep_val, full_val)
            for t in BUDGETS:
                for m_, met in rv[t].items():
                    vs[t][m_].append(met["win_bal"])
        chosen = {}
        for t in BUDGETS:
            lam = max(LAMS, key=lambda x: np.mean(vs[t][f"l3_{x}"]))
            cr = max(CR_GRID, key=lambda g: np.mean(vs[t][f"cr_{g[0]}_{g[1]}"]))
            kb = max(SLA_GRID, key=lambda g: np.mean(vs[t][f"sla_{g[0]}_{g[1]}"]))
            lam2 = max(LAMS, key=lambda x: np.mean(vs[t][f"slal3_{kb[0]}_{kb[1]}_{x}"]))
            chosen[t] = {"l3": lam, "cr": cr, "sla": kb, "slal3_lam": lam2}
        HP.append({"ts": ts, "sd": sd, **{t: {"l3": c["l3"], "cr": list(c["cr"]), "sla": list(c["sla"]),
                                             "slal3_lam": c["slal3_lam"]} for t, c in chosen.items()}})
        plan = {t: {"l3": [c["l3"]], "cr": [c["cr"]], "sla": [c["sla"]], "slal3": [(*c["sla"], c["slal3_lam"])]}
                for t, c in chosen.items()}
        rt = run_subject(Z, lab, cidx, P, T, dcls, ts, doms_of(meta, ts),
                         np.random.default_rng(1000 * ts + sd), args.n_rep, plan, extras=True)
        names = {}
        for t, c in chosen.items():
            k_, b_ = c["sla"]
            names[t] = (("center", "center"), ("l3", f"l3_{c['l3']}"), ("cr", f"cr_{c['cr'][0]}_{c['cr'][1]}"),
                        ("sla", f"sla_{k_}_{b_}"), ("slal3", f"slal3_{k_}_{b_}_{c['slal3_lam']}"))
        for cond, d in rt.items():
            pairs = names.get(cond, [(m_, m_) for m_ in d])
            for nm, key in pairs:
                for met, val_ in d[key].items():
                    OUT[f"{cond}|{nm}|{met}"][ts].append(val_)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  clip_bal 없음 {rt['none']['none']['clip_bal']:.3f}  " + "  ".join(
            f"{t} c/sla/l3 {rt[t]['center']['clip_bal']:.3f}/{rt[t][names[t][3][1]]['clip_bal']:.3f}"
            f"/{rt[t][names[t][1][1]]['clip_bal']:.3f}" for t in BUDGETS), flush=True)

    subs = sorted(OUT["none|none|clip_bal"])
    arr = lambda k: np.array([np.mean(OUT[k][s]) for s in subs])
    np.savez(args.out, subjects=np.array(subs), **{k.replace("|", "__"): arr(k) for k in OUT})
    with open(args.out.replace(".npz", "_hp.json"), "w") as fh:
        json.dump(HP, fh)
    print(f"\n[저장] {args.out}  (피험자 {len(subs)}명)")
    for t in BUDGETS:
        print(f"  선택 {t}: L3 λ {dict(Counter(h[t]['l3'] for h in HP))}  CR {dict(Counter(tuple(h[t]['cr']) for h in HP))}  "
              f"SLA {dict(Counter(tuple(h[t]['sla']) for h in HP))}")


if __name__ == "__main__":
    main()
