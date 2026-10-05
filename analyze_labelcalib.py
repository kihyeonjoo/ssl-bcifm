"""
분석 D: 자극 라벨 캘리브레이션.

캘리브레이션 영상은 **실험자가 고른 것**이므로, 그 창이 어떤 감정을 유도하려고
제시됐는지는 사용자 주석 없이 안다.  S10 의 M1·M2 는 이 정보를 안 쓰고 pseudo-label
로 추정했고, 그래서 oracle 상한의 5~16% 만 회수했다.  여기서는 그 라벨을 직접 쓴다.

**평가 클립의 라벨은 절대 쓰지 않는다.**  S10 의 oracle 회전은 평가 클립의 라벨을
썼으므로 도달 불가능한 상한이었다 — 여기 방법들은 전부 배치 가능하다.

프로토콜은 S8/S9/S10 과 같다: 세션마다 감정별 1클립씩 떼어 캘리브레이션, 나머지
12클립 평가.  예산은 **시청 시간** — 떼어낸 각 클립의 앞 T초를 본다.

방법 (전부 캘리브레이션 창 + 그 자극 라벨만 사용)
  L1   균형 기준점.  μ = 감정별 캘리브레이션 평균들의 평균.
       **주의**: 감정별로 같은 길이를 떼는 이 프로토콜에서는 L1 이 단순 평균과
       수학적으로 같다 (각 클래스가 같은 창 수를 내므로).  검증용으로 확인만 하고,
       의미 있는 라벨 버전은 아래 L1m 이다.
  L1m  M1 의 라벨 버전.  μ = mean_i( z_i − γ·P_{c(i)} ), c(i) 는 자극 라벨.
       추정된 p 대신 정답 one-hot 을 쓰는 M1.
  L2   회전.  캘리브레이션 감정별 평균을 학습 prototype 에 맞추는 Procrustes.
       S10 M2 와 같은 구조에 pseudo-label 대신 자극 라벨.
  L3   prototype 혼합.  P_new = λ·(캘리브레이션 감정 평균, 길이를 ‖P‖ 로 맞춤)
       + (1−λ)·P.  길이 보정을 넣는 이유는 S10 에서 잰 대로 테스트 피험자의 클래스
       편차가 학습 prototype 의 절반이기 때문이다.
  L4   head 마지막 층 적응.  캘리브레이션 창으로 몇 스텝 미세조정하되 원래 가중치
       쪽으로 L2 규제.
  조합 L1m+L2, L1m+L3

초매개변수는 각 fold 의 검증 피험자 2명으로 고른다.
판정은 짝지은 검정 (결정 규칙 비교).
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
import torch.nn as nn
from scipy import stats

# This machine has 56 cores and torch defaults to 28 threads.  L4 trains a
# 200->3 linear layer on a few hundred rows, and on tensors that small the
# thread handoff dominates: 60 Adam steps take 45 s at 56 threads, 0.12 s at 4.
# Measured, not guessed.
torch.set_num_threads(4)

from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               head_probs, load_head, bootstrap_ci)
from analyze_calib_protocol import _train_side
from analyze_m1 import score_with_mu
from adapt_methods import m2_rotation, train_prototypes_raw
from calib_common import (SEC_PER_WIN, clips_by_domain_label, draw_holdout,
                          n_windows)

GAMMAS = [0.25, 0.5, 0.75]
KEXTRA = [0, 2, 5]
BETAS = [0.25, 0.5, 1.0]
LAMBDAS = [0.25, 0.5, 0.75, 1.0]
L4_LRS = [1e-3, 3e-3]
L4_STEPS = [20, 60]
L4_REGS = [1e-2, 1e-1]


def calib_by_label(cidx, held, doms, T):
    """{domain: {class: window indices}} — 앞 T초 (T='full' 이면 클립 전체)."""
    out = {}
    for d in doms:
        per = {}
        for c, k in enumerate(held[d]):
            w = cidx[k]
            n = len(w) if T == "full" else n_windows(float(T))
            per[c] = np.asarray(w[:n])
        out[d] = per
    return out


# ── L1 / L1m : centring reference points ────────────────────────────────────

def mu_simple(X, per):
    idx = np.concatenate([per[c] for c in per])
    return X[idx].mean(0)


def mu_balanced(X, per):
    return np.stack([X[per[c]].mean(0) for c in per]).mean(0)


def mu_label_m1(X, per, P, gamma):
    """M1 with the stimulus label instead of an estimated posterior."""
    parts = [X[per[c]] - gamma * P[c] for c in per]
    return np.concatenate(parts).mean(0)


# ── L2 : rotation from calibration class means ──────────────────────────────

def l2_rotation(Zc, per, mu, P, k_extra, beta):
    """캘리브레이션 감정별 평균 -> 학습 prototype 으로 가는 회전."""
    M = np.stack([(Zc[per[c]] - mu).mean(0) for c in sorted(per)])
    pred = np.arange(N_CLS)
    return m2_rotation(M, pred, P, extra=M, k_extra=k_extra, beta=beta,
                       min_per_class=1)


# ── L3 : prototype mixing ───────────────────────────────────────────────────

def l3_prototypes(Zc, per, mu, P, lam):
    """새 prototype = λ·(캘리브레이션 감정 평균, 길이 보정) + (1−λ)·학습 prototype."""
    out = P.copy()
    for c in sorted(per):
        m = (Zc[per[c]] - mu).mean(0)
        n = np.linalg.norm(m)
        if n < 1e-9:
            continue
        m = m / n * np.linalg.norm(P[c])          # match the train prototype's length
        out[c] = lam * m + (1 - lam) * P[c]
    return out


# ── L4 : last-layer head adaptation ─────────────────────────────────────────

def l4_head(head, Xc, yc, lr, steps, reg):
    """head 의 마지막 Linear 만 캘리브레이션 창으로 적응.  원 가중치로 L2 당김.

    전체 head 를 풀면 603 개가 아니라 41k 개를 15~45 창으로 맞추게 되어 곧바로
    과적합한다.  마지막 층만 풀고 규제를 건다."""
    import copy
    h = copy.deepcopy(head)
    last = h[-1]
    W0, b0 = last.weight.detach().clone(), last.bias.detach().clone()
    for p in h.parameters():
        p.requires_grad_(False)
    last.weight.requires_grad_(True)
    last.bias.requires_grad_(True)
    opt = torch.optim.Adam([last.weight, last.bias], lr=lr)
    X = torch.from_numpy(np.ascontiguousarray(Xc)).float()
    y = torch.from_numpy(np.ascontiguousarray(yc)).long()
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(h(X), y)
        loss = loss + reg * ((last.weight - W0).pow(2).sum()
                             + (last.bias - b0).pow(2).sum())
        loss.backward()
        opt.step()
    return h.eval()


# ── evaluation ──────────────────────────────────────────────────────────────

def eval_proto(Z, cidx, eval_keys, y_c, mu, P):
    """조건4 경로: 중심화된 특징을 prototype 과 코사인 비교."""
    Pn = l2(P)
    pw, pc, yw = [], [], []
    for i, k in enumerate(eval_keys):
        w = cidx[k]
        A = Z[w] - mu[(k[0], k[1])]
        pw.append((l2(A) @ Pn.T).argmax(1))
        pc.append(int((l2(A.mean(0)[None]) @ Pn.T).argmax(1)[0]))
        yw.append(np.full(len(w), int(y_c[i])))
    yw = np.concatenate(yw)
    return {"proto_win": float((np.concatenate(pw) == yw).mean()),
            "proto_clip": float((np.array(pc) == y_c).mean())}


def eval_head(cls, cidx, eval_keys, y_c, mu_raw, mu_tr_global, head):
    pw, pc, yw = [], [], []
    for i, k in enumerate(eval_keys):
        w = cidx[k]
        Ph = head_probs(head, cls[w] - mu_raw[(k[0], k[1])] + mu_tr_global)
        pw.append(Ph.argmax(1))
        pc.append(int(Ph.mean(0).argmax()))
        yw.append(np.full(len(w), int(y_c[i])))
    yw = np.concatenate(yw)
    return {"head_win": float((np.concatenate(pw) == yw).mean()),
            "head_clip": float((np.array(pc) == y_c).mean())}


def run_methods(cls, Z, cidx, eval_keys, y_c, calib, P_z, P_raw, mu_tr_global,
                head, hp, want_l4=True):
    """{방법: 지표} — 하나의 (피험자, 시청 시간, 추출) 조합."""
    doms = sorted(calib)
    out = {}

    mu_s = {d: mu_simple(Z, calib[d]) for d in doms}
    mu_s_raw = {d: mu_simple(cls, calib[d]) for d in doms}
    out["base"] = eval_proto(Z, cidx, eval_keys, y_c, mu_s, P_z)
    out["base"].update(eval_head(cls, cidx, eval_keys, y_c, mu_s_raw,
                                 mu_tr_global, head))

    mu_b = {d: mu_balanced(Z, calib[d]) for d in doms}
    out["L1"] = eval_proto(Z, cidx, eval_keys, y_c, mu_b, P_z)

    mu_m = {d: mu_label_m1(Z, calib[d], P_z, hp["gamma"]) for d in doms}
    out["L1m"] = eval_proto(Z, cidx, eval_keys, y_c, mu_m, P_z)

    out["L2"] = _score_rotated(Z, cidx, eval_keys, y_c, calib, mu_s, P_z,
                               hp["k_extra"], hp["beta"])
    out["L1m+L2"] = _score_rotated(Z, cidx, eval_keys, y_c, calib, mu_m, P_z,
                                   hp["k_extra"], hp["beta"])

    out["L3"] = _score_mixed(Z, cidx, eval_keys, y_c, calib, mu_s, P_z,
                             hp["lam"])
    out["L1m+L3"] = _score_mixed(Z, cidx, eval_keys, y_c, calib, mu_m, P_z,
                                 hp["lam"])

    if want_l4:
        Xc, yc = [], []
        for d in doms:
            for c, idx in calib[d].items():
                Xc.append(cls[idx] - mu_s_raw[d] + mu_tr_global)
                yc.append(np.full(len(idx), c))
        h = l4_head(head, np.concatenate(Xc), np.concatenate(yc),
                    hp["l4_lr"], hp["l4_steps"], hp["l4_reg"])
        out["L4"] = eval_head(cls, cidx, eval_keys, y_c, mu_s_raw,
                              mu_tr_global, h)
    return out


# ── scorers shared by selection and evaluation ─────────────────────

def _score_rotated(Z, cidx, eval_keys, y_c, calib, mu_map, P, k_extra, beta):
    Ws = {}
    for d in sorted(calib):
        W, ok = l2_rotation(Z, calib[d], mu_map[d], P, k_extra, beta)
        Ws[d] = W if ok else np.eye(Z.shape[1], dtype=np.float32)
    Pn = l2(P)
    pw, pc, yw = [], [], []
    for i, k in enumerate(eval_keys):
        w = cidx[k]
        d = (k[0], k[1])
        A = (Z[w] - mu_map[d]) @ Ws[d]
        pw.append((l2(A) @ Pn.T).argmax(1))
        pc.append(int((l2(A.mean(0)[None]) @ Pn.T).argmax(1)[0]))
        yw.append(np.full(len(w), int(y_c[i])))
    yw = np.concatenate(yw)
    return {"proto_win": float((np.concatenate(pw) == yw).mean()),
            "proto_clip": float((np.array(pc) == y_c).mean())}


def _score_mixed(Z, cidx, eval_keys, y_c, calib, mu_map, P, lam):
    Pn = {d: l2(l3_prototypes(Z, calib[d], mu_map[d], P, lam))
          for d in sorted(calib)}
    pw, pc, yw = [], [], []
    for i, k in enumerate(eval_keys):
        w = cidx[k]
        d = (k[0], k[1])
        A = Z[w] - mu_map[d]
        pw.append((l2(A) @ Pn[d].T).argmax(1))
        pc.append(int((l2(A.mean(0)[None]) @ Pn[d].T).argmax(1)[0]))
        yw.append(np.full(len(w), int(y_c[i])))
    yw = np.concatenate(yw)
    return {"proto_win": float((np.concatenate(pw) == yw).mean()),
            "proto_clip": float((np.array(pc) == y_c).mean())}


# ── validation-based hyperparameter choice ────────────────────────

def choose(cls, Z, meta, lab, cidx, head, ts, secs, sel_metric):
    """Pick each method's hyperparameters on this fold's two validation subjects.

    Each grid evaluates ONLY its own method.  Calling ``run_methods`` inside the
    loop instead re-runs L4's torch fine-tune for every value of every other
    method's grid, which is what made the first version unusable.
    Methods are selected separately so one method's optimum cannot drag another.
    """
    train, val, _ = roles_for_fold(ts)
    mg, P_z = _train_side(cls, meta, lab, train)
    P_raw = train_prototypes_raw(cls, meta, lab, cidx, train)
    packs = []
    for v in val:
        doms = sorted({(k[0], k[1]) for k in cidx if k[0] == v})
        by = clips_by_domain_label(cidx, lab, v)
        held, hs = draw_holdout(by, doms, np.random.default_rng(1000 * v + 7))
        ev = [k for k in sorted(cidx) if k[0] == v and k not in hs]
        y = np.array([lab[cidx[k][0]] for k in ev])
        for T in secs:
            packs.append((calib_by_label(cidx, held, doms, T), ev, y))

    hp = {}
    pm = sel_metric
    hm = sel_metric.replace("proto", "head")

    best, bv = GAMMAS[0], -1.0
    for g in GAMMAS:
        vals = [eval_proto(Z, cidx, ev, y,
                           {d: mu_label_m1(Z, cal[d], P_z, g) for d in sorted(cal)},
                           P_z)[pm] for cal, ev, y in packs]
        if np.mean(vals) > bv:
            bv, best = float(np.mean(vals)), g
    hp["gamma"] = best

    best, bv = (KEXTRA[0], BETAS[0]), -1.0
    for ke in KEXTRA:
        for b in BETAS:
            vals = [_score_rotated(Z, cidx, ev, y, cal,
                                   {d: mu_simple(Z, cal[d]) for d in sorted(cal)},
                                   P_z, ke, b)[pm] for cal, ev, y in packs]
            if np.mean(vals) > bv:
                bv, best = float(np.mean(vals)), (ke, b)
    hp["k_extra"], hp["beta"] = best

    best, bv = LAMBDAS[0], -1.0
    for lam in LAMBDAS:
        vals = [_score_mixed(Z, cidx, ev, y, cal,
                             {d: mu_simple(Z, cal[d]) for d in sorted(cal)},
                             P_z, lam)[pm] for cal, ev, y in packs]
        if np.mean(vals) > bv:
            bv, best = float(np.mean(vals)), lam
    hp["lam"] = best

    best, bv = (L4_LRS[0], L4_STEPS[0], L4_REGS[0]), -1.0
    for lr in L4_LRS:
        for st in L4_STEPS:
            for rg in L4_REGS:
                vals = []
                for cal, ev, y in packs:
                    doms = sorted(cal)
                    mu_raw = {d: mu_simple(cls, cal[d]) for d in doms}
                    Xc, yc = [], []
                    for d in doms:
                        for c, idx in cal[d].items():
                            Xc.append(cls[idx] - mu_raw[d] + mg)
                            yc.append(np.full(len(idx), c))
                    h = l4_head(head, np.concatenate(Xc), np.concatenate(yc),
                                lr, st, rg)
                    vals.append(eval_head(cls, cidx, ev, y, mu_raw, mg, h)[hm])
                if np.mean(vals) > bv:
                    bv, best = float(np.mean(vals)), (lr, st, rg)
    hp["l4_lr"], hp["l4_steps"], hp["l4_reg"] = best
    return hp, P_z, P_raw, mg


# ── main ────────────────────────────────────────────────────────────────────

METHODS = ["base", "L1", "L1m", "L2", "L3", "L1m+L2", "L1m+L3", "L4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prefix", default=CFG.ckpt_prefix)
    ap.add_argument("--secs", nargs="+", default=["20", "40", "full"])
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--sel_metric", default="proto_win")
    ap.add_argument("--out", default=f"results/{CFG.prefix('labelcalib')}.npz")
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
        hp, P_z, P_raw, mg = choose(cls, Z, meta, lab, cidx, head, ts,
                                    args.secs, args.sel_metric)
        HP[(ts, sd)] = dict(hp)

        doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
        by = clips_by_domain_label(cidx, lab, ts)
        rng = np.random.default_rng(1000 * ts + sd)
        for _ in range(args.n_rep):
            held, hs = draw_holdout(by, doms, rng)
            ev = [k for k in sorted(cidx) if k[0] == ts and k not in hs]
            y = np.array([lab[cidx[k][0]] for k in ev])
            for T in args.secs:
                calib = calib_by_label(cidx, held, doms, T)
                r = run_methods(cls, Z, cidx, ev, y, calib, P_z, P_raw, mg,
                                head, hp)
                for m, d in r.items():
                    for k, v in d.items():
                        R[f"T{T}|{m}|{k}"][ts].append(v)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  γ{hp['gamma']} "
              f"k{hp['k_extra']} β{hp['beta']} λ{hp['lam']} "
              f"L4({hp['l4_lr']},{hp['l4_steps']},{hp['l4_reg']})  ||  "
              f"T20 base {np.mean(R['T20|base|proto_win'][ts]):.3f} "
              f"L2 {np.mean(R['T20|L2|proto_win'][ts]):.3f} "
              f"L3 {np.mean(R['T20|L3|proto_win'][ts]):.3f}", flush=True)

    def arr(k):
        return np.array([np.mean(R[k][s]) for s in sorted(R[k])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    report(arr, R, HP, args)


def report(arr, R, HP, args):
    W = 104
    print(f"\n{'='*W}\n선택된 초매개변수 ({args.sel_metric}, 검증 피험자 2명)\n{'='*W}")
    for key, vals in (("gamma", GAMMAS), ("k_extra", KEXTRA), ("beta", BETAS),
                      ("lam", LAMBDAS)):
        c = [h[key] for h in HP.values()]
        print(f"  {key:<9}" + "  ".join(f"{v}:{c.count(v)}" for v in vals))
    print(f"  L4        lr " + " ".join(
        f"{v}:{[h['l4_lr'] for h in HP.values()].count(v)}" for v in L4_LRS)
        + "  steps " + " ".join(
        f"{v}:{[h['l4_steps'] for h in HP.values()].count(v)}" for v in L4_STEPS)
        + "  reg " + " ".join(
        f"{v}:{[h['l4_reg'] for h in HP.values()].count(v)}" for v in L4_REGS))

    for metric in ("proto_win", "proto_clip"):
        print(f"\n{'='*W}\n{metric}   기준선 = 라벨 없는 최선 (단순 평균 중심화, "
              f"앞부분)\n결정 규칙 비교 -> 짝지은 검정\n{'='*W}")
        print(f"  {'방법':<10}" + "".join(
            f"{('시청 '+t+'초' if t!='full' else '클립 전체'):>24}"
            for t in args.secs))
        for m in METHODS:
            if m == "L4":
                continue
            cells = []
            for T in args.secs:
                k = f"T{T}|{m}|{metric}"
                if k not in R:
                    cells.append(f"{'-':>24}")
                    continue
                a, b = arr(k), arr(f"T{T}|base|{metric}")
                d = a - b
                try:
                    p = stats.wilcoxon(a, b).pvalue if m != "base" else np.nan
                except ValueError:
                    p = np.nan
                tail = "" if m == "base" else \
                    f" {d.mean():+.4f} p{p:.3f} {int((d>0).sum())}/{len(d)}"
                cells.append(f"{a.mean():.4f}{tail}".rjust(24))
            print(f"  {m:<10}" + "".join(cells))

    for metric in ("head_win", "head_clip"):
        print(f"\n{'='*W}\n{metric}   기준선 = 조건2 (단순 평균 중심화)\n{'='*W}")
        print(f"  {'방법':<10}" + "".join(
            f"{('시청 '+t+'초' if t!='full' else '클립 전체'):>24}"
            for t in args.secs))
        for m in ("base", "L4"):
            cells = []
            for T in args.secs:
                k = f"T{T}|{m}|{metric}"
                a, b = arr(k), arr(f"T{T}|base|{metric}")
                d = a - b
                p = (stats.wilcoxon(a, b).pvalue if m != "base" else np.nan)
                tail = "" if m == "base" else \
                    f" {d.mean():+.4f} p{p:.3f} {int((d>0).sum())}/{len(d)}"
                cells.append(f"{a.mean():.4f}{tail}".rjust(24))
            print(f"  {m:<10}" + "".join(cells))

    # L1 equals the simple mean only when every class contributes the SAME
    # number of windows.  Fixed T does that; whole clips do not, because clips
    # differ in length and the simple mean is then weighted by clip length.
    print("\n  [확인] L1 과 단순 평균의 최대 차이 (proto_win)")
    for T in args.secs:
        d = abs(arr(f"T{T}|L1|proto_win") - arr(f"T{T}|base|proto_win")).max()
        note = ("같아야 한다 (감정별 같은 창 수)" if T != "full"
                else "달라도 된다 (클립 길이가 달라 단순 평균이 긴 클립에 가중된다)")
        print(f"    시청 {T:<6} {d:.2e}  — {note}")

    # 라벨 상한은 analyze_centering 이 데이터셋마다 계산해 둔 ceil_logreg 다
    # (SEED 0.8464, SEED-V 0.5431).  박아 두면 다른 데이터셋에서 분모가 틀린다.
    _cpath = f"results/{CFG.prefix('centering_analysis')}.npz"
    CEIL = float(np.asarray(np.load(_cpath)["ceil_logreg"]).mean())
    print(f"\n{'='*W}\nS2 라벨 상한({CEIL:.4f} clip, 피험자 단위 LOCO 라벨 + "
          f"로지스틱 회귀) 대비 회수율\n{'='*W}")
    for T in args.secs:
        b = arr(f"T{T}|base|proto_clip").mean()
        print(f"  시청 {T:<6} 기준선 {b:.4f}  " + "  ".join(
            f"{m} {arr(f'T{T}|{m}|proto_clip').mean():.4f}"
            f"({(arr(f'T{T}|{m}|proto_clip').mean()-b)/(CEIL-b)*100:+.1f}%)"
            for m in ("L1m", "L2", "L3")))

    print(f"\n{'='*W}\n피험자별 실패 (시청 40초, proto_win)\n{'='*W}")
    for m in ("L1m", "L2", "L3", "L1m+L2", "L1m+L3"):
        k = f"T40|{m}|proto_win"
        if k not in R:
            continue
        a, b = arr(k), arr("T40|base|proto_win")
        d = a - b
        bad = [(i + 1, b[i], d[i]) for i in range(len(d)) if d[i] < 0]
        rho, pv = stats.spearmanr(b, d)
        print(f"  {m:<8} 나빠진 {len(bad):2d}/{len(b)}  "
              + (", ".join(f"S{s}({v:.3f}{dd:+.3f})" for s, v, dd in bad[:6])
                 or "없음")
              + f"   ρ(기준선,이득)={rho:+.3f} p={pv:.3f}")
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
