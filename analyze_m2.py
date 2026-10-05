"""
M2 (pseudo-label Procrustes 회전) 와 기존 테스트시점 적응 방법들 — 한 표에.

중심화는 도메인 오프셋(1차 모멘트)만 없앤다.  S8 에서 클래스 방향의 코사인이
학습-테스트 사이 0.82~0.96 에 그쳤으므로 **남는 것은 회전**이다.  M2 는 그 회전을
pseudo-label 로 추정한다 — 다만 d=200 전체 회전을 클래스 평균 3개로 맞출 수는
없으므로, **학습 prototype 이 이루는 저차원 부분공간 안에서만** Procrustes 로 풀고
단위행렬 쪽으로 β 만큼 수축한다.

같은 표에 올리는 기존 방법
  adaBN   도메인별 평균·분산 정규화
  LA      Latent Alignment — 도메인별 표준화 후 학습 도메인 통계로 되돌림
  T3A     확신도 높은 테스트 특징으로 prototype 점진 갱신

평가는 **균형 시나리오**(감정별 20/40초, 3클립 전체)에서 하고, 기준선은 S9 0단계의
"가장 강한 현실적 파이프라인"이다.  전부 결정 규칙 비교이므로 짝지은 검정으로 판정한다.

확신도는 **여백(top1 − top2 코사인)의 상위 q 비율**로 정한다 — 소프트맥스 문턱을 쓰면
온도라는 초매개변수가 하나 더 생기고, 그 온도가 실제로 고르는 것은 "몇 개를
넣을지" 이기 때문이다.
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

from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               head_probs, load_head, bootstrap_ci)
from analyze_calib_protocol import _train_side
from adapt_methods import m1_mean, m2_rotation, train_prototypes_raw, adabn
from calib_common import (calib_indices, clips_by_domain_label, draw_holdout)

# oracle 상한이 k_extra 와 함께 커진다 (0:+0.019, 2:+0.041, 5:+0.044 clip) —
# 도움이 되는 회전이 prototype 3개가 이루는 span 밖에도 있다는 뜻이므로 5 까지 본다.
KEXTRA = [0, 2, 5]
QS = [0.3, 0.5, 1.0]
BETAS = [0.25, 0.5, 1.0]
ITERS = [1, 2, 3]
T3A_QS = [0.1, 0.3, 0.5]


def confident(sim, q):
    """여백 상위 q 비율의 인덱스.  q=1.0 이면 전부."""
    if q >= 1.0:
        return np.arange(len(sim))
    s = np.sort(sim, axis=1)
    margin = s[:, -1] - s[:, -2]
    k = max(N_CLS, int(round(q * len(sim))))
    return np.argsort(-margin)[:k]


def predict(A, P):
    return (l2(A) @ l2(P).T)


def eval_rep(A_clip, A_win, wlen, y_c, P):
    """clip/window 정확도.  ``wlen[i]`` 는 클립 i 의 창 개수."""
    pc = predict(A_clip, P).argmax(1)
    pw = predict(A_win, P).argmax(1)
    yw = np.repeat(y_c, wlen)
    return {"proto_clip": float((pc == y_c).mean()),
            "proto_win": float((pw == yw).mean())}


def build_rep(cls, Z, cidx, eval_keys, calib, centring, P_raw, P_z, hp_m1):
    """중심화된 clip/window 표현을 만든다.  ``centring`` 은 'mean' 또는 'm1'."""
    mu = {}
    for d, idx in calib.items():
        idx = np.asarray(idx)
        mu[d] = (Z[idx].mean(0) if centring == "mean" else
                 m1_mean(Z[idx], P_z, tau=hp_m1[0], n_iter=hp_m1[1],
                         gamma=hp_m1[2]))
    A_clip, A_win, wlen = [], [], []
    for k in eval_keys:
        w = cidx[k]
        d = (k[0], k[1])
        A_clip.append(Z[w].mean(0) - mu[d])
        A_win.append(Z[w] - mu[d])
        wlen.append(len(w))
    return (np.stack(A_clip), np.concatenate(A_win), np.array(wlen), mu)


def m2_apply(A_clip, A_win, P, k_extra, q, beta, n_iter, y_true=None):
    """M2: 예측 -> 회전 -> 재예측.  ``y_true`` 가 있으면 oracle 회전."""
    Ac, Aw = A_clip.copy(), A_win.copy()
    applied = False
    for _ in range(n_iter):
        sim = predict(Ac, P)
        pred = y_true if y_true is not None else sim.argmax(1)
        idx = (np.arange(len(Ac)) if y_true is not None
               else confident(sim, q))
        W, ok = m2_rotation(Ac[idx], pred[idx], P, extra=Ac,
                            k_extra=k_extra, beta=beta)
        if not ok:
            break
        Ac, Aw = Ac @ W, Aw @ W
        applied = True
    return Ac, Aw, applied


# ── 기존 방법 ────────────────────────────────────────────────────────────────

def baseline_reps(cls, Z, cidx, eval_keys, calib, P_z, mu_tr_z, hp):
    """{방법: (A_clip, A_win, P)} — 각 방법이 쓰는 표현과 prototype.

    prototype 을 바꾸는 방법(T3A)과 특징을 바꾸는 방법(adaBN, LA)을 한 인터페이스로
    두어, 평가 코드가 하나만 있게 한다."""
    out = {}
    doms = sorted(calib)
    wlen = np.array([len(cidx[k]) for k in eval_keys])

    def assemble(fn):
        Ac, Aw = [], []
        for k in eval_keys:
            w = cidx[k]
            zz = fn((k[0], k[1]), Z[w])
            Ac.append(zz.mean(0))
            Aw.append(zz)
        return np.stack(Ac), np.concatenate(Aw)

    cal = {d: np.asarray(i) for d, i in calib.items()}

    # adaBN: 캘리브레이션 창의 차원별 평균·표준편차로 정규화
    stats_ = {d: (Z[i].mean(0), Z[i].std(0) + 1e-8) for d, i in cal.items()}
    Ac, Aw = assemble(lambda d, X: (X - stats_[d][0]) / stats_[d][1])
    out["adaBN"] = (Ac, Aw, P_z, wlen)

    # LA: 같은 표준화 뒤 **학습 도메인 통계로 되돌린다**
    # (구현 가정: head/prototype 이 학습 때 본 범위를 유지해야 하므로.  원 논문들은
    #  대상 분포를 소스 분포에 맞추므로 되돌림이 있는 쪽이 그 취지에 가깝다.)
    tr_mu, tr_sd = mu_tr_z
    Ac, Aw = assemble(
        lambda d, X: (X - stats_[d][0]) / stats_[d][1] * tr_sd + tr_mu)
    out["LA"] = (Ac, Aw, P_z, wlen)

    # T3A: 중심화는 단순 평균, prototype 을 확신도 높은 테스트 clip 으로 갱신
    mu = {d: Z[i].mean(0) for d, i in cal.items()}
    Ac, Aw = assemble(lambda d, X: X - mu[d])
    sim = predict(Ac, P_z)
    idx = confident(sim, hp["t3a_q"])
    pred = sim.argmax(1)
    Pt = P_z.copy()
    for c in range(N_CLS):
        sel = idx[pred[idx] == c]
        if len(sel):
            Pt[c] = np.concatenate([P_z[c][None], Ac[sel]]).mean(0)
    out["T3A"] = (Ac, Aw, Pt, wlen)
    return out


def _val_pack(cls, Z, meta, lab, cidx, ts, seconds, mode="prefix"):
    """검증 피험자별 (prototype, raw prototype, 캘리브레이션, 평가 클립, 라벨).

    prototype 은 그 fold 의 **학습 피험자만**으로 만든다 — 검증 피험자를 넣으면
    자기 자신으로 자기를 맞추는 셈이 된다."""
    train, val, _ = roles_for_fold(ts)
    pack = []
    for v in val:
        _, pz = _train_side(cls, meta, lab, train)
        pr = train_prototypes_raw(cls, meta, lab, cidx, train)
        doms = sorted({(k[0], k[1]) for k in cidx if k[0] == v})
        by = clips_by_domain_label(cidx, lab, v)
        held, hs = draw_holdout(by, doms, np.random.default_rng(1000 * v + 7))
        ev = [k for k in sorted(cidx) if k[0] == v and k not in hs]
        y = np.array([lab[cidx[k][0]] for k in ev])
        for sec in seconds:
            pack.append((pz, pr, calib_indices(held, cidx, doms, sec, mode),
                         ev, y))
    return pack


def _t3a_prototypes(Ac, P, q):
    sim = predict(Ac, P)
    idx = confident(sim, q)
    pred = sim.argmax(1)
    Pt = P.copy()
    for c in range(N_CLS):
        sel = idx[pred[idx] == c]
        if len(sel):
            Pt[c] = np.concatenate([P[c][None], Ac[sel]]).mean(0)
    return Pt


def choose_m2(cls, Z, meta, lab, cidx, ts, args, hp_m1):
    """검증 피험자 2명으로 (k_extra, q, β, 반복) 과 T3A 의 q 를 고른다."""
    pack = _val_pack(cls, Z, meta, lab, cidx, ts, args.seconds, args.mode)
    reps = [(pz, build_rep(cls, Z, cidx, ev, cal, "mean", pr, pz, hp_m1), y)
            for pz, pr, cal, ev, y in pack]

    best, bv = (KEXTRA[0], QS[0], BETAS[0], ITERS[0]), -1.0
    for ke in KEXTRA:
        for q in QS:
            for b in BETAS:
                for it in ITERS:
                    vals = []
                    for pz, (Ac, Aw, wl, _), y in reps:
                        Ac2, Aw2, _ = m2_apply(Ac, Aw, pz, ke, q, b, it)
                        vals.append(eval_rep(Ac2, Aw2, wl, y,
                                             pz)[args.sel_metric])
                    m = float(np.mean(vals))
                    if m > bv:
                        bv, best = m, (ke, q, b, it)

    bq, best_q = -1.0, T3A_QS[0]
    for q in T3A_QS:
        vals = []
        for pz, (Ac, Aw, wl, _), y in reps:
            vals.append(eval_rep(Ac, Aw, wl, y,
                                 _t3a_prototypes(Ac, pz, q))[args.sel_metric])
        m = float(np.mean(vals))
        if m > bq:
            bq, best_q = m, q
    return best, bv, best_q


def run_checkpoint(cls, meta, lab, head, ts, args, hp_m1):
    cidx = clip_index(meta)
    Z = l2(cls)
    train, _, _ = roles_for_fold(ts)
    mu_tr_global, P_z = _train_side(cls, meta, lab, train)
    P_raw = train_prototypes_raw(cls, meta, lab, cidx, train)
    # LA needs the TRAIN domains' normalised-feature statistics to map back to
    tr_mask = np.isin(meta[:, 0], train)
    mu_tr_z = (Z[tr_mask].mean(0), Z[tr_mask].std(0) + 1e-8)

    hp_m2, vscore, t3a_q = choose_m2(cls, Z, meta, lab, cidx, ts, args, hp_m1)
    ke, q, beta, n_it = hp_m2
    out = {"hp_m2": hp_m2, "hp_m2_val": vscore, "hp_t3a_q": t3a_q}

    doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
    by = clips_by_domain_label(cidx, lab, ts)
    rng = np.random.default_rng(1000 * ts + args.seed_offset)
    for _ in range(args.n_rep):
        held, hs = draw_holdout(by, doms, rng)
        ev = [k for k in sorted(cidx) if k[0] == ts and k not in hs]
        y = np.array([lab[cidx[k][0]] for k in ev])
        for sec in list(args.seconds) + ["full"]:
            tag = "Tfull" if sec == "full" else f"T{sec:g}"
            cal = calib_indices(held, cidx, doms, sec, args.mode)

            Ac, Aw, wl, _ = build_rep(cls, Z, cidx, ev, cal, "mean",
                                      P_raw, P_z, hp_m1)
            base = eval_rep(Ac, Aw, wl, y, P_z)          # = 조건4 기준선
            for k, v in base.items():
                out.setdefault(f"{tag}|base|{k}", []).append(v)

            # M2 on top of simple-mean centring
            Ac2, Aw2, ok = m2_apply(Ac, Aw, P_z, ke, q, beta, n_it)
            r = eval_rep(Ac2, Aw2, wl, y, P_z)
            for k, v in r.items():
                out.setdefault(f"{tag}|M2|{k}", []).append(v)
            out.setdefault(f"{tag}|M2_applied", []).append(float(ok))

            # M2 on top of M1 centring
            Am, Awm, wl2, _ = build_rep(cls, Z, cidx, ev, cal, "m1",
                                        P_raw, P_z, hp_m1)
            r = eval_rep(Am, Awm, wl2, y, P_z)
            for k, v in r.items():
                out.setdefault(f"{tag}|M1|{k}", []).append(v)
            Am2, Awm2, _ = m2_apply(Am, Awm, P_z, ke, q, beta, n_it)
            r = eval_rep(Am2, Awm2, wl2, y, P_z)
            for k, v in r.items():
                out.setdefault(f"{tag}|M1+M2|{k}", []).append(v)

            # oracle rotation, two forms.
            #  matched : the SELECTED configuration with true labels — says how
            #            much better M2 would be if only its labels were right.
            #  ceiling : the best configuration (widest subspace, no shrinkage)
            #            with true labels — the headroom of the MECHANISM, which
            #            a validation-selected beta would otherwise hide.
            Ao, Awo, _ = m2_apply(Ac, Aw, P_z, ke, q, beta, n_it, y_true=y)
            r = eval_rep(Ao, Awo, wl, y, P_z)
            for k, v in r.items():
                out.setdefault(f"{tag}|oracle|{k}", []).append(v)
            Ao, Awo, _ = m2_apply(Ac, Aw, P_z, max(KEXTRA), 1.0, 1.0, 3,
                                  y_true=y)
            r = eval_rep(Ao, Awo, wl, y, P_z)
            for k, v in r.items():
                out.setdefault(f"{tag}|oracleMax|{k}", []).append(v)

            # existing methods, same protocol
            bl = baseline_reps(cls, Z, cidx, ev, cal, P_z, mu_tr_z,
                               {"t3a_q": t3a_q})
            for nm, (Ab, Abw, Pb, wlb) in bl.items():
                r = eval_rep(Ab, Abw, wlb, y, Pb)
                for k, v in r.items():
                    out.setdefault(f"{tag}|{nm}|{k}", []).append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prefix", default=CFG.ckpt_prefix)
    ap.add_argument("--seconds", type=float, nargs="+", default=[20, 40])
    ap.add_argument("--mode", default="prefix", choices=["prefix", "spread"],
                    help="캘리브레이션 창 위치.  0단계 (c) 가 확정한 최강 기준선은 "
                         "'spread' 이므로, M2 가 그 위에서도 남는지 보려면 이것을 쓴다.")
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--sel_metric", default="proto_win")
    ap.add_argument("--m1_hp", type=float, nargs=3, default=[0.2, 3, 0.5],
                    help="tau iter gamma — M1 의 초매개변수 (analyze_m1 이 검증으로 "
                         "고른 최빈값을 넣는다)")
    ap.add_argument("--out", default=f"results/{CFG.prefix('m2')}.npz")
    args = ap.parse_args()
    hp_m1 = (args.m1_hp[0], int(args.m1_hp[1]), args.m1_hp[2])

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
        out = run_checkpoint(cls, meta, lab, head, ts, args, hp_m1)
        HP[(ts, sd)] = (out["hp_m2"], out["hp_t3a_q"])
        for k, v in out.items():
            if isinstance(v, list):
                R[k][ts].append(float(np.mean(v)))
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  M2 hp={out['hp_m2']} "
              f"(val {out['hp_m2_val']:.3f}) T3A q={out['hp_t3a_q']}  ||  "
              f"T20 base {np.mean(out['T20|base|proto_win']):.3f} "
              f"M2 {np.mean(out['T20|M2|proto_win']):.3f} "
              f"oracle {np.mean(out['T20|oracle|proto_win']):.3f}", flush=True)

    def arr(key):
        return np.array([np.mean(R[key][s]) for s in sorted(R[key])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    report(arr, R, HP, args)


def paired(a, b, label):
    d = a - b
    lo, hi = bootstrap_ci(d)
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    print(f"  {label:<40} {b.mean():.4f} -> {a.mean():.4f}  Δ{d.mean():+.4f} "
          f"CI[{lo:+.4f},{hi:+.4f}]{'  0포함' if lo <= 0 <= hi else '       '} "
          f"p={p:.4f} {int((d > 0).sum())}/{len(d)}")


def report(arr, R, HP, args):
    W = 106
    print(f"\n{'='*W}\n선택된 초매개변수 ({args.sel_metric}, 검증 피험자 2명)\n{'='*W}")
    hs = [v[0] for v in HP.values()]
    for nm, i, vals in (("k_extra", 0, KEXTRA), ("q", 1, QS),
                        ("beta", 2, BETAS), ("iter", 3, ITERS)):
        print(f"  {nm:<9}" + "  ".join(
            f"{v}:{sum(1 for h in hs if h[i] == v)}" for v in vals))
    qs = [v[1] for v in HP.values()]
    print("  T3A q   " + "  ".join(f"{v}:{qs.count(v)}" for v in T3A_QS))

    tags = [f"T{s:g}" for s in args.seconds] + ["Tfull"]
    methods = ["base", "M1", "M2", "M1+M2", "adaBN", "LA", "T3A",
               "oracle", "oracleMax"]
    for metric in ("proto_win", "proto_clip"):
        print(f"\n{'='*W}\n{metric}   기준선(base) = 조건4 단순 평균 중심화\n"
              f"결정 규칙 비교 -> 판정은 짝지은 검정 (재학습용 +0.0386 은 쓰지 않는다)"
              f"\n{'='*W}")
        print(f"  {'방법':<10}" + "".join(f"{t:>22}" for t in tags))
        for m in methods:
            cells = []
            for t in tags:
                k = f"{t}|{m}|{metric}"
                if k not in R:
                    cells.append(f"{'-':>22}")
                    continue
                a, b = arr(k), arr(f"{t}|base|{metric}")
                d = a - b
                try:
                    pv = stats.wilcoxon(a, b).pvalue if m != "base" else np.nan
                except ValueError:
                    pv = np.nan
                tail = "" if m == "base" else f" {d.mean():+.4f} p{pv:.3f}"
                cells.append(f"{a.mean():.4f}{tail:>15}")
            print(f"  {m:<10}" + "".join(f"{c:>22}" for c in cells))

    print(f"\n{'='*W}\noracle 회수율 (M2 가 oracle 회전 이득의 몇 %를 얻나)\n{'='*W}")
    for metric in ("proto_win", "proto_clip"):
        for t in tags:
            b = arr(f"{t}|base|{metric}").mean()
            o = arr(f"{t}|oracle|{metric}").mean()
            om = arr(f"{t}|oracleMax|{metric}").mean()
            m2 = arr(f"{t}|M2|{metric}").mean()
            f1 = (m2 - b) / (o - b) * 100 if abs(o - b) > 1e-9 else float("nan")
            f2 = (m2 - b) / (om - b) * 100 if abs(om - b) > 1e-9 else float("nan")
            print(f"  {metric:<11}{t:<8} base {b:.4f}  M2 {m2:.4f}  "
                  f"oracle(선택설정) {o:.4f} 회수 {f1:6.1f}%   "
                  f"oracle(기전상한) {om:.4f} 회수 {f2:6.1f}%")

    print(f"\n{'='*W}\n피험자별: M2 가 기준선보다 나빠지는 경우 (confirmation bias)\n{'='*W}")
    for metric in ("proto_win", "proto_clip"):
        b = arr(f"Tfull|base|{metric}")
        m2 = arr(f"Tfull|M2|{metric}")
        d = m2 - b
        bad = [(i + 1, b[i], d[i]) for i in range(len(b)) if d[i] < 0]
        rho, pv = stats.spearmanr(b, d)
        print(f"  [{metric}] 나빠진 피험자 {len(bad)}/{len(b)}: "
              + ", ".join(f"S{s}({v:.3f}{dd:+.3f})" for s, v, dd in bad))
        print(f"    기준선 정확도 vs M2 이득  Spearman ρ={rho:+.3f} p={pv:.3f}"
              "  (음수면 잘 맞는 피험자일수록 M2 가 해롭다)")
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
