"""
M1: 감정 편향을 뺀 중심화 기준점 추정.

단순 평균 중심화는 캘리브레이션 창이 담은 감정의 클래스 방향을 도메인 오프셋으로
착각한다.  감정이 균형이면 클래스 방향들이 상쇄되지만, 한 감정만 들어 있으면
μ 는 오프셋 + 그 클래스 방향이 되고 그것을 빼면 그 클래스 신호까지 지워진다
(S8: 단일 감정 캘리브레이션은 SEED 15/15 실패, 중심화를 안 하는 것보다 나쁨).

M1 은 클래스 확률로 그 성분을 빼면서 μ 를 다시 추정한다.  따라서 **효과가 나타날
곳은 오염 시나리오**이고, 균형 시나리오에서는 남는 오차가 추정 잡음이라 효과가
작을 수 있다.  두 시나리오를 섞어 평균내면 둘 다 흐려지므로 **따로 판정한다.**

시나리오
--------
[오염]  단일 감정 클립 (클래스 0/1/2 각각)
        세션 앞부분 연속 30/60/120초  (모두 첫 클립 안이므로 감정 하나)
        온라인 인과적 누적            (시간 곡선)
[균형]  감정별 20초 / 40초 / 감정별 1클립 전체 (총 시간은 클래스 수 배)

τ 와 반복 횟수는 각 fold 의 **검증 피험자 2명**으로 고른다 — 테스트를 보고 고르면
상한을 재는 셈이 된다.

결정 규칙 비교이므로 판정은 짝지은 검정으로 한다 (재학습용 +0.0386 은 쓰지 않는다).
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
from adapt_methods import m1_mean, train_prototypes_raw
from calib_common import (SEC_PER_WIN, calib_indices, clips_by_domain_label,
                          draw_holdout, n_windows)

TAUS = [0.05, 0.1, 0.2, 0.5]
ITERS = [1, 3]
# γ 는 학습 prototype 이 테스트 피험자의 클래스 편차보다 약 2배 길다는 측정에서
# 나온다 (adapt_methods.m1_mean 참조).  0 은 단순 평균과 같으므로 넣지 않는다.
GAMMAS = [0.25, 0.5, 0.75, 1.0]


# ── scoring ─────────────────────────────────────────────────────────────────

def score_with_mu(cls, Z, head, cidx, eval_keys, y_c, mu_raw, mu_z,
                  P_z, mu_tr_global, proto_only=False):
    """조건2(head)·조건4(prototype) 를 주어진 μ 로 채점.  window/clip 둘 다.

    ``proto_only`` 는 초매개변수 탐색용 — head 경로는 창마다 순전파가 필요해서
    그리드 전체에 돌리면 이 분석의 대부분을 잡아먹는다."""
    Pz = l2(P_z)
    pw_h, pc_h, pw_p, pc_p, yw = [], [], [], [], []
    for i, k in enumerate(eval_keys):
        w = cidx[k]
        d = (k[0], k[1])
        if not proto_only:
            Ph = head_probs(head, cls[w] - mu_raw[d] + mu_tr_global)
            pw_h.append(Ph.argmax(1))
            pc_h.append(int(Ph.mean(0).argmax()))
        pw_p.append((l2(Z[w] - mu_z[d]) @ Pz.T).argmax(1))
        ec = Z[w].mean(0) - mu_z[d]
        pc_p.append(int((l2(ec[None]) @ Pz.T).argmax(1)[0]))
        yw.append(np.full(len(w), int(y_c[i])))
    yw = np.concatenate(yw)
    out = {"proto_win": float((np.concatenate(pw_p) == yw).mean()),
           "proto_clip": float((np.array(pc_p) == y_c).mean())}
    if not proto_only:
        out["head_win"] = float((np.concatenate(pw_h) == yw).mean())
        out["head_clip"] = float((np.array(pc_h) == y_c).mean())
    return out


def mus(cls, Z, calib, method, P_raw, P_z, hp):
    """{domain: mean} for the raw path (head) and the normalised path (prototype).

    Each path must use the prototypes OF ITS OWN SPACE with their natural
    magnitude.  Passing ``l2(P_raw)`` to the normalised path — unit vectors —
    subtracts something ~10x too long and destroys the method; that was the
    first version's bug.
    """
    tau, n_iter, gamma = hp
    raw, zz = {}, {}
    for d, idx in calib.items():
        idx = np.asarray(idx)
        if method == "mean":
            raw[d] = cls[idx].mean(0)
            zz[d] = Z[idx].mean(0)
        else:
            raw[d] = m1_mean(cls[idx], P_raw, tau=tau, n_iter=n_iter,
                             gamma=gamma)
            zz[d] = m1_mean(Z[idx], P_z, tau=tau, n_iter=n_iter, gamma=gamma)
    return raw, zz


# ── scenarios ───────────────────────────────────────────────────────────────

def balanced_scenarios(held, cidx, te_doms, secs):
    out = {}
    for s in secs:
        out[f"bal{s:g}"] = calib_indices(held, cidx, te_doms, s, "prefix")
    out["balfull"] = calib_indices(held, cidx, te_doms, "full", "prefix")
    return out


def single_emotion_scenarios(held, cidx, te_doms):
    """한 감정 클립 전체만으로 캘리브레이션 (클래스별)."""
    return {f"single{c}": {d: list(cidx[held[d][c]]) for d in te_doms}
            for c in range(N_CLS)}


def prefix_scenarios(cidx, te_doms, secs):
    """세션 앞부분 연속 N초.  **첫 클립 안에 들어가야 감정이 하나다** — 그것이
    이 시나리오(오염된 캘리브레이션)의 전제다.

    SEED 는 첫 클립이 232초로 균일해서 전제가 자동으로 성립했다.  SEED-V 는
    첫 클립이 72~236초로 흩어지므로 전제를 **검사한다** — 주석으로만 두면 예산을
    올렸을 때 조용히 두 감정이 섞인 채로 숫자가 나온다.
    """
    out = {}
    for s in secs:
        n = n_windows(s)
        cal = {}
        for d in te_doms:
            keys = [k for k in sorted(cidx) if (k[0], k[1]) == d]
            first = cidx[keys[0]]
            if n > len(first):
                raise ValueError(
                    f"앞 {s:g}초({n}창)가 도메인 {d} 의 첫 클립({len(first)}창 "
                    f"= {4*len(first):g}초)을 넘는다 — 이 시나리오는 감정 하나를 "
                    f"전제하므로 예산을 줄이거나 시나리오를 다시 정의할 것")
            w = np.concatenate([cidx[k] for k in keys])
            cal[d] = list(w[:n])
        out[f"pre{s:g}"] = cal
    return out


def prefix_eval_keys(cidx, subject, max_sec):
    """앞부분 캘리브레이션이 건드린 클립을 제외한 평가 클립.

    가장 긴 prefix 가 닿은 클립까지 제외해 모든 길이가 **같은 평가 집합**을 쓴다."""
    n = n_windows(max_sec)
    keys = [k for k in sorted(cidx) if k[0] == subject]
    drop = set()
    for d in sorted({(k[0], k[1]) for k in keys}):
        dk = [k for k in keys if (k[0], k[1]) == d]
        seen = 0
        for k in dk:
            if seen < n:
                drop.add(k)
            seen += len(cidx[k])
    return [k for k in keys if k not in drop]


# ── validation-based hyperparameter choice ──────────────────────────────────

GROUP_OF = {}          # scenario name -> "오염" / "균형", filled by group_of()


def group_of(name):
    """M1 은 오염 시나리오를 겨냥한다.  두 군을 한 그리드로 고르면 균형 쪽에서의
    손해가 오염 쪽에서의 이득을 상쇄해 τ 가 '보정 없음' 으로 끌려간다 — 실제로
    첫 실행에서 그렇게 됐다.  그래서 초매개변수도 **군별로** 고른다."""
    return "균형" if name.startswith("bal") else "오염"


def choose_hparams(cls, Z, meta, lab, cidx, head, ts, build, metric):
    """{군: (τ, 반복, γ)} — 각 fold 의 검증 피험자 2명으로 고른다.

    prototype 은 그 fold 의 학습 피험자에서만 만든다; 검증 피험자를 넣으면 자기
    자신으로 자기를 맞추게 된다."""
    train, val, _ = roles_for_fold(ts)
    per_val = {v: build(v, train) for v in val}
    best = {}
    for grp in ("오염", "균형"):
        bv, bh = -1.0, (TAUS[0], ITERS[0], GAMMAS[0])
        for tau in TAUS:
            for it in ITERS:
                for g in GAMMAS:
                    vals = []
                    for v, (P_z, mu_g, P_raw, scen, ev, y_c) in per_val.items():
                        for name, cal in scen.items():
                            if group_of(name) != grp:
                                continue
                            r_raw, r_z = mus(cls, Z, cal, "m1", P_raw, P_z,
                                             (tau, it, g))
                            vals.append(score_with_mu(
                                cls, Z, head, cidx, ev, y_c, r_raw, r_z,
                                P_z, mu_g, proto_only=True)[metric])
                    m = float(np.mean(vals))
                    if m > bv:
                        bv, bh = m, (tau, it, g)
        best[grp] = (bh, bv)
    return best


# ── main ────────────────────────────────────────────────────────────────────

def run_checkpoint(cls, meta, lab, head, ts, sd, args):
    """한 체크포인트의 모든 시나리오 결과."""
    cidx = clip_index(meta)
    Z = l2(cls)
    train, val, _ = roles_for_fold(ts)
    mu_tr_global, P_z = _train_side(cls, meta, lab, train)
    P_raw = train_prototypes_raw(cls, meta, lab, cidx, train)

    def build(subject, train_subjects):
        """(P_z, mu_tr_global, P_raw, scenarios, eval_keys, y_c) for one subject.

        Used for the test subject and, during hyperparameter selection, for each
        validation subject — always with prototypes from ``train_subjects``."""
        mg, pz = _train_side(cls, meta, lab, train_subjects)
        pr = train_prototypes_raw(cls, meta, lab, cidx, train_subjects)
        doms = sorted({(k[0], k[1]) for k in cidx if k[0] == subject})
        by = clips_by_domain_label(cidx, lab, subject)
        rng = np.random.default_rng(1000 * subject + 7)
        held, held_set = draw_holdout(by, doms, rng)
        ev = [k for k in sorted(cidx) if k[0] == subject and k not in held_set]
        y = np.array([lab[cidx[k][0]] for k in ev])
        scen = {}
        scen.update(balanced_scenarios(held, cidx, doms, args.bal_seconds))
        scen.update(single_emotion_scenarios(held, cidx, doms))
        return pz, mg, pr, scen, ev, y

    HP = choose_hparams(cls, Z, meta, lab, cidx, head, ts, build,
                        args.sel_metric)
    out = {f"hp_{g}": HP[g] for g in HP}

    # ── the test subject: same holdout draw family as S8/S9 ──────────────
    doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
    by = clips_by_domain_label(cidx, lab, ts)
    # same draws as calib_protocol / S8 / S9, so the evaluation clip sets
    # line up and the baselines there are directly comparable
    rng = np.random.default_rng(1000 * ts + sd)
    for _ in range(args.n_rep):
        held, held_set = draw_holdout(by, doms, rng)
        ev = [k for k in sorted(cidx) if k[0] == ts and k not in held_set]
        y = np.array([lab[cidx[k][0]] for k in ev])
        scen = {}
        scen.update(balanced_scenarios(held, cidx, doms, args.bal_seconds))
        scen.update(single_emotion_scenarios(held, cidx, doms))
        # no centring at all, on THIS evaluation set — the floor that a
        # contaminated calibration has to beat to be worth using
        zero0 = {d: np.zeros(cls.shape[1], np.float32) for d in doms}
        sc = score_with_mu(cls, Z, head, cidx, ev, y, zero0, zero0,
                           P_z, mu_tr_global)
        for k, v in sc.items():
            out.setdefault(f"holdnone|none|{k}", []).append(v)
        for name, cal in scen.items():
            hp = HP[group_of(name)][0]
            for meth in ("mean", "m1"):
                r_raw, r_z = mus(cls, Z, cal, meth, P_raw, P_z, hp)
                sc = score_with_mu(cls, Z, head, cidx, ev, y, r_raw, r_z,
                                   P_z, mu_tr_global)
                for k, v in sc.items():
                    out.setdefault(f"{name}|{meth}|{k}", []).append(v)

    # ── session-prefix scenarios: no holdout draw, the prefix IS the split ──
    ev = prefix_eval_keys(cidx, ts, max(args.pre_seconds))
    y = np.array([lab[cidx[k][0]] for k in ev])
    for name, cal in prefix_scenarios(cidx, doms, args.pre_seconds).items():
        hp = HP[group_of(name)][0]
        for meth in ("mean", "m1"):
            r_raw, r_z = mus(cls, Z, cal, meth, P_raw, P_z, hp)
            sc = score_with_mu(cls, Z, head, cidx, ev, y, r_raw, r_z,
                               P_z, mu_tr_global)
            for k, v in sc.items():
                out.setdefault(f"{name}|{meth}|{k}", []).append(v)
    # no centring at all, on the same evaluation set, as the floor
    zero = {d: np.zeros(cls.shape[1], np.float32) for d in doms}
    sc = score_with_mu(cls, Z, head, cidx, ev, y, zero, zero, P_z, mu_tr_global)
    for k, v in sc.items():
        out.setdefault(f"prenone|none|{k}", []).append(v)

    # ── online causal accumulation ──────────────────────────────────────
    for meth in ("mean", "m1"):
        curve = causal_curve(cls, Z, head, cidx, lab, ts, doms, P_raw, P_z,
                             mu_tr_global, meth, HP["오염"][0],
                             args.curve_points)
        for t, acc in curve:
            out.setdefault(f"online{t:g}|{meth}|proto_win", []).append(acc)
    return out


def causal_curve(cls, Z, head, cidx, lab, ts, doms, P_raw, P_z, mu_tr_global,
                 method, hp, points):
    """캘리브레이션 녹화 없이 바로 시작하고 μ 를 주기적으로 갱신하는 경우.

    시점 t (세션의 앞 t 비율) 마다 **그때까지 본 창만으로** μ 를 만들고, 그 μ 로
    **다음 구간의 창**을 분류한다.  미래를 보지 않으므로 인과적이고, 시점을 12 개로
    제한하므로 창마다 μ 를 다시 추정하는 O(n^2) 를 피한다.

    S3 의 "인과적 누적" 은 창마다 갱신했으므로 이 곡선과 같은 값이 아니다 —
    갱신 주기가 다르다.  비교는 이 곡선 안에서 mean 대 M1 으로만 한다.

    초반 구간은 본 것이 한 감정뿐이므로 단순 평균이 크게 치우친다: M1 이 겨냥하는
    상황이고, 곡선이 그 차이를 시간에 따라 보여준다.
    """
    Pz = l2(P_z)
    per_t = {}
    for d in doms:
        w = np.concatenate([cidx[k] for k in sorted(cidx)
                            if (k[0], k[1]) == d])
        yw = lab[w]
        n = len(w)
        bounds = [max(1, int(round(f * n))) for f in points] + [n]
        for j, t in enumerate(bounds[:-1]):
            seen, nxt = w[:t], w[t:bounds[j + 1]]
            if len(nxt) == 0:
                continue
            mz = (Z[seen].mean(0) if method == "mean" else
                  m1_mean(Z[seen], P_z, tau=hp[0], n_iter=hp[1], gamma=hp[2]))
            pred = (l2(Z[nxt] - mz) @ Pz.T).argmax(1)
            key = round(t * SEC_PER_WIN / 60.0, 2)
            per_t.setdefault(key, []).append(float((pred == yw[t:bounds[j + 1]]
                                                    ).mean()))
    return [(k, float(np.mean(v))) for k, v in sorted(per_t.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prefix", default=CFG.ckpt_prefix)
    ap.add_argument("--bal_seconds", type=float, nargs="+", default=[20, 40])
    ap.add_argument("--pre_seconds", type=float, nargs="+",
                    default=[30, 60, 120])
    ap.add_argument("--curve_points", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--sel_metric", default="proto_win")
    ap.add_argument("--out", default=f"results/{CFG.prefix('m1')}.npz")
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
        out = run_checkpoint(cls, meta, lab, head, ts, sd, args)
        HP[(ts, sd)] = {g: out[f"hp_{g}"] for g in ("오염", "균형")}
        for k, v in out.items():
            if isinstance(v, list):
                R[k][ts].append(float(np.mean(v)))
        h1, h2 = HP[(ts, sd)]["오염"][0], HP[(ts, sd)]["균형"][0]
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  오염 τ{h1[0]} i{h1[1]} "
              f"γ{h1[2]} | 균형 τ{h2[0]} i{h2[1]} γ{h2[2]}  ||  "
              f"single1 {np.mean(out['single1|mean|proto_clip']):.3f}->"
              f"{np.mean(out['single1|m1|proto_clip']):.3f}  "
              f"bal20 {np.mean(out['bal20|mean|proto_clip']):.3f}->"
              f"{np.mean(out['bal20|m1|proto_clip']):.3f}", flush=True)

    def arr(key):
        return np.array([np.mean(R[key][s]) for s in sorted(R[key])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    report(arr, R, HP, args)


def paired(a, b, label):
    d = a - b
    lo, hi = bootstrap_ci(d)
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    print(f"  {label:<44} {b.mean():.4f} -> {a.mean():.4f}  Δ{d.mean():+.4f} "
          f"CI[{lo:+.4f},{hi:+.4f}]{'  0포함' if lo <= 0 <= hi else '       '} "
          f"p={p:.4f} {int((d > 0).sum())}/{len(d)}")
    return d.mean(), lo, hi, p


def report(arr, R, HP, args):
    W = 108
    print(f"\n{'='*W}\n선택된 초매개변수 (검증 피험자 2명, {args.sel_metric})\n{'='*W}")
    n = len(HP)
    for grp in ("오염", "균형"):
        hs = [v[grp][0] for v in HP.values()]
        print(f"  [{grp}]  τ " + " ".join(
            f"{t}:{sum(1 for h in hs if h[0]==t)}" for t in TAUS) +
            "   iter " + " ".join(
            f"{i}:{sum(1 for h in hs if h[1]==i)}" for i in ITERS) +
            "   γ " + " ".join(
            f"{g}:{sum(1 for h in hs if h[2]==g)}" for g in GAMMAS) +
            f"   (총 {n}회)")

    groups = [
        ("[오염] 단일 감정 클립", [(f"single{c}", f"클래스 {c} 클립만")
                             for c in range(N_CLS)]),
        ("[오염] 세션 앞부분 연속", [(f"pre{s:g}", f"앞 {s:g}초")
                              for s in args.pre_seconds]),
        ("[균형] 감정별 분산", [(f"bal{s:g}", f"감정별 {s:g}초(총 {N_CLS*s:g}초)")
                           for s in args.bal_seconds] +
                          [("balfull", f"{N_CLS}클립 전체")]),
    ]
    for metric in ("proto_win", "proto_clip", "head_win", "head_clip"):
        cond = "조건4 prototype" if metric.startswith("proto") else "조건2 head"
        unit = "window" if metric.endswith("win") else "clip"
        print(f"\n{'='*W}\n{cond}  {unit}   단순 평균 -> M1   "
              f"(결정 규칙 비교 = 짝지은 검정)\n{'='*W}")
        for title, items in groups:
            print(f"\n  {title}")
            for key, nm in items:
                ka, kb = f"{key}__m1__{metric}", f"{key}__mean__{metric}"
                if f"{key}|m1|{metric}" not in R:
                    continue
                paired(arr(f"{key}|m1|{metric}"), arr(f"{key}|mean|{metric}"),
                       f"    {nm}")
            ref = ("prenone" if title.startswith("[오염] 세션") else "holdnone")
            if f"{ref}|none|{metric}" in R:
                fl = arr(f"{ref}|none|{metric}")
                print(f"    {'(참고) 중심화 없음, 같은 평가셋':<42} {fl.mean():.4f}")

    print(f"\n{'='*W}\n[오염] 온라인 인과적 누적 (조건4 window)\n{'='*W}")
    ts_keys = sorted({float(k.split('|')[0][6:]) for k in R
                      if k.startswith('online')})
    print(f"  {'경과':>8}{'단순 평균':>12}{'M1':>10}{'Δ':>10}{'p':>10}  이긴 수")
    for t in ts_keys:
        a, b = arr(f"online{t:g}|m1|proto_win"), arr(f"online{t:g}|mean|proto_win")
        d = a - b
        try:
            p = stats.wilcoxon(a, b).pvalue
        except ValueError:
            p = float("nan")
        print(f"  {t:>6.1f}분{b.mean():>12.4f}{a.mean():>10.4f}{d.mean():>+10.4f}"
              f"{p:>10.4f}  {int((d>0).sum())}/{len(d)}")
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
