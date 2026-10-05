"""
분석 2: 현실적인 캘리브레이션 프로토콜 시뮬레이션

기존 중심화 결과(조건2·4)는 테스트 세션 **전체 45클립**의 평균을 뺀 transductive
설정이었다.  실전에서는 평가할 데이터를 미리 볼 수 없으므로, 캘리브레이션 전용
녹화를 따로 떼어야 한다.  여기서는 그 프로토콜을 그대로 흉내낸다.

  각 테스트 세션에서 **감정별 한 클립씩**(SEED 3, SEED-V 5) 캘리브레이션 전용으로
  떼고, 평가는 남은 클립(SEED 12, SEED-V 10)으로만 한다.  중심화 평균은 떼어낸 클립에서만 만든다 —
  평가 데이터는 평균 계산에 절대 들어가지 않는다.

캘리브레이션 길이:
  떼어낸 클립에서 클립마다 **앞부분 T초씩**.  창이 4초 단위이므로 T초는
  ceil(T/4) 창으로 올림되고, 실제 사용량을 함께 보고한다 (T=2 는 4초가 된다).

대조:
  (a) 중심화 없음            조건1(head) / 조건3(prototype)
  (b) 남은 클립 자체로 중심화  기존 transductive 방식 = 상한
  (c) 한 감정 클립 전체만     단일 상태 캘리브레이션 (휴식 상태 녹화의 대리)

평가 클립 집합은 모든 조건에서 동일하므로 비교는 **결정 규칙 비교**다 (재학습 없음).
어느 클립을 떼는지 무작위 20회 반복 평균.  A팔 캐시만 쓴다.  GPU 없음.
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

from scipy import stats

from calib_common import cond_key, pick_from_clip


def extra_metrics(pred, true, prefix):
    """macro-F1 과 balanced accuracy.

    SEED 는 클립 라벨이 225/225/225 로 완전 균형이라 **clip 의 balanced accuracy 는
    정확도와 정확히 같다**.  창 수는 클래스별로 1.058배까지 차이나므로 window 에서는
    다르다.  그래도 두 열을 모두 두는 이유는 대조할 때 계산 방식을 명시하기 위해서다."""
    from sklearn.metrics import f1_score, balanced_accuracy_score
    return {f"{prefix}_f1": float(f1_score(true, pred, average="macro")),
            f"{prefix}_bal": float(balanced_accuracy_score(true, pred))}
from analyze_centering import (N_CLS, FS, SEG, roles_for_fold, clip_index, l2,
                               clip_reduce, all_domains, domain_means,
                               head_probs, load_head, bootstrap_ci)

SEC_PER_WIN = SEG / FS                      # 4.0 s


def tkey(sec):
    """Canonical protocol key: 2 and 2.0 must not become different keys."""
    return "Tfull" if sec == "full" else f"T{float(sec):g}"


def eff_seconds(sec):
    """Seconds actually used: windows are 4 s, so a request rounds UP."""
    return max(1, int(np.ceil(sec / SEC_PER_WIN))) * SEC_PER_WIN


def _train_side(cls, meta, lab, train):
    """Everything about the training subjects: head shift target and prototypes."""
    cidx = clip_index(meta)
    doms = all_domains(meta)
    tr_doms = [d for d in doms if d[0] in train]
    mu_all = domain_means(cls, meta, doms)
    mu_tr_global = np.mean([mu_all[d] for d in tr_doms], axis=0)

    Z = l2(cls)
    tr_keys = sorted(k for k in cidx if k[0] in train)
    B = clip_reduce(Z, cidx, tr_keys)
    y_tr = np.array([lab[cidx[k][0]] for k in tr_keys])
    db = np.array([(k[0], k[1]) for k in tr_keys])
    for d in tr_doms:
        m = (db[:, 0] == d[0]) & (db[:, 1] == d[1])
        B[m] -= B[m].mean(0)
    P = np.stack([np.nanmean(
        [B[(db[:, 0] == d[0]) & (db[:, 1] == d[1]) & (y_tr == k)].mean(0)
         for d in tr_doms], axis=0) for k in range(N_CLS)])
    return mu_tr_global, P


def _score(cls, Zall, head, cidx, eval_keys, y_c, mu_raw, mu_z, P,
           mu_tr_global, centred):
    """Conditions 2/4 (centred) or 1/3 (not) on a fixed clip set.

    ``mu_raw`` / ``mu_z`` map a test domain to its centring mean; when
    ``centred`` is False they are ignored.  Returns
    {"head_win", "head_clip", "proto_win", "proto_clip"}."""
    pw_h, pc_h, pw_p, pc_p, yw = [], [], [], [], []
    for i, k in enumerate(eval_keys):
        w = cidx[k]
        d = (k[0], k[1])
        Xh = cls[w] - mu_raw[d] + mu_tr_global if centred else cls[w]
        Ph = head_probs(head, Xh)
        pw_h.append(Ph.argmax(1))
        pc_h.append(Ph.mean(0).argmax())

        Zw = Zall[w] - mu_z[d] if centred else Zall[w]
        pw_p.append((l2(Zw) @ l2(P).T).argmax(1))
        ec = Zall[w].mean(0) - (mu_z[d] if centred else 0.0)
        pc_p.append(int((l2(ec[None]) @ l2(P).T).argmax(1)[0]))
        yw.append(np.full(len(w), int(y_c[i])))
    yw = np.concatenate(yw)
    out = {
        "head_win": float((np.concatenate(pw_h) == yw).mean()),
        "head_clip": float((np.array(pc_h) == y_c).mean()),
        "proto_win": float((np.concatenate(pw_p) == yw).mean()),
        "proto_clip": float((np.array(pc_p) == y_c).mean()),
    }
    out.update(extra_metrics(np.concatenate(pw_h), yw, "head_win"))
    out.update(extra_metrics(np.array(pc_h), y_c, "head_clip"))
    out.update(extra_metrics(np.concatenate(pw_p), yw, "proto_win"))
    out.update(extra_metrics(np.array(pc_p), y_c, "proto_clip"))
    return out


def run_one(cls, meta, lab, head, test_subj, seconds, n_rep, rng,
            modes=("prefix",)):
    """One checkpoint -> {protocol: {metric: value}} averaged over repeats."""
    train, _, _ = roles_for_fold(test_subj)
    mu_tr_global, P = _train_side(cls, meta, lab, train)
    cidx = clip_index(meta)
    Zall = l2(cls)
    te_doms = [d for d in all_domains(meta) if d[0] == test_subj]

    # clips of the test subject, grouped by (domain, label)
    by = defaultdict(list)
    for k in sorted(cidx):
        if k[0] == test_subj:
            by[((k[0], k[1]), int(lab[cidx[k][0]]))].append(k)

    acc = defaultdict(list)
    for _ in range(n_rep):
        # one held-out clip per emotion per session
        held = {}
        for d in te_doms:
            held[d] = [by[(d, c)][rng.integers(len(by[(d, c)]))]
                       for c in range(N_CLS)]
        held_set = {k for v in held.values() for k in v}
        eval_keys = [k for k in sorted(cidx)
                     if k[0] == test_subj and k not in held_set]
        y_c = np.array([lab[cidx[k][0]] for k in eval_keys])

        def means_from(picker):
            """picker(domain) -> window indices used for that domain's mean."""
            raw, zz = {}, {}
            for d in te_doms:
                w = picker(d)
                raw[d] = cls[w].mean(0)
                zz[d] = Zall[w].mean(0)
            return raw, zz

        # (a) no centering
        r = _score(cls, Zall, head, cidx, eval_keys, y_c, None, None, P,
                   mu_tr_global, centred=False)
        for m, v in r.items():
            acc[f"none|{m}"].append(v)

        # (b) transductive: the 12 evaluation clips themselves
        raw, zz = means_from(lambda d: np.concatenate(
            [cidx[k] for k in eval_keys if (k[0], k[1]) == d]))
        r = _score(cls, Zall, head, cidx, eval_keys, y_c, raw, zz, P,
                   mu_tr_global, centred=True)
        for m, v in r.items():
            acc[f"transductive|{m}"].append(v)

        # (c) single-state: one emotion's held-out clip only (whole clip)
        for c in range(N_CLS):
            raw, zz = means_from(lambda d, c=c: np.array(cidx[held[d][c]]))
            r = _score(cls, Zall, head, cidx, eval_keys, y_c, raw, zz, P,
                       mu_tr_global, centred=True)
            for m, v in r.items():
                acc[f"single{c}|{m}"].append(v)

        # the protocol itself: T s out of each held-out clip, taken from the
        # clip's start ('prefix') or spread evenly over it ('spread').  A prefix
        # sits entirely in the clip's opening seconds, where the emotional
        # response has not developed yet, so the two are not interchangeable.
        for mode in modes:
            for sec in list(seconds) + ["full"]:
                if sec == "full" and mode != modes[0]:
                    continue                  # whole clip: the modes coincide
                def pick(d, sec=sec, mode=mode):
                    out = []
                    for k in held[d]:
                        w = cidx[k]                   # already time-ordered
                        n = len(w) if sec == "full" else max(
                            1, int(np.ceil(sec / SEC_PER_WIN)))
                        out.append(pick_from_clip(w, n, mode))
                    return np.concatenate([np.asarray(x) for x in out])
                raw, zz = means_from(pick)
                r = _score(cls, Zall, head, cidx, eval_keys, y_c, raw, zz, P,
                           mu_tr_global, centred=True)
                for m, v in r.items():
                    acc[f"{cond_key(sec, mode)}|{m}"].append(v)

    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prefix", default=CFG.ckpt_prefix)
    ap.add_argument("--seconds", type=float, nargs="+",
                    default=[2, 5, 10, 20, 40])
    ap.add_argument("--n_rep", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=["prefix"],
                    help="prefix / spread — 캘리브레이션 창을 클립 앞부분에서 "
                         "뽑을지 클립 전체에 고르게 뽑을지")
    ap.add_argument("--out", default=f"results/{CFG.prefix('calib_protocol')}.npz")
    args = ap.parse_args()

    R = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)",
                                    os.path.basename(f)).groups())
        z = np.load(f)
        cls, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
        head = load_head(os.path.join(args.ckpt_dir,
                                      f"{args.prefix}S{ts}_seed{sd}.pt"))
        rng = np.random.default_rng(1000 * ts + sd)      # reproducible per run
        r = run_one(cls, meta, lab, head, ts, args.seconds, args.n_rep, rng,
                    tuple(args.modes))
        for k, v in r.items():
            R[k][ts].append(v)
        k0 = tkey(args.seconds[0])
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  "
              f"없음 {r['none|proto_clip']:.3f}  {k0} {r[k0+'|proto_clip']:.3f}  "
              f"Tfull {r['Tfull|proto_clip']:.3f}  "
              f"transductive {r['transductive|proto_clip']:.3f}", flush=True)

    def arr(key):
        return np.array([np.mean(R[key][s]) for s in sorted(R[key])])

    W = 100
    labels = ([("none", "중심화 없음 (대조 a)")]
              + [(cond_key(s, m),
                  f"감정별 {eff_seconds(s):g}초 x{N_CLS}클립 = {N_CLS*eff_seconds(s):g}초"
                  f" [{'앞부분' if m == 'prefix' else '고르게'}]"
                  + (f"  (요청 {s:g}s -> 올림)" if eff_seconds(s) != s else ""))
                 for m in args.modes for s in args.seconds]
              + [("Tfull", f"떼어낸 {N_CLS}클립 전체 (클립 길이 중앙 "
                           f"{CFG.median_clip_sec:g}초)")]
              + [(f"single{c}", f"단일 상태: 클래스 {c} 클립만 (대조 c)")
                 for c in range(N_CLS)]
              + [("transductive", f"{CFG.n_eval_clips}클립 자체로 중심화 (대조 b, 상한)")])

    for metric, nm in (("proto_clip", "조건4 prototype  clip"),
                       ("proto_win", "조건4 prototype  window"),
                       ("head_clip", "조건2 head  clip"),
                       ("head_win", "조건2 head  window")):
        print(f"\n{'='*W}\n{nm}   (n={CFG.n_subjects}, 시드 평균 x 반복 {args.n_rep} 평균)\n{'='*W}")
        base = arr(f"none|{metric}")
        top = arr(f"transductive|{metric}")
        print(f"  {'프로토콜':<40}{'정확도':>10}{'없음 대비':>12}"
              f"{'상한 대비':>12}{'p (vs 없음)':>14}  이긴 수")
        for key, lb in labels:
            a = arr(f"{key}|{metric}")
            d = a - base
            try:
                p = stats.wilcoxon(a, base).pvalue if key != "none" else float("nan")
            except ValueError:
                p = float("nan")
            print(f"  {lb:<40}{a.mean():>10.4f}{d.mean():>+12.4f}"
                  f"{a.mean()-top.mean():>+12.4f}{p:>14.4f}"
                  f"  {int((d > 0).sum()) if key != 'none' else '-'}/{len(d)}")

    print(f"\n{'='*W}\n짝지은 비교 (조건4 clip 기준, 결정 규칙 비교)\n{'='*W}")

    def cmp(a_key, b_key, label):
        a, b = arr(f"{a_key}|proto_clip"), arr(f"{b_key}|proto_clip")
        d = a - b
        lo, hi = bootstrap_ci(d)
        try:
            p = stats.wilcoxon(a, b).pvalue
        except ValueError:
            p = float("nan")
        print(f"  {label:<52} Δ{d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]"
              f"{'  0포함' if lo <= 0 <= hi else '       '} p={p:.4f} "
              f"{int((d > 0).sum())}/{len(d)}")

    s0 = tkey(args.seconds[0])
    cmp(s0, "none", f"{s0} 프로토콜 ({3*eff_seconds(args.seconds[0]):g}초)  vs 중심화 없음")
    cmp("Tfull", s0, f"{N_CLS}클립 전체  vs {s0}")
    cmp("transductive", "Tfull",
        f"{CFG.n_eval_clips}클립 transductive  vs {N_CLS}클립 전체")
    for c in range(N_CLS):
        cmp(f"single{c}", "Tfull",
            f"단일 상태({CFG.class_names[c]})  vs {N_CLS}클립 전체")
    print("  단일 상태 평균:")
    a = np.mean([arr(f"single{c}|proto_clip") for c in range(N_CLS)], axis=0)
    b = arr("Tfull|proto_clip")
    d = a - b
    lo, hi = bootstrap_ci(d)
    print(f"  {f'{N_CLS}클래스 평균  vs {N_CLS}클립 전체':<52} Δ{d.mean():+.4f} "
          f"CI[{lo:+.4f},{hi:+.4f}]{'  0포함' if lo <= 0 <= hi else '       '} "
          f"p={stats.wilcoxon(a, b).pvalue:.4f} {int((d > 0).sum())}/{len(d)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
