"""
few-shot 라벨 곡선 — 라벨 붙은 영상 몇 편이면 라벨 상한에 닿나.

라벨 없는 최선(중심화 + 학습 prototype)에서 S2 의 라벨 상한(`ceil_logreg`,
테스트 피험자 자기 클립 라벨 + 로지스틱 회귀)까지의 간격을, **감정별 영상 k편**으로
얼마나 메우는지 잰다.  라벨은 영상 자극 라벨이므로 "라벨 클립 k개" =
"감정별 영상 k편 시청" 이고, 사용자 주석은 필요 없다.

프로토콜 (S9 분석 A 와 같은 방향: 평가를 떼고 나머지를 풀로 쓴다)
  각 테스트 세션에서 **감정별 1클립씩**(SEED 3, SEED-V 5) 평가용으로 고정해 떼고,
  나머지(SEED 12, SEED-V 10)를 라벨 캘리브레이션 풀로 쓴다.
  풀에서 감정별 k클립을 뽑는다 (k = 1..max_k; SEED 4, SEED-V 2).  무작위 20회 반복.
  중심화 μ 는 **뽑힌 클립의 창으로만** 계산한다 (라벨 불필요).
  평가 클립은 μ 에도, 어떤 학습에도 쓰지 않는다.

평가 세트가 작으므로(SEED 피험자당 9클립, SEED-V 15클립)
**window 정확도를 주 지표**로 하고
clip 은 참고로 둔다.

방법 (모두 뽑힌 클립의 창과 라벨만 사용)
  a_none   중심화 + 학습 prototype                       기준선, 라벨 불필요
  b_mix    중심화 + 사용자 prototype 혼합 (S11 L3, 길이 보정)
  c_lr     중심화 + 뽑힌 창으로 학습한 로지스틱 회귀 (규제 없음)
  c_lrreg  같은 것에 **학습 prototype 쪽 L2 규제**
  d_head   중심화 + head 마지막 층 적응 (S11 L4)

초매개변수(λ, 규제 세기, lr/스텝)는 각 fold 의 **검증 피험자 2명**으로 고른다.
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

from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               head_probs, load_head, bootstrap_ci)
from analyze_calib_protocol import _train_side
from analyze_labelcalib import l4_head   # S11 L4 와 같은 구현을 쓴다
from calib_common import clips_by_domain_label

# 이 분석의 텐서는 (수백 x 200) 으로 작다.  그 크기에서는 스레드 핸드오프가
# 계산보다 비싸다: 400 Adam 스텝이 4스레드 2.36s, 1스레드 0.44s (5.4배).
torch.set_num_threads(1)
# 라벨 상한은 analyze_centering 이 데이터셋마다 계산한 ceil_logreg 다
# (SEED 0.8464, SEED-V 0.5431).  박아 두면 다른 데이터셋에서 분모가 틀린다.
CEILING = float(np.asarray(
    np.load(f"results/{CFG.prefix('centering_analysis')}.npz")["ceil_logreg"]
).mean())
# 감정별 클립 k 개를 쓰므로, 평가용 1편을 뺀 나머지가 상한이다
# (SEED 5-1=4, SEED-V 3-1=2).  박아 두면 SEED-V 에서 k=3,4 가 조용히
# k=2 와 같은 집합으로 떨어져 중복 수치가 결과 파일에 들어간다.
KS = list(range(1, CFG.max_k + 1))
LAMBDAS = [0.25, 0.5, 0.75, 1.0]
LR_REGS = [0.0, 0.1, 1.0]
L4_GRID = [(1e-3, 20, 1e-1), (1e-3, 60, 1e-2), (3e-3, 20, 1e-1)]


def draw_eval_and_pool(by, doms, rng):
    """감정별 1클립을 평가용으로 떼고 나머지를 풀로 돌려준다."""
    ev, pool = {}, {}
    for d in doms:
        ev[d] = [by[(d, c)][rng.integers(len(by[(d, c)]))] for c in range(N_CLS)]
        pool[d] = {c: [k for k in by[(d, c)] if k not in ev[d]]
                   for c in range(N_CLS)}
    return ev, pool


def pick_k(pool, doms, k, rng):
    """풀에서 감정별 k클립.  k 가 풀 크기 이상이면 전부."""
    out = {}
    for d in doms:
        per = {}
        for c in range(N_CLS):
            cand = pool[d][c]
            if k >= len(cand):
                per[c] = list(cand)
            else:
                idx = rng.choice(len(cand), size=k, replace=False)
                per[c] = [cand[i] for i in sorted(idx)]
        out[d] = per
    return out


def centred(Z, cidx, keys_by_dom, sel):
    """뽑힌 클립의 창으로 도메인별 μ 를 만들고, 평가 창을 중심화한다."""
    mu = {}
    for d, per in sel.items():
        idx = np.concatenate([cidx[k] for c in per for k in per[c]])
        mu[d] = Z[idx].mean(0)
    return mu


def lr_fit(X, y, P, reg, steps=200, lr=1e-2):
    """중심화된 창으로 선형 분류기.  ``reg>0`` 이면 가중치를 학습 prototype 쪽으로 당긴다.

    prototype 은 코사인 분류기의 가중치와 같은 역할을 하므로, 그쪽으로 규제하면
    "라벨이 적을 때는 학습 prototype 을 믿고, 많아지면 데이터를 믿는" 연속체가 된다."""
    d = X.shape[1]
    W = torch.zeros(N_CLS, d, requires_grad=True)
    b = torch.zeros(N_CLS, requires_grad=True)
    P0 = torch.from_numpy(l2(P).astype(np.float32))
    with torch.no_grad():
        W.copy_(P0)                      # prototype 에서 출발
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float()
    yt = torch.from_numpy(np.ascontiguousarray(y)).long()
    opt = torch.optim.Adam([W, b], lr=lr)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(Xt @ W.T + b, yt)
        if reg > 0:
            loss = loss + reg * (W - P0).pow(2).sum()
        loss.backward()
        opt.step()
    return W.detach().numpy(), b.detach().numpy()


def score_all(cls, Z, head, cidx, ev_keys, y_c, mu, sel, P_z, mu_tr_global,
              hp, only=None):
    """``only`` 가 주어지면 그 방법만 계산한다 — 초매개변수 탐색에서 나머지 방법의
    로지스틱 적합과 head 미세조정이 매번 돌면 탐색이 분석 전체를 잡아먹는다."""
    want = set(only) if only else set(METHODS)
    """모든 방법을 한 평가 세트에서 채점.  window 주 지표, clip 병기."""
    Pn = l2(P_z)
    # 라벨 붙은 학습 재료 (중심화된 창)
    Xl, yl, Xl_raw = [], [], []
    for d, per in sel.items():
        for c, keys in per.items():
            for k in keys:
                w = cidx[k]
                Xl.append(Z[w] - mu[d])
                Xl_raw.append(cls[w] - mu_raw_of(cls, cidx, sel)[d]
                              + mu_tr_global)
                yl.append(np.full(len(w), c))
    Xl = np.concatenate(Xl)
    Xl_raw = np.concatenate(Xl_raw)
    yl = np.concatenate(yl)

    # 평가 창
    A, Aw_len, yw = [], [], []
    for i, k in enumerate(ev_keys):
        w = cidx[k]
        A.append(Z[w] - mu[(k[0], k[1])])
        Aw_len.append(len(w))
        yw.append(np.full(len(w), int(y_c[i])))
    Aw = np.concatenate(A)
    yw = np.concatenate(yw)
    off, s = [], 0
    for n in Aw_len:
        off.append(slice(s, s + n))
        s += n

    def clip_from_win(pred_w):
        return np.array([np.bincount(pred_w[o], minlength=N_CLS).argmax()
                         for o in off])

    out = {}

    def rec(name, pw, pc):
        out[f"{name}|win"] = float((pw == yw).mean())
        out[f"{name}|clip"] = float((pc == y_c).mean())

    # (a) 라벨 없음
    if "a_none" in want:
        pass
    pw = (l2(Aw) @ Pn.T).argmax(1)
    pc = np.array([(l2(Aw[o].mean(0)[None]) @ Pn.T).argmax(1)[0] for o in off])
    rec("a_none", pw, pc)

    # (b) prototype 혼합
    if "b_mix" in want:
      Pm = l3_prototypes_from(Z, cidx, sel, mu, P_z, hp["lam"])
      Pmn = l2(Pm)
      pw = (l2(Aw) @ Pmn.T).argmax(1)
      pc = np.array([(l2(Aw[o].mean(0)[None]) @ Pmn.T).argmax(1)[0] for o in off])
      rec("b_mix", pw, pc)

    # (c) 로지스틱 회귀 — 규제 없음 / 학습 prototype 규제
    for tag, reg in (("c_lr", 0.0), ("c_lrreg", hp["lr_reg"])):
        if tag not in want:
            continue
        W, b = lr_fit(Xl, yl, P_z, reg)
        logits = Aw @ W.T + b
        pw = logits.argmax(1)
        pc = np.array([logits[o].mean(0).argmax() for o in off])
        rec(tag, pw, pc)

    # (d) head 마지막 층 적응
    if "d_head" in want:
        mr = mu_raw_of(cls, cidx, sel)
        h = l4_head(head, Xl_raw, yl, *hp["l4"])
        pwl, pcl = [], []
        for i, k in enumerate(ev_keys):
            w = cidx[k]
            Ph = head_probs(h, cls[w] - mr[(k[0], k[1])] + mu_tr_global)
            pwl.append(Ph.argmax(1))
            pcl.append(int(Ph.mean(0).argmax()))
        rec("d_head", np.concatenate(pwl), np.array(pcl))
    return out


def mu_raw_of(cls, cidx, sel):
    return {d: cls[np.concatenate([cidx[k] for c in per for k in per[c]])].mean(0)
            for d, per in sel.items()}


def l3_prototypes_from(Z, cidx, sel, mu, P_z, lam):
    """뽑힌 클립의 클래스 평균을 학습 prototype 과 섞는다 (길이 보정 포함).

    도메인별로 섞지 않고 **한 벌**로 만든다 — 라벨이 적을 때 도메인별로 쪼개면
    클래스당 표본이 더 줄어든다."""
    per_c = {c: [] for c in range(N_CLS)}
    for d, per in sel.items():
        for c, keys in per.items():
            for k in keys:
                per_c[c].append(Z[cidx[k]].mean(0) - mu[d])
    out = P_z.copy()
    for c in range(N_CLS):
        if not per_c[c]:
            continue
        m = np.mean(per_c[c], axis=0)
        n = np.linalg.norm(m)
        if n < 1e-9:
            continue
        m = m / n * np.linalg.norm(P_z[c])
        out[c] = lam * m + (1 - lam) * P_z[c]
    return out


# ── 검증 피험자로 초매개변수 선택 ───────────────────────────────────────────

def choose(cls, Z, meta, lab, cidx, head, ts, n_rep):
    """{k: 초매개변수} — fold 의 검증 피험자 2명으로, **k 마다 따로** 고른다.

    라벨이 몇 개인지에 따라 규제 세기를 바꾸는 것은 실무에서 당연한 운용이고,
    k 전체 평균으로 하나만 고르면 k=1 에서 필요한 강한 규제가 k=4 의 이득을 막는다
    (실제로 첫 실행에서 reg=1.0 이 뽑혀 k>=2 에서 로지스틱이 전혀 늘지 않았다).
    """
    train, val, _ = roles_for_fold(ts)
    mu_tr_global, P_z = _train_side(cls, meta, lab, train)
    packs = {k: [] for k in KS}
    for v in val:
        doms = sorted({(kk[0], kk[1]) for kk in cidx if kk[0] == v})
        by = clips_by_domain_label(cidx, lab, v)
        rng = np.random.default_rng(1000 * v + 7)
        for _ in range(max(2, n_rep // 4)):
            ev, pool = draw_eval_and_pool(by, doms, rng)
            ev_keys = sorted(kk for vv in ev.values() for kk in vv)
            y_c = np.array([lab[cidx[kk][0]] for kk in ev_keys])
            for k in KS:
                sel = pick_k(pool, doms, k, rng)
                packs[k].append((ev_keys, y_c, centred(Z, cidx, None, sel), sel))

    HP = {}
    for k in KS:
        hp = {"lam": LAMBDAS[0], "lr_reg": LR_REGS[1], "l4": L4_GRID[0]}

        def best(field, values, method):
            bv, bb = -1.0, values[0]
            for v in values:
                hp[field] = v
                acc = [score_all(cls, Z, head, cidx, ek, yc, mu, sel, P_z,
                                 mu_tr_global, hp, only=[method])[f"{method}|win"]
                       for ek, yc, mu, sel in packs[k]]
                m = float(np.mean(acc))
                if m > bv:
                    bv, bb = m, v
            hp[field] = bb

        best("lam", LAMBDAS, "b_mix")
        best("lr_reg", LR_REGS[1:], "c_lrreg")
        best("l4", L4_GRID, "d_head")
        HP[k] = dict(hp)
    return HP, P_z, mu_tr_global


METHODS = ["a_none", "b_mix", "c_lr", "c_lrreg", "d_head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prefix", default=CFG.ckpt_prefix)
    ap.add_argument("--n_rep", type=int, default=20)
    ap.add_argument("--out", default=f"results/{CFG.prefix('fewshot')}.npz")
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
        HPk, P_z, mg = choose(cls, Z, meta, lab, cidx, head, ts, args.n_rep)
        HP[(ts, sd)] = HPk

        doms = sorted({(k[0], k[1]) for k in cidx if k[0] == ts})
        by = clips_by_domain_label(cidx, lab, ts)
        rng = np.random.default_rng(1000 * ts + sd)
        for _ in range(args.n_rep):
            ev, pool = draw_eval_and_pool(by, doms, rng)
            ev_keys = sorted(k for v in ev.values() for k in v)
            y_c = np.array([lab[cidx[k][0]] for k in ev_keys])
            for k in KS:
                sel = pick_k(pool, doms, k, rng)
                mu = centred(Z, cidx, None, sel)
                r = score_all(cls, Z, head, cidx, ev_keys, y_c, mu, sel, P_z,
                              mg, HPk[k])
                n_win = sum(len(cidx[kk]) for d in sel for c in sel[d]
                            for kk in sel[d][c])
                R[f"k{k}|nwin"][ts].append(float(n_win))
                for m, v in r.items():
                    R[f"k{k}|{m}"][ts].append(v)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  "
              f"reg(k1..k{KS[-1]}) {[HPk[k]['lr_reg'] for k in KS]}  ||  " + "  ".join(
              f"k{k} {np.mean(R[f'k{k}|a_none|win'][ts]):.3f}->"
              f"{np.mean(R[f'k{k}|c_lrreg|win'][ts]):.3f}" for k in KS),
              flush=True)

    def arr(key):
        return np.array([np.mean(R[key][s]) for s in sorted(R[key])])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    report(arr, R, HP, args)


def report(arr, R, HP, args):
    W = 104
    NAMES = {"a_none": "(a) 라벨 없음 (기준선)",
             "b_mix": "(b) prototype 혼합",
             "c_lr": "(c) 로지스틱 (규제 없음)",
             "c_lrreg": "(c) 로지스틱 (prototype 규제)",
             "d_head": "(d) head 마지막 층"}
    print(f"\n{'='*W}\n선택된 초매개변수 (검증 피험자 2명, window 기준)\n{'='*W}")
    for k in KS:
        for key, vals in (("lam", LAMBDAS), ("lr_reg", LR_REGS[1:])):
            c = [h[k][key] for h in HP.values()]
            print(f"  k={k} {key:<8}" + "  ".join(f"{v}:{c.count(v)}"
                                                  for v in vals))

    # 시청 시간 환산: 감정별 k편 x 3감정 x 클립 평균 길이(창 수 x 4초)
    print(f"\n{'='*W}\nfew-shot 라벨 곡선 — window 정확도 (주 지표)\n"
          f"평가: 세션당 {N_CLS}클립(피험자당 {N_CLS*CFG.n_sessions}클립), "
          f"반복 {args.n_rep}\n{'='*W}")
    hdr = f"  {'방법':<28}" + "".join(f"{'k='+str(k):>16}" for k in KS)
    print(hdr)
    for m in METHODS:
        cells = []
        for k in KS:
            a = arr(f"k{k}|{m}|win")
            if m == "a_none":
                cells.append(f"{a.mean():>16.4f}")
            else:
                b = arr(f"k{k}|a_none|win")
                d = a - b
                p = stats.wilcoxon(a, b).pvalue
                cells.append(f"{a.mean():.4f} {d.mean():+.4f} p{p:.3f}".rjust(16))
        print(f"  {NAMES[m]:<28}" + "".join(cells))

    print(f"\n  {'시청 시간 환산':<28}" + "".join(
        f"{arr(f'k{k}|nwin').mean() * 4 / 60:>13.1f}분" for k in KS))
    print(f"  {'(라벨 붙은 창 수)':<28}" + "".join(
        f"{arr(f'k{k}|nwin').mean():>16.0f}" for k in KS))

    print(f"\n{'='*W}\nclip 정확도 (참고 — 평가 클립이 {N_CLS*CFG.n_sessions}개뿐이라 눈금이 {1/(N_CLS*CFG.n_sessions):.3f})\n{'='*W}")
    print(hdr)
    for m in METHODS:
        cells = []
        for k in KS:
            a = arr(f"k{k}|{m}|clip")
            if m == "a_none":
                cells.append(f"{a.mean():>16.4f}")
            else:
                b = arr(f"k{k}|a_none|clip")
                d = a - b
                p = stats.wilcoxon(a, b).pvalue
                cells.append(f"{a.mean():.4f} {d.mean():+.4f} p{p:.3f}".rjust(16))
        print(f"  {NAMES[m]:<28}" + "".join(cells))

    print(f"\n{'='*W}\n라벨 상한({CEILING:.4f}, 피험자 단위 LOCO 라벨) 대비 회수율 — clip"
          f"\n  ** 상한은 전체 클립으로 평가한 값이고 아래는 떼어낸 "
          f"{N_CLS*CFG.n_sessions}클립으로 평가한 값이다 — 평가 세트가 다르므로\n  회수율은 눈금이지 정확한 비율이 아니다. **\n{'='*W}")
    print(f"  {'방법':<28}" + "".join(f"{'k='+str(k):>12}" for k in KS))
    for m in METHODS[1:]:
        cells = []
        for k in KS:
            b = arr(f"k{k}|a_none|clip").mean()
            a = arr(f"k{k}|{m}|clip").mean()
            cells.append(f"{100*(a-b)/(CEILING-b):>11.1f}%")
        print(f"  {NAMES[m]:<28}" + "".join(cells))

    print(f"\n{'='*W}\n파이프라인 검증: k=1 (감정별 1클립 전체) 과 S11 '클립 전체'\n{'='*W}")
    print(f"  이 분석 k=1 기준선 window {arr('k1|a_none|win').mean():.4f}  "
          f"clip {arr('k1|a_none|clip').mean():.4f}")
    try:
        L = np.load(f"results/{CFG.prefix('labelcalib')}.npz")
        print(f"  S11 클립 전체 기준선   window {L['Tfull__base__proto_win'].mean():.4f}  "
              f"clip {L['Tfull__base__proto_clip'].mean():.4f}")
        print("  두 값은 **같지 않아야 정상**이다 — 캘리브레이션 구성(감정별 1클립 전체)은")
        print(f"  같지만 평가 세트가 다르다: 이 분석은 떼어낸 {N_CLS}클립, "
          f"S11 은 나머지 {CFG.n_eval_clips}클립.")
    except FileNotFoundError:
        pass
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
