"""
What centering does NOT remove, how to choose the variant honestly, and
whether the calibration result survives leaving the evaluated clip out.

Three loose ends from S3.

**The subject probe was rigged.**  Reporting 0.972 -> 0.030 as "centering
removes subject identity" overstates it: centering sets each domain's mean to
exactly zero, so a linear probe reading the mean has nothing left BY
CONSTRUCTION.  The honest question is what survives — second-order structure
inside a clip, or anything a nonlinear boundary can find.  Note in advance
that subtracting a constant cannot change a covariance, so the within-clip
covariance probe must be IDENTICAL before and after centering; only whitening
can move it.  That identity is the point, not a bug.

**The variant was chosen on the test set.**  0.7694 for whitening is the best
of six configurations picked by looking at the number it is being compared to.
Choosing on each fold's two validation subjects instead gives a number that
does not borrow from the test subject.

**The calibration mean saw the clip it was scoring.**  'random' and 'perclip'
draw from the whole session, including the clip being classified, so part of
the gain could be that clip informing its own centre.  Excluding it is the
clean version.

Reads cache_ft/.  No GPU.
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

from analyze_centering import (N_CLS, N_SUBJ, FS, SEG, roles_for_fold,
                               load_head, head_probs, clip_index, l2,
                               clip_reduce, bootstrap_ci)
from analyze_centering_mechanism import (_inv_sqrt_shrunk, domain_transform,
                                         _apply, calib_masks)


# ── 보완 1: what survives centering ─────────────────────────────────────────

def clip_covariance_features(X, meta, k=20, seed=0):
    """Per clip: the upper triangle of the covariance of its windows, after
    projecting to the top-k PCA directions.

    A 200x200 covariance has 20,100 free numbers against 675 clips, so the
    projection is what makes the probe estimable at all.  k=20 gives 210
    features.
    """
    from sklearn.decomposition import PCA
    cidx = clip_index(meta)
    keys = sorted(cidx)
    P = PCA(n_components=k, random_state=seed).fit(X)
    iu = np.triu_indices(k)
    F = []
    for key in keys:
        W = P.transform(X[cidx[key]])
        F.append(np.cov(W, rowvar=False)[iu])
    return np.asarray(F), keys


def probe(F, target, kind="linear", seed=0, n_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    accs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(F, target):
        sc = StandardScaler().fit(F[tr])
        clf = (LogisticRegression(max_iter=1500, random_state=seed) if kind == "linear"
               else SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed))
        clf.fit(sc.transform(F[tr]), target[tr])
        accs.append(float((clf.predict(sc.transform(F[te])) == target[te]).mean()))
    return float(np.mean(accs))


def survival_probes(cls, meta, lab, alpha=0.1, k=20):
    """Subject identity under three treatments and two probe families."""
    doms = sorted({(int(r[0]), int(r[1])) for r in meta})
    idx_all = np.arange(len(meta))
    cidx = clip_index(meta)
    keys = sorted(cidx)
    subj = np.array([k_[0] for k_ in keys])
    y = np.array([lab[cidx[k_][0]] for k_ in keys])

    tf_c = domain_transform(cls, meta, doms, "a")
    tf_w = domain_transform(cls, meta, doms, "c", alpha)
    Xc = _apply(cls, idx_all, meta, tf_c)
    Xw = _apply(cls, idx_all, meta, tf_w)

    out = {}
    for tag, X in (("raw", cls), ("cent", Xc), ("whit", Xw)):
        E = clip_reduce(l2(X), cidx, keys)
        out[f"lin_subj_{tag}"] = probe(E, subj, "linear")
        out[f"rbf_subj_{tag}"] = probe(E, subj, "rbf")
        out[f"lin_emo_{tag}"] = probe(E, y, "linear")
        out[f"rbf_emo_{tag}"] = probe(E, y, "rbf")
        F, _ = clip_covariance_features(X, meta, k)
        out[f"cov_subj_{tag}"] = probe(F, subj, "linear")
        out[f"cov_emo_{tag}"] = probe(F, y, "linear")
    return out


# ── 보완 2: choose the variant on the validation subjects ───────────────────

CONFIGS = [("a", None), ("b", None)] + [("c", a) for a in (0.01, 0.05, 0.1, 0.3, 0.5)]
CONFIG_NAMES = ["중심화만", "중심화+차원별std"] + [f"화이트닝 α={a}" for a in
                                              (0.01, 0.05, 0.1, 0.3, 0.5)]


def _clip_acc(cls, meta, lab, head, target_subjects, train, variant, alpha):
    """Conditions 2 and 4 at clip level for an arbitrary target group."""
    cidx = clip_index(meta)
    doms = sorted({(int(r[0]), int(r[1])) for r in meta})
    tr_doms = [d for d in doms if d[0] in train]
    tg_doms = [d for d in doms if d[0] in target_subjects]
    tg_keys = sorted(k for k in cidx if k[0] in target_subjects)
    tr_keys = sorted(k for k in cidx if k[0] in train)
    tg_w = np.concatenate([cidx[k] for k in tg_keys])
    tr_w = np.concatenate([cidx[k] for k in tr_keys])
    y_c = np.array([lab[cidx[k][0]] for k in tg_keys])
    y_trc = np.array([lab[cidx[k][0]] for k in tr_keys])
    dom_trc = np.array([(k[0], k[1]) for k in tr_keys])
    local = {k: np.searchsorted(tg_w, cidx[k]) for k in tg_keys}

    A = _apply(cls, tg_w, meta, domain_transform(cls, meta, tg_doms, variant, alpha or 0.1))
    B = _apply(cls, tr_w, meta, domain_transform(cls, meta, tr_doms, variant, alpha or 0.1))

    P = head_probs(head, A + B.mean(0))
    c2 = float((np.stack([P[local[k]].mean(0) for k in tg_keys]).argmax(1) == y_c).mean())

    Zt, Zr = l2(A), l2(B)
    Ec = np.stack([Zt[local[k]].mean(0) for k in tg_keys])
    off, Etr = 0, []
    for k in tr_keys:
        n = len(cidx[k]); Etr.append(Zr[off:off + n].mean(0)); off += n
    Etr = np.stack(Etr)
    PR = np.stack([np.nanmean(
        [Etr[(dom_trc == np.asarray(d)).all(1) & (y_trc == c)].mean(0)
         for d in tr_doms], axis=0) for c in range(N_CLS)])
    c4 = float(((l2(Ec) @ l2(PR).T).argmax(1) == y_c).mean())
    return c2, c4


def variant_selection(cls, meta, lab, head, test_subj):
    """Pick the transform on the validation subjects, report it on the test one."""
    train, val, _ = roles_for_fold(test_subj)
    v_c2, v_c4, t_c2, t_c4 = [], [], [], []
    for variant, alpha in CONFIGS:
        a2, a4 = _clip_acc(cls, meta, lab, head, val, train, variant, alpha)
        b2, b4 = _clip_acc(cls, meta, lab, head, [test_subj], train, variant, alpha)
        v_c2.append(a2); v_c4.append(a4); t_c2.append(b2); t_c4.append(b4)
    out = {}
    for path, v, t in (("c2", v_c2, t_c2), ("c4", v_c4, t_c4)):
        j = int(np.argmax(v))
        out[f"sel_{path}_idx"] = j
        out[f"sel_{path}_test"] = t[j]              # honest: chosen on val
        out[f"oracle_{path}_test"] = float(max(t))  # what test-picking gives
        for i, nm in enumerate(CONFIG_NAMES):
            out[f"all_{path}_{i}"] = t[i]
    return out


# ── 보완 3: leave the evaluated clip out of its own centre ──────────────────

def calib_loco(cls, meta, lab, head, test_subj, seconds=(30, 60, 120),
               n_rep=20, hold_sec=300):
    """Calibration with and without the scored clip contributing to the mean."""
    cidx = clip_index(meta)
    doms = sorted({(int(r[0]), int(r[1])) for r in meta})
    te_doms = [d for d in doms if d[0] == test_subj]
    train = roles_for_fold(test_subj)[0]
    tr_doms = [d for d in doms if d[0] in train]
    tr_keys = sorted(k for k in cidx if k[0] in train)

    n_hold = int(round(hold_sec * FS / SEG))
    eval_ok = np.zeros(len(meta), bool)
    for d in te_doms:
        idx = np.flatnonzero((meta[:, 0] == d[0]) & (meta[:, 1] == d[1]))
        eval_ok[idx[n_hold:]] = True
    te_keys = [k for k in sorted(cidx) if k[0] == test_subj
               and eval_ok[cidx[k]].mean() >= 0.5]
    y_c = np.array([lab[cidx[k][0]] for k in te_keys])

    Z = l2(cls)
    B = clip_reduce(Z, cidx, tr_keys)
    y_tr = np.array([lab[cidx[k][0]] for k in tr_keys])
    db = np.array([(k[0], k[1]) for k in tr_keys])
    for d in tr_doms:
        m = (db == np.asarray(d)).all(1); B[m] -= B[m].mean(0)
    PR = np.stack([np.nanmean([B[(db == np.asarray(d)).all(1) & (y_tr == c)].mean(0)
                               for d in tr_doms], axis=0) for c in range(N_CLS)])
    mu_tr_global = np.mean(
        [cls[(meta[:, 0] == d[0]) & (meta[:, 1] == d[1])].mean(0) for d in tr_doms], axis=0)

    def score(keep, loco):
        p2, p4 = [], []
        for k in te_keys:
            w = np.array([i for i in cidx[k] if eval_ok[i]])
            d = (k[0], k[1])
            m = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1]) & keep
            if loco:
                own = np.zeros(len(meta), bool); own[cidx[k]] = True
                m = m & ~own
            if not m.any():
                return float("nan"), float("nan")
            p2.append(head_probs(head, cls[w] - cls[m].mean(0) + mu_tr_global).mean(0))
            p4.append(Z[w].mean(0) - Z[m].mean(0))
        c2 = float((np.stack(p2).argmax(1) == y_c).mean())
        c4 = float(((l2(np.stack(p4)) @ l2(PR).T).argmax(1) == y_c).mean())
        return c2, c4

    out = {}
    rng = np.random.default_rng(0)
    for sec in seconds:
        for kind in ("random", "perclip"):
            reps = n_rep if kind == "random" else 1
            for loco in (False, True):
                a2, a4 = [], []
                for _ in range(reps):
                    keep = calib_masks(meta, te_doms, sec, kind, rng)
                    s2, s4 = score(keep, loco)
                    a2.append(s2); a4.append(s4)
                tag = "loco" if loco else "self"
                out[f"{kind}{sec}_{tag}_c2"] = float(np.nanmean(a2))
                out[f"{kind}{sec}_{tag}_c4"] = float(np.nanmean(a4))
    for loco in (False, True):
        s2, s4 = score(np.ones(len(meta), bool), loco)
        tag = "loco" if loco else "self"
        out[f"full_{tag}_c2"], out[f"full_{tag}_c4"] = s2, s4
    return out


# ── driver ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--prev", default=f"results/{CFG.prefix('centering_analysis')}.npz")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pca_k", type=int, default=20)
    ap.add_argument("--out", default=f"results/{CFG.prefix('centering_limits')}.npz")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "S*_seed*.npz")))
    print(f"[cache] {len(files)}개\n", flush=True)
    R = defaultdict(lambda: defaultdict(list))

    for f in files:
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
        z = np.load(f)
        cls, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
        head = load_head(os.path.join(args.ckpt_dir, f"{CFG.ckpt_prefix}S{ts}_seed{sd}.pt"))

        for k, v in survival_probes(cls, meta, lab, args.alpha, args.pca_k).items():
            R[k][ts].append(v)
        for k, v in variant_selection(cls, meta, lab, head, ts).items():
            R[k][ts].append(v)
        for k, v in calib_loco(cls, meta, lab, head, ts).items():
            R[k][ts].append(v)
        print(f"  S{ts} seed{sd}  lin {R['lin_subj_cent'][ts][-1]:.3f}  "
              f"rbf {R['rbf_subj_cent'][ts][-1]:.3f}  "
              f"cov {R['cov_subj_cent'][ts][-1]:.3f}", flush=True)

    def arr(k):
        return np.array([np.mean(R[k][s]) for s in sorted(R[k])])

    # ── 보완 1 ──────────────────────────────────────────────────────────
    print(f"\n{'='*96}\n보완 1: 중심화 후에도 남는 피험자 정보\n{'='*96}")
    print(f"  {'프로브':<34} {'원본':>9} {'중심화':>9} {'화이트닝':>10} {'우연':>8}")
    rows = [("선형, 클립 임베딩", "lin_subj"),
            ("RBF SVM, 클립 임베딩", "rbf_subj"),
            (f"선형, 클립 내 공분산(PCA {args.pca_k})", "cov_subj")]
    for nm, key in rows:
        print(f"  {nm:<34} {arr(key+'_raw').mean():9.4f} {arr(key+'_cent').mean():9.4f} "
              f"{arr(key+'_whit').mean():10.4f} {1/N_SUBJ:8.4f}")
    print(f"\n  {'감정 프로브':<34} {'원본':>9} {'중심화':>9} {'화이트닝':>10} {'우연':>8}")
    for nm, key in [("선형, 클립 임베딩", "lin_emo"), ("RBF SVM", "rbf_emo"),
                    ("선형, 클립 내 공분산", "cov_emo")]:
        print(f"  {nm:<34} {arr(key+'_raw').mean():9.4f} {arr(key+'_cent').mean():9.4f} "
              f"{arr(key+'_whit').mean():10.4f} {1/N_CLS:8.4f}")
    d = abs(arr("cov_subj_raw") - arr("cov_subj_cent")).max()
    print(f"\n  공분산 프로브의 원본 vs 중심화 최대 차이 {d:.2e}")
    print("  (평균을 빼도 공분산은 변하지 않으므로 정의상 같아야 한다 — 확인용)")

    # ── 보완 2 ──────────────────────────────────────────────────────────
    print(f"\n{'='*96}\n보완 2: 변형을 검증 피험자로 고른 경우 (clip 단위)\n{'='*96}")
    prev = np.load(args.prev) if os.path.exists(args.prev) else None
    for path, pname in (("c2", "head"), ("c4", "prototype")):
        print(f"\n[{pname}]")
        for i, nm in enumerate(CONFIG_NAMES):
            a = arr(f"all_{path}_{i}")
            print(f"  {nm:<24} 테스트 {a.mean():.4f} ± {a.std(ddof=1):.4f}")
        sel = arr(f"sel_{path}_test"); orc = arr(f"oracle_{path}_test")
        print(f"  {'-'*60}")
        print(f"  {'검증으로 선택':<24} 테스트 {sel.mean():.4f} ± {sel.std(ddof=1):.4f}"
              f"   <- 정직한 값")
        print(f"  {'테스트로 선택 (oracle)':<24} 테스트 {orc.mean():.4f} ± {orc.std(ddof=1):.4f}"
              f"   차이 {orc.mean()-sel.mean():+.4f}")
        idx = np.concatenate([R[f"sel_{path}_idx"][s] for s in sorted(R[f"sel_{path}_idx"])])
        cnt = {CONFIG_NAMES[i]: int((idx == i).sum()) for i in range(len(CONFIG_NAMES))}
        print(f"  선택 빈도 ({sum(cnt.values())}회): " + "  ".join(f"{k}={v}" for k, v in cnt.items() if v))
        if prev is not None:
            for ref, rname in (("c1_clip", "조건1(중심화 없음)"),
                               (f"{path}_clip", f"조건{path[1]}(S2, 중심화만)")):
                dd = sel - prev[ref]
                lo, hi = bootstrap_ci(dd)
                print(f"  {rname:<22} 대비  Δ{dd.mean():+.4f}  "
                      f"CI[{lo:+.4f},{hi:+.4f}]{'  0포함' if lo<=0<=hi else '       '}  "
                      f"{int((dd>0).sum())}/{len(dd)}")

    # ── 보완 3 ──────────────────────────────────────────────────────────
    print(f"\n{'='*96}\n보완 3: 평가 클립을 중심화 평균에서 제외\n{'='*96}")
    for path, pname in (("c2", "head"), ("c4", "prototype")):
        print(f"\n[{pname}]   {'':>10} {'자기 포함':>12} {'자기 제외':>12} {'차이':>9}")
        for kind, knm in (("random", "무작위"), ("perclip", "클립마다 조금")):
            for sec in (30, 60, 120):
                a = arr(f"{kind}{sec}_self_{path}").mean()
                b = arr(f"{kind}{sec}_loco_{path}").mean()
                print(f"  {knm:<14}{sec:>4}초 {a:12.4f} {b:12.4f} {b-a:+9.4f}")
        a = arr(f"full_self_{path}").mean(); b = arr(f"full_loco_{path}").mean()
        print(f"  {'세션 전체':<18} {a:12.4f} {b:12.4f} {b-a:+9.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{k: np.array([np.mean(v[s]) for s in sorted(v)])
                          for k, v in R.items()})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
