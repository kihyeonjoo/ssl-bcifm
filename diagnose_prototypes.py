"""
Stage 1: are class directions the same across subjects, once EA has run?

The prototype-alignment idea assumes that after Euclidean Alignment a class
occupies roughly the same direction in the foundation model's feature space
for every subject, and that LOSO fails only because that direction is not
*exactly* shared.  This script measures the assumption instead of assuming it.

Design decisions that matter
----------------------------
**The clip is the unit.**  SEED's label is constant for a whole ~4 min film
clip, so the ~56 four-second windows inside one clip are near-duplicates.
Counting them as independent samples overstates n by ~50x and makes every
interval far too tight.  Every number here is computed from the 15 clip
embeddings per domain, never from windows.

**Domain = (subject, session), not subject.**  EA fits one whitening transform
per recording session (``ea_scope: session``), so a subject's three sessions
have passed through three different transforms and are not interchangeable.

**Domain centering is label-free.**  Subtracting each domain's own mean clip
embedding removes a per-recording offset without using any label, so it is
something a deployed system could do from unlabelled calibration data.  This
is what makes metric 1 an honest cross-subject number rather than a fit.

**Metric 2 is the ceiling, not a result.**  Classifying a clip against
prototypes built from the *same* domain's other clips says how separable the
classes are when subject transfer is taken out of the problem.  The gap
between 2 and 1 is the part prototype alignment could in principle recover.

Feature choice
--------------
The fine-tuned model's classifier reads the CLS token, so that is the vector a
prototype loss would act on.  The pretrained model never trained CLS as a
summary — LaBraM's pretext task is masked patch prediction — so for the
pretrained path CLS may carry little, and the patch-token mean is reported
alongside for both paths.

Usage
-----
    # (a) pretrained backbone, no checkpoint needed — features cached once
    python diagnose_prototypes.py --mode pretrained --device cpu

    # (b) fine-tuned, whichever A-arm checkpoints exist so far
    python diagnose_prototypes.py --mode finetuned \\
        --ckpt_glob 'checkpoints_s0/a02_S*_seed*.pt' --device cpu

    # after the A arm finishes, add the accuracy correlation
    python diagnose_prototypes.py --mode finetuned \\
        --ckpt_glob 'checkpoints_s0/a02_S*_seed*.pt' \\
        --arm_csv results/s0_diag_a02.csv --device cuda --pca_folds 1,8
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
from scipy import stats

from data.seed_raw_dataset import SEEDRawDataset, SEED_CH_NAMES

from dataset_config import get as _cfg_root
CFG = _cfg_root()
ROOT = _cfg_root().root
from dataset_config import get as _cfg

N_SUBJ = _cfg().n_subjects
N_CLS = _cfg().n_classes
N_VAL = _cfg().n_val_subjects      # matches run_cv's n_val_subjects

# sessions where one channel carries >50% of the session power (EA_ANALYSIS.md 1)
BAD_ELECTRODE = {
    (1, 1): "PO6", (1, 2): "P1", (2, 2): "F7", (5, 1): "P7", (5, 3): "F7",
    (6, 1): "P1", (7, 1): "F7", (7, 2): "F7", (7, 3): "F7", (10, 2): "P1",
}


# ── roles ───────────────────────────────────────────────────────────────────

def roles_for_fold(test_subj, all_subjects=None):
    """Reproduce run_cv's split exactly: test = one subject, val = the next
    two in cyclic order, train = the remaining twelve."""
    subs = all_subjects or list(range(1, N_SUBJ + 1))
    i = subs.index(test_subj)
    val = [subs[(i + k) % len(subs)] for k in range(1, N_VAL + 1)]
    train = [s for s in subs if s != test_subj and s not in val]
    return train, val, [test_subj]


# ── feature extraction ──────────────────────────────────────────────────────

def build_model(labram_repo, ckpt_path, device, state=None, ch_names=None):
    """Pretrained LaBraM, optionally overwritten by a fine-tuned state dict.

    ``ch_names``: the dataset's 10-20 channel names (None = 62-ch SEED layout)."""
    sys.path.insert(0, os.path.abspath(os.path.expanduser(labram_repo)))
    from finetune_labram_hemi_aux import LaBraMHemiAuxClassifier
    m = LaBraMHemiAuxClassifier(
        labram_repo=labram_repo, ckpt_path=ckpt_path,
        n_classes=N_CLS, dropout=0.0, lambda_asym=0.0, use_lora=False,
        ch_names=ch_names,
    )
    if state is not None:
        missing, unexpected = m.load_state_dict(state, strict=False)
        if len(missing) > 20:
            raise ValueError(
                f"checkpoint does not fit the model: {len(missing)} missing keys "
                f"(e.g. {missing[:3]}). A LoRA checkpoint cannot be loaded here."
            )
    return m.to(device).eval()


@torch.no_grad()
def extract(model, ds, device, batch_size=32, amp=False):
    """-> (cls, patch_mean) both L2-normalised, plus (subject, session, clip, label)."""
    from torch.utils.data import DataLoader
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=(device.type == "cuda"))
    CLS, PAT, meta, lab = [], [], [], []
    for b in ld:
        eeg = b["eeg"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp and device.type == "cuda"):
            allt = model.labram.forward_features(
                eeg, input_chans=model.input_chans, return_all_tokens=True)
        allt = allt.float()
        CLS.append(torch.nn.functional.normalize(allt[:, 0], dim=-1).cpu())
        PAT.append(torch.nn.functional.normalize(allt[:, 1:].mean(1), dim=-1).cpu())
        meta.append(torch.stack([b["subject"], b["session"], b["clip"]], 1))
        lab.append(b["label"])
    return (torch.cat(CLS).numpy(), torch.cat(PAT).numpy(),
            torch.cat(meta).numpy(), torch.cat(lab).numpy())


def select_windows(meta, n_per_clip):
    """Indices of ``n_per_clip`` evenly spaced windows inside each film clip.

    Evenly spaced, not a prefix: a clip's emotional response is not uniform
    over its four minutes, and the first N windows would sample only its
    opening.  Returns every index when ``n_per_clip`` is None or already
    covers the clip.

    The SAME helper is used when restricting the forward passes and when
    aggregating from a full cache, so "12 windows" means the identical twelve
    windows in both paths and the two are directly comparable.
    """
    if not n_per_clip:
        return np.arange(len(meta))
    seen = {}
    for i, r in enumerate(meta):
        seen.setdefault(tuple(r), []).append(i)
    out = []
    for k in sorted(seen):
        idx = seen[k]                      # already in time order
        if len(idx) <= n_per_clip:
            out.extend(idx)
        else:
            take = np.linspace(0, len(idx) - 1, n_per_clip).astype(int)
            out.extend(idx[t] for t in take)
    return np.asarray(sorted(out))


def clip_embeddings(F, meta, lab, windows_per_clip=None):
    """Average the window features inside each film clip.

    ``windows_per_clip`` keeps only that many evenly spaced windows per clip
    before averaging — the cheap variant used when a full extraction would
    cost too many forward passes.

    -> E (n_clips, D), dom (n_clips, 2) = (subject, session), y (n_clips,)
    """
    keep = set(select_windows(meta, windows_per_clip).tolist())
    seen = {}
    for i, r in enumerate(meta):
        if i in keep:
            seen.setdefault(tuple(r), []).append(i)
    E, dom, y = [], [], []
    for k in sorted(seen):
        idx = seen[k]
        E.append(F[idx].mean(0))
        dom.append((k[0], k[1]))
        y.append(int(lab[idx[0]]))
    return np.asarray(E), np.asarray(dom), np.asarray(y)


# ── prototype metrics ───────────────────────────────────────────────────────

def center_by_domain(E, dom):
    """Subtract each domain's own mean clip embedding.  No labels involved."""
    C = E.copy()
    for d in np.unique(dom, axis=0):
        m = (dom == d).all(1)
        C[m] -= C[m].mean(0)
    return C


def _cos(A, B):
    A = A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=-1, keepdims=True), 1e-12)
    return A @ B.T


def domain_prototypes(C, y, mask):
    """Class prototypes of one domain -> (n_cls, D); NaN row if a class is absent."""
    P = np.full((N_CLS, C.shape[1]), np.nan)
    for k in range(N_CLS):
        m = mask & (y == k)
        if m.any():
            P[k] = C[m].mean(0)
    return P


def metrics_for_target(C, dom, y, target_domains, source_subjects):
    """Metrics 1-4 for one held-out subject against a pool of source subjects."""
    src = np.isin(dom[:, 0], source_subjects)
    src_doms = np.unique(dom[src], axis=0)
    Ps = np.stack([domain_prototypes(C, y, (dom == d).all(1)) for d in src_doms])
    Pbar = np.nanmean(Ps, axis=0)                       # (n_cls, D)

    tgt = np.zeros(len(dom), bool)
    for d in target_domains:
        tgt |= (dom == np.asarray(d)).all(1)

    # metric 1 — cross-subject: test clips vs averaged source prototypes
    m1 = float((_cos(C[tgt], Pbar).argmax(1) == y[tgt]).mean())

    # metric 2 — ceiling: leave-one-clip-out inside the test domain itself
    hit, n = 0, 0
    for d in target_domains:
        m = (dom == np.asarray(d)).all(1)
        idx = np.flatnonzero(m)
        for i in idx:
            P = np.full((N_CLS, C.shape[1]), np.nan)
            for k in range(N_CLS):
                o = idx[(y[idx] == k) & (idx != i)]
                if len(o):
                    P[k] = C[o].mean(0)
            ok = ~np.isnan(P).any(1)
            if ok.sum() < 2:
                continue
            pred = np.flatnonzero(ok)[_cos(C[i:i + 1], P[ok]).argmax(1)[0]]
            hit += int(pred == y[i]); n += 1
    m2 = hit / max(n, 1)

    # metric 4 — per-class cosine between the test domain's prototype and Pbar
    Pt = np.stack([domain_prototypes(C, y, (dom == np.asarray(d)).all(1))
                   for d in target_domains])
    Pt = np.nanmean(Pt, axis=0)
    m4 = [float(_cos(Pt[k:k + 1], Pbar[k:k + 1])[0, 0]) for k in range(N_CLS)]

    return {"m1_cross": m1, "m2_within": m2, "m3_gap": m2 - m1,
            "m4_cos": m4, "m4_cos_mean": float(np.mean(m4))}


def run_fold(C, dom, y, test_subj, pool=None):
    """Metrics for one LOSO fold, plus metric 5's train-vs-train reference.

    ``pool`` is the set of subjects actually loaded.  On a full run it is all
    fifteen and the split matches run_cv exactly.  On a subset run (validation
    only) the roles are recomputed inside the subset, so the numbers are NOT
    comparable to a real fold — the source pool is smaller.
    """
    loaded = sorted(np.unique(dom[:, 0]).tolist())
    pool = pool or loaded
    train, val, _ = roles_for_fold(test_subj, pool)
    train = [s for s in train if s in loaded]
    if not train:
        raise ValueError(
            f"fold S{test_subj}: no training subject is loaded "
            f"(loaded={loaded}). Pass --subjects covering the training pool.")
    tgt = [tuple(d) for d in np.unique(dom[dom[:, 0] == test_subj], axis=0)]
    if not tgt:
        raise ValueError(f"fold S{test_subj}: that subject is not loaded")
    out = metrics_for_target(C, dom, y, tgt, train)

    # metric 5 — hold out one TRAINING subject and score it against the rest.
    # This is the same measurement on domains the model did see, so it
    # separates "subjects differ" from "this subject was not trained on".
    ref = []
    for s in train:
        others = [t for t in train if t != s]
        td = [tuple(d) for d in np.unique(dom[dom[:, 0] == s], axis=0)]
        if not others or not td:
            continue
        ref.append(metrics_for_target(C, dom, y, td, others))
    out["m5_train_m1"] = float(np.mean([r["m1_cross"] for r in ref])) if ref else float("nan")
    out["m5_train_gap"] = float(np.mean([r["m3_gap"] for r in ref])) if ref else float("nan")
    out["m5_train_cos"] = float(np.mean([r["m4_cos_mean"] for r in ref])) if ref else float("nan")
    out["test_subj"] = test_subj
    return out


def report(rows, feat_name, tag, arm_acc=None):
    """Per-subject table plus the interpretation the numbers support."""
    by = {}
    for r in rows:
        by.setdefault(r["test_subj"], []).append(r)
    subs = sorted(by)
    print(f"\n{'='*94}")
    print(f"{tag}  —  특징: {feat_name}")
    print(f"{'='*94}")
    print(f"{'subj':>4} {'교차(1)':>16} {'자기도메인(2)':>16} {'격차(3)':>16} "
          f"{'클래스cos(4)':>14} {'불량전극':>9}")

    def ms(v):
        v = np.asarray(v, float)
        return (f"{v.mean():.4f}" if len(v) == 1
                else f"{v.mean():.4f}±{v.std(ddof=1):.4f}")

    gaps, accs = [], []
    for s in subs:
        rs = by[s]
        bad = sum(1 for sess in (1, 2, 3) if (s, sess) in BAD_ELECTRODE)
        g = float(np.mean([r["m3_gap"] for r in rs]))
        gaps.append(g)
        print(f"{s:>4} {ms([r['m1_cross'] for r in rs]):>16} "
              f"{ms([r['m2_within'] for r in rs]):>16} "
              f"{ms([r['m3_gap'] for r in rs]):>16} "
              f"{ms([r['m4_cos_mean'] for r in rs]):>14} "
              f"{('★'*bad if bad else '-'):>9}")

    m1 = np.array([np.mean([r["m1_cross"] for r in by[s]]) for s in subs])
    m2 = np.array([np.mean([r["m2_within"] for r in by[s]]) for s in subs])
    m5 = np.array([np.mean([r["m5_train_m1"] for r in by[s]]) for s in subs])
    m5g = np.array([np.mean([r["m5_train_gap"] for r in by[s]]) for s in subs])
    cos = np.array([np.mean([r["m4_cos_mean"] for r in by[s]]) for s in subs])
    print(f"\n  (1) 교차 피험자          {m1.mean():.4f} ± {m1.std(ddof=1):.4f}")
    print(f"  (2) 자기 도메인 상한     {m2.mean():.4f} ± {m2.std(ddof=1):.4f}")
    print(f"  (3) 격차                 {(m2-m1).mean():+.4f}")
    print(f"  (4) 클래스 코사인 평균    {cos.mean():+.4f}")
    print(f"  (5) 학습 도메인끼리 교차  {m5.mean():.4f}  격차 {m5g.mean():+.4f}"
          f"   <- 학습에 쓴 피험자도 이 정도면, 문제는 '미학습'이 아니라 '피험자 차이'")
    print(f"      우연 수준 {1/N_CLS:.4f},  클립 단위이므로 도메인당 15개 x "
          f"{len(subs)}명 = 표본 {15*3*len(subs)}")

    # bad electrodes vs gap
    g = np.array(gaps)
    bad_mask = np.array([any((s, ss) in BAD_ELECTRODE for ss in (1, 2, 3)) for s in subs])
    if bad_mask.any() and (~bad_mask).any():
        print(f"\n  불량 전극 보유 피험자 {int(bad_mask.sum())}명 격차 "
              f"{g[bad_mask].mean():+.4f}   나머지 {g[~bad_mask].mean():+.4f}")
        u = stats.mannwhitneyu(g[bad_mask], g[~bad_mask])
        print(f"  Mann-Whitney p={u.pvalue:.4f}  "
              f"(격차가 큰 쪽에 몰리는가)")

    # metric 6
    if arm_acc:
        common = [s for s in subs if s in arm_acc]
        if len(common) >= 5:
            x = np.array([np.mean([r["m3_gap"] for r in by[s]]) for s in common])
            a = np.array([arm_acc[s] for s in common])
            rho, pv = stats.spearmanr(x, a)
            print(f"\n  (6) 격차 vs A팔 정확도  Spearman rho={rho:+.3f}  p={pv:.4f}  "
                  f"(n={len(common)})")
            print("      음수면 '격차가 큰 피험자일수록 LOSO 정확도가 낮다' = "
                  "격차가 실패를 설명한다")
        else:
            print(f"\n  (6) A팔 결과가 {len(common)}명뿐 — 상관은 A팔 완료 후")
    return {"m1": m1.mean(), "m2": m2.mean(), "gap": (m2 - m1).mean()}


def pca_plot(C, dom, y, folds, path, feat_name):
    """2D PCA of the centered class prototypes.  colour = class, marker = role."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Korean titles render as boxes without a CJK font on the Agg backend.
    matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, len(folds), figsize=(5.2 * len(folds), 4.6),
                             squeeze=False)
    colours = ["#d62728", "#7f7f7f", "#2ca02c"]      # neg / neu / pos
    names = ["negative", "neutral", "positive"]
    for ax, ts in zip(axes[0], folds):
        train, val, _ = roles_for_fold(ts)
        P, lbl, role = [], [], []
        for d in np.unique(dom, axis=0):
            m = (dom == d).all(1)
            pr = domain_prototypes(C, y, m)
            for k in range(N_CLS):
                if np.isnan(pr[k]).any():
                    continue
                P.append(pr[k]); lbl.append(k)
                role.append("test" if d[0] == ts else
                            ("val" if d[0] in val else "train"))
        P = np.asarray(P); lbl = np.asarray(lbl); role = np.asarray(role)
        Z = PCA(n_components=2).fit_transform(P)
        for r, mk, sz, al in (("train", "o", 26, .45), ("val", "^", 46, .8),
                              ("test", "*", 230, 1.0)):
            for k in range(N_CLS):
                m = (role == r) & (lbl == k)
                if m.any():
                    ax.scatter(Z[m, 0], Z[m, 1], c=colours[k], marker=mk,
                               s=sz, alpha=al, linewidths=0,
                               label=f"{names[k]} / {r}")
        ax.set_title(f"fold: test = S{ts}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.axhline(0, lw=.4, c="k", alpha=.3); ax.axvline(0, lw=.4, c="k", alpha=.3)
    axes[0][-1].legend(fontsize=6, loc="best", ncol=2)
    fig.suptitle(f"도메인 중심화 후 클래스 prototype — {feat_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[pca] {path}")


def load_arm_csv(path):
    """Per-subject A-arm accuracy averaged over seeds."""
    import csv
    from collections import defaultdict
    acc = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            acc[int(float(r["subject"]))].append(float(r["accuracy"]))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrained", "finetuned"], required=True)
    ap.add_argument("--ckpt_glob", default=f"checkpoints_s0/{CFG.ckpt_prefix}S*_seed*.pt")
    ap.add_argument("--labram_repo", default="/home/kihyeonjoo/LaBraM")
    ap.add_argument("--labram_ckpt",
                    default="/home/kihyeonjoo/LaBraM/checkpoints/labram-base.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--windows_per_clip", type=int, default=None,
                    help="use only this many evenly spaced windows per clip. "
                         "Cuts the forward passes by ~56/N; the pretrained "
                         "comparison says how much the metrics move.")
    ap.add_argument("--folds", default=None, help="which test subjects to score")
    ap.add_argument("--cache", default="cache_diag/pretrained_feats.npz")
    ap.add_argument("--arm_csv", default=None)
    ap.add_argument("--pca_folds", default=None)
    ap.add_argument("--pca_out", default="figs/prototypes_pca.png")
    ap.add_argument("--partial", default=None,
                    help="JSONL file of per-checkpoint results; written as they "
                         "finish and re-read on restart so a died run resumes.")
    args = ap.parse_args()

    device = torch.device(args.device)
    subjects = (list(range(1, N_SUBJ + 1)) if args.subjects == "all"
                else [int(x) for x in args.subjects.split(",")])
    if len(subjects) < N_SUBJ:
        print(f"[!] 부분집합 실행 ({len(subjects)}명): 소스 풀이 작아 수치가 "
              f"실제 fold 와 다르다. 동작 확인용으로만 쓸 것.", flush=True)
    folds = ([int(x) for x in args.folds.split(",")] if args.folds else subjects)
    arm_acc = load_arm_csv(args.arm_csv) if args.arm_csv else None

    ds_kwargs = dict(root=ROOT, subjects=subjects, sessions=[1, 2, 3],
                     segment_length=800, step=800, patch_size=200,
                     norm="scale100", ea=True, ea_mode="diag",
                     ea_scope="session", ea_scale=0.2)

    # Build the dataset ONCE.  It holds every segment in memory (~7.5 GB for
    # all fifteen subjects), so rebuilding it per checkpoint both re-reads the
    # .mat files 45 times and repeatedly allocates that much — the first run
    # of this script died silently at checkpoint 16, which is what that looks
    # like from outside.
    _ds_cache = {}

    def _dataset(windows_per_clip=None):
        if "ds" not in _ds_cache:
            from torch.utils.data import Subset
            full = SEEDRawDataset(**ds_kwargs)
            if windows_per_clip:
                keep = select_windows(np.asarray(full._seg_meta), windows_per_clip)
                print(f"  창 {len(keep)}/{len(full)} 개만 forward "
                      f"({len(full)/max(len(keep),1):.1f}배 절약)", flush=True)
                _ds_cache["ds"] = Subset(full, keep.tolist())
            else:
                _ds_cache["ds"] = full
        return _ds_cache["ds"]

    def feats_from(state, windows_per_clip=None):
        """Extract features from the shared dataset with one model."""
        ds = _dataset(windows_per_clip)
        model = build_model(args.labram_repo, args.labram_ckpt, device, state)
        cls, pat, meta, lab = extract(model, ds, device, args.batch_size, args.amp)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return cls, pat, meta, lab

    results = {}
    if args.mode == "pretrained":
        # The pretrained backbone does not depend on the fold, so its features
        # are extracted once; only the train/val/test role assignment changes.
        if os.path.exists(args.cache):
            z = np.load(args.cache)
            cls, pat, meta, lab = z["cls"], z["pat"], z["meta"], z["lab"]
            if sorted(np.unique(meta[:, 0]).tolist()) != subjects:
                raise ValueError(
                    f"{args.cache} holds subjects {sorted(np.unique(meta[:,0]).tolist())} "
                    f"but {subjects} were asked for; delete the cache or pass --cache")
            print(f"[cache] {args.cache}  {cls.shape}")
        else:
            cls, pat, meta, lab = feats_from(None)     # always cache full
            os.makedirs(os.path.dirname(os.path.abspath(args.cache)) or ".",
                        exist_ok=True)
            np.savez(args.cache, cls=cls, pat=pat, meta=meta, lab=lab)
            print(f"[cache] wrote {args.cache}  {cls.shape}")

        for name, slug, F in (("CLS 토큰", "cls", cls),
                              ("패치 토큰 평균", "patch", pat)):
            E, dom, y = clip_embeddings(F, meta, lab, args.windows_per_clip)
            C = center_by_domain(E, dom)
            rows = [run_fold(C, dom, y, t, subjects) for t in folds]
            results[name] = report(rows, name, "사전학습 LaBraM (fine-tuning 없음)",
                                   arm_acc)
            if args.pca_folds:
                pca_plot(C, dom, y, [int(x) for x in args.pca_folds.split(",")],
                         args.pca_out.replace(".png", f"_pre_{slug}.png"), name)
    else:
        paths = sorted(glob.glob(args.ckpt_glob))
        if not paths:
            raise SystemExit(f"no checkpoints match {args.ckpt_glob}")
        print(f"[ckpt] {len(paths)}개")
        rows = {"CLS 토큰": [], "패치 토큰 평균": []}
        # Per-checkpoint results are written as they finish, so a run that dies
        # part-way can be restarted without redoing what it already scored.
        import json
        partial = args.partial or None
        if partial and os.path.exists(partial):
            with open(partial) as f:
                for line in f:
                    r = json.loads(line)
                    rows[r.pop("_feat")].append(r)
            seen = {(r["test_subj"], r["seed"]) for r in rows["CLS 토큰"]}
            print(f"[resume] {partial} 에서 {len(seen)}개 복원")
        else:
            seen = set()
        for pth in paths:
            m = re.search(r"S(\d+)_seed(\d+)", os.path.basename(pth))
            if not m:
                print(f"  건너뜀 (이름에서 fold/seed 를 못 읽음): {pth}")
                continue
            ts, sd = int(m.group(1)), int(m.group(2))
            if ts not in folds or (ts, sd) in seen:
                continue
            ck = torch.load(pth, map_location="cpu", weights_only=False)
            state = ck["model"] if "model" in ck else ck
            print(f"  S{ts} seed{sd}  (epoch {ck.get('epoch','?')})", flush=True)
            cls, pat, meta, lab = feats_from(state, args.windows_per_clip)
            for name, slug, F in (("CLS 토큰", "cls", cls),
                                  ("패치 토큰 평균", "patch", pat)):
                # already restricted at extraction time, so no second cut
                E, dom, y = clip_embeddings(F, meta, lab)
                C = center_by_domain(E, dom)
                r = run_fold(C, dom, y, ts, subjects)
                r["seed"] = sd
                rows[name].append(r)
                if partial:
                    with open(partial, "a") as f:
                        f.write(json.dumps({**r, "_feat": name}) + "\n")
                if args.pca_folds and ts in [int(x) for x in args.pca_folds.split(",")] and sd == 0:
                    pca_plot(C, dom, y, [ts],
                             args.pca_out.replace(".png", f"_ft_S{ts}_{slug}.png"),
                             name)
        for name in rows:
            if rows[name]:
                results[name] = report(rows[name], name,
                                       "Fine-tuned (A팔 체크포인트)", arm_acc)

    print(f"\n{'='*94}\n해석\n{'='*94}")
    for name, r in results.items():
        print(f"  [{name}] 교차 {r['m1']:.4f}  상한 {r['m2']:.4f}  "
              f"격차 {r['gap']:+.4f}")
    print("""
  읽는 법
    - 교차(1)가 우연(0.333)에 가까우면, EA 후에도 클래스 방향이 피험자 간에
      공유되지 않는다는 뜻이다. prototype 정렬 loss 의 전제가 성립하지 않는다.
    - 상한(2)이 높은데 격차(3)가 크면, 클래스는 분리 가능하되 방향이 피험자마다
      다른 것이다. 이 경우가 prototype 정렬이 노릴 수 있는 유일한 경우다.
    - 상한(2)도 낮으면 문제는 정렬이 아니라 특징 자체다.
    - (5) 학습 도메인끼리의 교차 정확도가 (1)과 비슷하면, 원인은 '학습에 없던
      피험자' 가 아니라 '피험자 차이' 자체다. 학습으로 메울 수 없다.""")


if __name__ == "__main__":
    main()
