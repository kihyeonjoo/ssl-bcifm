"""
Does  z_L - z_R  actually carry less subject identity than  z  does?

That is the thesis behind the hemisphere-asymmetry design: the difference of
the two hemispheres should cancel whatever is idiosyncratic to a person
(skull, impedance, electrode placement) and keep what is about the emotion.

This script measures it directly, without fine-tuning anything.  A frozen
pretrained LaBraM sees all 62 channels in ONE forward pass; its per-channel
tokens are pooled into z_L and z_R.  Two linear probes are then fitted on each
candidate representation:

    subject probe   15-way "who is this?"       ← want LOW for the difference
    emotion probe    3-way valence, LOSO         ← want HIGH

A representation that is subject-invariant AND emotion-bearing is the one the
design needs.  The interesting comparison is  z_L - z_R  against  z_all  and
against  z_L  alone, all 200-d, so dimensionality is not a confound.

Reading the result
------------------
  subject acc drops a lot, emotion acc holds   → thesis supported; the loss is
                                                 in how the architecture uses
                                                 the difference, not in the
                                                 difference itself
  subject acc barely drops                     → subtraction does not remove
                                                 subject identity; no fusion
                                                 head will fix that
  subject acc drops, emotion acc drops too     → the difference throws away the
                                                 signal along with the nuisance

Note the subject probe is trained on clips 1-10 and tested on clips 11-15, so a
segment never shares a film clip with its own training data.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data.seed_raw_dataset import SEEDRawDataset, SEED_CH_NAMES
from data.seediv_raw_dataset import SEEDIVRawDataset
from data.seedv_raw_dataset import SEEDVRawDataset
from data.preprocessing import LEFT_IDX, RIGHT_IDX
from finetune_hemi_labram import _register_labram, _get_input_chans


DATASETS = {
    # name:      (class,             root,                           n_clips, n_classes)
    "seed":      (SEEDRawDataset,   "/mnt/data/original/SEED",       15, 3),
    "seed-iv":   (SEEDIVRawDataset, "/mnt/data/original/SEED-IV",    24, 4),
    "seed-v":    (SEEDVRawDataset,  "/mnt/data/original/SEED-V",     15, 5),
}


# ── feature extraction ──────────────────────────────────────────────────────

def build_labram(repo: str, ckpt: str, device: torch.device,
                 finetuned: str | None = None):
    """Frozen LaBraM, no classification head.

    ``finetuned`` overrides the pretrained weights with a backbone saved by
    ``finetune_labram_hemi_aux.py``.  That is how the adversary's claim gets
    audited: the discriminator's own accuracy is circular (the backbone was
    trained to beat exactly that head), so a *fresh* linear probe on the frozen
    representation is the honest test of whether subject identity is gone.
    """
    _register_labram(repo)
    import modeling_finetune  # noqa: F401
    from timm.models import create_model

    model = create_model(
        "labram_base_patch200_200",
        pretrained=False, num_classes=0,
        drop_rate=0.0, drop_path_rate=0.0,
        use_mean_pooling=False, init_scale=0.001,
        use_rel_pos_bias=True, use_abs_pos_emb=True, init_values=0.1,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    if any(k.startswith("student.") for k in sd):
        sd = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}
    drop = ("head.", "lm_head.", "projection_head.", "mask_token", "logit_scale")
    sd = {k: v for k, v in sd.items() if not any(k.startswith(p) for p in drop)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LaBraM] loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    if finetuned:
        ck = torch.load(finetuned, map_location="cpu", weights_only=False)
        ft = ck["labram"] if isinstance(ck, dict) and "labram" in ck else ck
        # A LoRA run saves wrapper keys the plain model does not have; keep the
        # intersection and say how much was actually replaced.
        own = model.state_dict()
        use = {k: v for k, v in ft.items() if k in own and own[k].shape == v.shape}
        # A LoRA checkpoint stores the frozen layer under `<name>.base.weight`
        # plus `lora_A`/`lora_B`, none of which match this plain model's keys.
        # Every adapted layer would be dropped and the probe would silently
        # measure the PRETRAINED weights while printing a fine-tuned path.
        if any(".lora_A" in k or ".base." in k for k in ft):
            raise ValueError(
                f"{finetuned} is a LoRA checkpoint; its adapted layers cannot be "
                "loaded into a plain LaBraM and would be silently dropped. "
                "Probe a full fine-tuning checkpoint, or merge the LoRA deltas first.")
        if len(use) < 0.9 * len(own):
            raise ValueError(
                f"{finetuned}: only {len(use)}/{len(own)} tensors matched this "
                "model — refusing to report a probe of mostly-pretrained weights.")
        model.load_state_dict(use, strict=False)
        meta = ck if isinstance(ck, dict) else {}
        print(f"[LaBraM] fine-tuned backbone: {finetuned}  "
              f"({len(use)}/{len(own)} tensors replaced, "
              f"lambda_adv={meta.get('lambda_adv', '?')}, "
              f"effective={meta.get('effective_lambda', '?')}, "
              f"epoch={meta.get('epoch', '?')})")
    return model.to(device).eval()


@torch.no_grad()
def extract(model, ds, device, batch_size=64, amp=True):
    """Per-segment (z_all, z_L, z_R) from one 62-channel forward pass.

    LaBraM's patch tokens come back channel-major — ``B N A T -> B (N A) T`` in
    its TemporalConv — so reshaping to (B, C, P, D) recovers the channel axis
    and the hemispheres can be pooled without a second pass.
    """
    input_chans = torch.tensor(_get_input_chans(SEED_CH_NAMES), device=device)
    l_idx = torch.tensor(LEFT_IDX, device=device)
    r_idx = torch.tensor(RIGHT_IDX, device=device)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    Z, ZL, ZR, Y, CLS = [], [], [], [], []
    t0 = time.time()
    for i, batch in enumerate(loader):
        eeg = batch["eeg"].to(device, non_blocking=True)      # (B, 62, P, 200)
        ctx = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
               if amp and device.type == "cuda" else torch.autocast("cpu", enabled=False))
        with ctx:
            allt = model.forward_features(
                eeg, input_chans=input_chans,
                return_all_tokens=True,
            )                                                  # (B, 1 + C*P, D)
        allt = allt.float()
        # The adversary attacks the CLS token, so an audit of it must measure
        # the CLS token — pooling patch tokens probes a different vector.
        CLS.append(allt[:, 0].cpu())
        tok = allt[:, 1:]
        B, CP, D = tok.shape
        C = eeg.shape[1]
        tok = tok.view(B, C, CP // C, D).mean(2)               # (B, C, D) pool patches
        Z.append(tok.mean(1).cpu())                            # all channels
        ZL.append(tok.index_select(1, l_idx).mean(1).cpu())
        ZR.append(tok.index_select(1, r_idx).mean(1).cpu())
        Y.append(batch["label"])
        if i % 50 == 0:
            print(f"  batch {i}/{len(loader)}  ({time.time()-t0:.0f}s)", flush=True)
    return (torch.cat(Z).numpy(), torch.cat(ZL).numpy(),
            torch.cat(ZR).numpy(), torch.cat(Y).numpy(),
            torch.cat(CLS).numpy())


# ── probes ──────────────────────────────────────────────────────────────────

def _fit(Xtr, ytr, Xte, yte, seed=0):
    sc = StandardScaler().fit(Xtr)
    # sklearn >=1.7 dropped `multi_class`; multinomial is the default for lbfgs.
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xte)) == yte).mean())


def subject_probe(X, subj, clip, n_train_clips=10):
    """N-way subject identification, split by film clip to avoid clip leakage."""
    tr, te = clip <= n_train_clips, clip > n_train_clips
    return _fit(X[tr], subj[tr], X[te], subj[te])


def emotion_probe(X, y, subj, subjects):
    """3-way valence, leave-one-subject-out, mean over folds."""
    accs = []
    for s in subjects:
        te = subj == s
        accs.append(_fit(X[~te], y[~te], X[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/seed_labram_loso.yaml")
    ap.add_argument("--dataset", default="seed", choices=sorted(DATASETS),
                    help="which SEED-family dataset to probe")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth segment (speed)")
    ap.add_argument("--ea", action="store_true", help="apply Euclidean Alignment first")
    ap.add_argument("--cache", default="probe_features.npz")
    ap.add_argument("--backbone", default=None,
                    help="fine-tuned backbone checkpoint to probe instead of "
                         "the pretrained weights")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dc, lc = cfg["data"], cfg["labram"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.cache):
        print(f"[cache] {args.cache}")
        d = np.load(args.cache)
        Z, ZL, ZR, y = d["Z"], d["ZL"], d["ZR"], d["y"]
        subj, clip = d["subj"], d["clip"]
        CLS = d["CLS"] if "CLS" in d else None
    else:
        cls, root, n_clips, n_classes = DATASETS[args.dataset]
        ds = cls(
            root=root, subjects=None, sessions=None,
            segment_length=dc["segment_length"], step=dc["step"],
            patch_size=dc["patch_size"], norm=dc.get("norm", "scale100"),
            ea=args.ea, ea_scope=dc.get("ea_scope", "session"),
            ea_scale=dc.get("ea_scale", 0.2),
        )
        subj = np.asarray(ds.subjects_of)      # property, not a call
        clip = np.asarray(ds.clips_of)
        if args.stride > 1:
            keep = np.arange(0, len(ds), args.stride)
            ds = torch.utils.data.Subset(ds, keep.tolist())
            subj, clip = subj[keep], clip[keep]
        print(f"[data] {len(ds)} segments  subjects={len(np.unique(subj))}")

        model = build_labram(lc["repo_path"], lc["ckpt_path"], device,
                             finetuned=args.backbone)
        Z, ZL, ZR, y, CLS = extract(model, ds, device, args.batch_size)
        np.savez(args.cache, Z=Z, ZL=ZL, ZR=ZR, y=y, subj=subj, clip=clip, CLS=CLS)
        print(f"[cache] wrote {args.cache}")

    subjects = sorted(np.unique(subj))
    n_clips = int(clip.max())
    n_train_clips = max(1, round(n_clips * 2 / 3))   # 2/3 of clips train the subject probe
    reps = {}
    if CLS is not None:
        # First row: this is the vector the adversary actually attacks.
        reps["CLS token"] = CLS
    reps.update({
        "z_all  (62ch mean)":      Z,
        "z_L    (left only)":      ZL,
        "z_R    (right only)":     ZR,
        "z_L - z_R  (asymmetry)":  ZL - ZR,
        "z_L * z_R  (product)":    ZL * ZR,
        "[z_L ; z_R]  (concat)":   np.concatenate([ZL, ZR], 1),
    })

    n_cls = len(np.unique(y))
    print(f"\n[{args.dataset}]  {len(subjects)}명 · {n_clips}클립 · {n_cls}-class · "
          f"{len(y)} 세그먼트"
          + ("  (EA 적용)" if args.ea else ""))
    print(f"{'representation':26s} {'dim':>5} {'subject ID':>11} {'emotion LOSO':>14}")
    print(f"{'':26s} {'':>5} {'(낮을수록)':>11} {'(높을수록)':>14}")
    print("-" * 60)
    rows = []
    for name, X in reps.items():
        s = subject_probe(X, subj, clip, n_train_clips=n_train_clips)
        e, esd = emotion_probe(X, y, subj, subjects)
        rows.append((s, e))
        print(f"{name:26s} {X.shape[1]:5d} {s:11.4f} {e:9.4f}±{esd:.3f}", flush=True)
    print("-" * 60)
    print(f"{'chance':26s} {'':>5} {1/len(subjects):11.4f} {1/n_cls:14.4f}")

    # Does subject decodability trade off against emotion, or track it?
    S = np.array([r[0] for r in rows]); E = np.array([r[1] for r in rows])
    from scipy import stats as _st
    r_, p_ = _st.pearsonr(S, E)
    print(f"\n피험자 식별 vs 감정 성능   r={r_:+.3f}  p={p_:.4f}  (n={len(rows)})")


if __name__ == "__main__":
    main()
