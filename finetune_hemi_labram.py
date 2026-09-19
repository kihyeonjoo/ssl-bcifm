"""
Hemisphere-aware LaBraM fine-tuning for SEED emotion recognition.
=================================================================

Architecture
------------
    Raw EEG (62ch)
       │
       ├─ Left 27ch ──► LaBraM + LoRA (shared instance) ──► CLS_L
       │
       └─ Right 27ch ──► LaBraM + LoRA (shared instance) ──► CLS_R
                                                               │
                                  [CLS_L ; CLS_R ; CLS_L-CLS_R ; CLS_L⊙CLS_R]
                                                               │
                                                       MLP Head → 3 classes

Key design:
  1. LaBraM is called TWICE — once per hemisphere — with shared weights.
     This preserves hemisphere locality (no cross-hemisphere attention leakage).
  2. LoRA adapters on LaBraM's FFN layers make it emotion-adaptive.
  3. The 4-way asymmetry fusion head captures lateralisation explicitly.
  4. No separate SSL pretraining stage needed — fine-tune directly on labels.

Usage
-----
    python finetune_hemi_labram.py --config configs/seed_labram.yaml \\
        --eval_mode both --wandb_project ssl-bcifm-hemi
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score

import yaml

from data.seed_raw_dataset import SEEDRawDataset, SEED_CH_NAMES, SD_TRAIN_CLIPS
from data.preprocessing import LEFT_IDX, RIGHT_IDX, LEFT_CH, RIGHT_CH


# ── LaBraM hemisphere wrapper ───────────────────────────────────────────────

def _register_labram(repo: str):
    repo = os.path.abspath(os.path.expanduser(repo))
    if repo not in sys.path:
        sys.path.insert(0, repo)


# LaBraM standard_1020 (copied from labram_wrapper.py to avoid circular import)
_LABRAM_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


def _get_input_chans(ch_names):
    return [0] + [_LABRAM_1020.index(ch) + 1 for ch in ch_names]


class HemiLaBraMClassifier(nn.Module):
    """LaBraM called per-hemisphere with shared weights + LoRA + fusion head.

    LaBraM is run TWICE (once per hemisphere) using the SAME model instance.
    This ensures:
      - Hemisphere-specific features (no cross-hemisphere attention leakage)
      - True weight sharing (z_L - z_R measures genuine asymmetry)
      - LoRA adapts LaBraM to be emotion-aware
    """

    def __init__(
        self,
        labram_repo: str,
        ckpt_path: str,
        d_model: int = 200,    # LaBraM-base embed_dim
        n_classes: int = 3,
        dropout: float = 0.2,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
    ) -> None:
        super().__init__()
        _register_labram(labram_repo)

        import modeling_finetune  # noqa
        from timm.models import create_model

        # Build LaBraM
        model = create_model(
            "labram_base_patch200_200",
            pretrained=False,
            num_classes=0,
            drop_rate=0.0,
            drop_path_rate=0.1,
            use_mean_pooling=False,
            init_scale=0.001,
            use_rel_pos_bias=True,
            use_abs_pos_emb=True,
            init_values=0.1,
        )

        # Load pretrained weights
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        if any(k.startswith("student.") for k in sd.keys()):
            sd = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}
        drop = ("head.", "lm_head.", "projection_head.", "mask_token", "logit_scale")
        sd = {k: v for k, v in sd.items() if not any(k.startswith(p) for p in drop)}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[LaBraM] loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        self.labram = model
        self.embed_dim = model.embed_dim

        # Freeze all LaBraM weights, then inject LoRA on FFN layers
        for p in self.labram.parameters():
            p.requires_grad = False

        from models.lora import inject_lora, disable_transformer_fast_path, count_trainable_parameters

        # LaBraM uses custom Block, not nn.TransformerEncoderLayer.
        # Target the MLP's fc1 and fc2 inside each Block.
        n_wrapped = inject_lora(
            self.labram,
            target_names=("fc1", "fc2"),
            r=lora_r,
            alpha=lora_alpha,
        )

        # LaBraM's Block doesn't use PyTorch's fast path, so no need for
        # disable_transformer_fast_path (it's a custom forward).

        trainable, total = count_trainable_parameters(self.labram)
        print(
            f"[LaBraM] LoRA: {n_wrapped} layers wrapped (r={lora_r}).  "
            f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )

        # Precompute channel indices for left/right hemispheres
        self.left_input_chans = _get_input_chans(LEFT_CH)
        self.right_input_chans = _get_input_chans(RIGHT_CH)

        # Store channel-to-index for raw EEG slicing
        self.register_buffer(
            "left_idx", torch.tensor(LEFT_IDX, dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "right_idx", torch.tensor(RIGHT_IDX, dtype=torch.long), persistent=False,
        )

        # Fusion head: [z_L; z_R; z_L-z_R; z_L⊙z_R] → 4 * d_model
        fusion_dim = 4 * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def _encode_hemisphere(self, eeg_hemi, input_chans):
        """Run LaBraM on one hemisphere's channels → CLS token."""
        # eeg_hemi: (B, 27, P, 200)
        # CLS token = x[:, 0] when return_patch_tokens=False, return_all_tokens=False
        cls = self.labram.forward_features(
            eeg_hemi,
            input_chans=input_chans,
            return_patch_tokens=False,
            return_all_tokens=False,
        )
        return cls  # (B, embed_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg : (B, 62, n_patches, patch_size=200)

        Returns
        -------
        logits : (B, n_classes)
        """
        # Split hemispheres from raw EEG
        eeg_L = eeg.index_select(1, self.left_idx)    # (B, 27, P, 200)
        eeg_R = eeg.index_select(1, self.right_idx)   # (B, 27, P, 200)

        # Shared LaBraM + LoRA, called independently per hemisphere
        z_L = self._encode_hemisphere(eeg_L, self.left_input_chans)   # (B, d)
        z_R = self._encode_hemisphere(eeg_R, self.right_input_chans)  # (B, d)

        # 4-way asymmetry fusion
        z_diff = z_L - z_R
        z_prod = z_L * z_R
        z_fused = torch.cat([z_L, z_R, z_diff, z_prod], dim=-1)  # (B, 4d)

        return self.head(z_fused)


# ── Train / eval ────────────────────────────────────────────────────────────

def _evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            logits = model(eeg)
            preds.append(logits.argmax(-1).cpu())
            labels.append(batch["label"])
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def _train_and_eval(cfg, train_ds, test_ds, device):
    fc = cfg["finetune"]
    lc = cfg["labram"]

    train_loader = DataLoader(
        train_ds, batch_size=fc["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=fc["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    model = HemiLaBraMClassifier(
        labram_repo=lc["repo_path"],
        ckpt_path=lc["ckpt_path"],
        d_model=200,  # labram-base embed_dim
        n_classes=fc["n_classes"],
        dropout=fc["dropout"],
        lora_r=fc.get("lora_r", 8),
        lora_alpha=fc.get("lora_alpha", 16.0),
    ).to(device)

    # Param groups: LoRA params + head params
    lora_params = [p for p in model.labram.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    params = [
        {"params": lora_params, "lr": fc.get("lora_lr", 2e-4)},
        {"params": head_params, "lr": fc["lr"]},
    ]
    optimizer = AdamW(params, lr=fc["lr"], weight_decay=fc["weight_decay"])

    total_epochs  = fc["epochs"]
    warmup_epochs = fc.get("warmup_epochs", 0)
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(label_smoothing=fc.get("label_smoothing", 0.0))
    verbose = fc.get("verbose", True)
    eval_every = fc.get("eval_every_epoch", False)
    best_acc, best_f1 = 0.0, 0.0

    for epoch in range(total_epochs):
        model.train()
        epoch_loss, n_correct, n_total = 0.0, 0, 0

        for batch in train_loader:
            eeg   = batch["eeg"].to(device)
            label = batch["label"].to(device)

            logits = model(eeg)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()

            epoch_loss += loss.item() * label.size(0)
            n_correct  += (logits.argmax(-1) == label).sum().item()
            n_total    += label.size(0)

        scheduler.step()
        train_loss = epoch_loss / max(n_total, 1)
        train_acc  = n_correct / max(n_total, 1)

        if verbose:
            lr_now = optimizer.param_groups[-1]["lr"]
            msg = (
                f"    epoch {epoch+1:>2}/{total_epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"lr={lr_now:.2e}"
            )
            if eval_every:
                eval_acc, eval_f1 = _evaluate(model, test_loader, device)
                if eval_acc > best_acc:
                    best_acc, best_f1 = eval_acc, eval_f1
                msg += f"  test_acc={eval_acc:.4f}  test_f1={eval_f1:.4f}"
            print(msg, flush=True)

    acc, f1 = _evaluate(model, test_loader, device)
    return {"accuracy": acc, "f1_macro": f1, "best_accuracy": max(best_acc, acc)}


# ── LOSO + Subject-dependent ────────────────────────────────────────────────

def run_cv(cfg, mode, wandb_project):
    fc = cfg["finetune"]
    dc = cfg["data"]
    device = torch.device(fc["device"] if torch.cuda.is_available() else "cpu")
    all_subjects = dc.get("subjects") or list(range(1, 16))

    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config=cfg, reinit=True)

    ds_kwargs = dict(
        root=dc["root"],
        sessions=dc.get("sessions"),
        segment_length=dc["segment_length"],
        step=dc["step"],
        patch_size=dc["patch_size"],
        norm=dc.get("norm", "scale100"),
        ea=dc.get("ea", False),
        ea_scope=dc.get("ea_scope", "session"),
        ea_scale=dc.get("ea_scale", 0.2),
        ea_trim=dc.get("ea_trim", 0.05),
        ea_eps=dc.get("ea_eps", 1e-6),
    )

    results = defaultdict(list)
    for subj in all_subjects:
        t0 = time.time()

        if mode == "loso":
            train_subjects = [s for s in all_subjects if s != subj]
            train_ds = SEEDRawDataset(subjects=train_subjects, **ds_kwargs)
            test_ds  = SEEDRawDataset(subjects=[subj],         **ds_kwargs)
            m = _train_and_eval(cfg, train_ds, test_ds, device)

        else:  # subject_dependent — within-session split
            # Standard SEED protocol: within each session,
            # first 9 trials train / last 6 trials test.
            # Average across 3 sessions.
            sd_train_per_session = fc.get("sd_train_per_session", SD_TRAIN_CLIPS)
            session_accs, session_f1s = [], []

            for sess in [1, 2, 3]:
                sess_kwargs = {**ds_kwargs, "sessions": [sess]}
                sess_ds = SEEDRawDataset(subjects=[subj], **sess_kwargs)
                # Split on the film-clip index, not on load order
                train_idx, test_idx = sess_ds.sd_split(sd_train_per_session)
                if not train_idx or not test_idx:
                    continue
                sess_m = _train_and_eval(
                    cfg, Subset(sess_ds, train_idx), Subset(sess_ds, test_idx), device,
                )
                session_accs.append(sess_m["accuracy"])
                session_f1s.append(sess_m["f1_macro"])
                print(
                    f"  [Subject {subj:>2} / Session {sess}]  "
                    f"acc={sess_m['accuracy']:.4f}  F1={sess_m['f1_macro']:.4f}  "
                    f"best={sess_m['best_accuracy']:.4f}"
                )

            # Average across sessions for this subject
            m = {
                "accuracy": float(np.mean(session_accs)) if session_accs else 0.0,
                "f1_macro": float(np.mean(session_f1s))  if session_f1s  else 0.0,
                "best_accuracy": 0.0,
            }

        dt = time.time() - t0
        results["accuracy"].append(m["accuracy"])
        results["f1_macro"].append(m["f1_macro"])
        print(
            f"[Subject {subj:>2}]  acc={m['accuracy']:.4f}  "
            f"F1={m['f1_macro']:.4f}  ({dt:.1f}s)"
        )
        if use_wandb:
            wandb.log({
                "fold/subject": subj,
                "fold/accuracy": m["accuracy"],
                "fold/f1_macro": m["f1_macro"],
            })

    acc_arr = np.array(results["accuracy"])
    f1_arr  = np.array(results["f1_macro"])
    label = "LOSO" if mode == "loso" else "Subject-Dependent"

    print(f"\n{'=' * 60}")
    print(f"{label} Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy : {acc_arr.mean():.4f} ± {acc_arr.std():.4f}")
    print(f"  Macro-F1 : {f1_arr.mean():.4f} ± {f1_arr.std():.4f}")
    print(f"{'=' * 60}")
    print(f"\n{'Subject':>8}  {'Accuracy':>8}  {'F1':>8}")
    print("-" * 28)
    for i, subj in enumerate(all_subjects):
        print(f"{subj:>8}  {acc_arr[i]:>8.4f}  {f1_arr[i]:>8.4f}")
    print("-" * 28)
    print(f"{'Mean':>8}  {acc_arr.mean():>8.4f}  {f1_arr.mean():>8.4f}")

    if use_wandb:
        wandb.log({
            "summary/accuracy_mean": acc_arr.mean(),
            "summary/accuracy_std":  acc_arr.std(),
            "summary/f1_macro_mean": f1_arr.mean(),
            "summary/f1_macro_std":  f1_arr.std(),
        })
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Hemisphere-aware LaBraM Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/seed_labram.yaml")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--eval_mode", type=str, default="subject_dependent",
                        choices=["loso", "subject_dependent", "both"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.eval_mode in ("loso", "both"):
        print("\n" + "#" * 60)
        print("#  Hemisphere LaBraM LOSO")
        print("#" * 60)
        run_cv(cfg, "loso", args.wandb_project)

    if args.eval_mode in ("subject_dependent", "both"):
        print("\n" + "#" * 60)
        print("#  Hemisphere LaBraM Subject-Dependent")
        print("#" * 60)
        run_cv(cfg, "subject_dependent", args.wandb_project)


if __name__ == "__main__":
    main()
