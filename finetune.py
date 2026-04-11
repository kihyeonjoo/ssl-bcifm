"""
Downstream fine-tuning: Leave-One-Subject-Out cross-validation
==============================================================

1. Load pretrained encoder E_φ from a pretraining checkpoint.
2. Attach AsymmetryFusionClassifier head.
3. For each of the 15 SEED subjects:
     - Train on the other 14 subjects
     - Evaluate on the held-out subject
4. Report per-fold and mean accuracy / macro-F1.

Usage
-----
    python finetune.py --config configs/seed.yaml [--wandb_project ssl-bcifm-ft]
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

import yaml

from data.seed_dataset import SEEDDataset
from models.encoder import DualStreamEncoder
from models.classifier import AsymmetryFusionClassifier


# ── Pretrained encoder loading ───────────────────────────────────────────────

def load_pretrained_encoder(cfg: dict) -> DualStreamEncoder:
    """Build a DualStreamEncoder and load weights from a pretraining checkpoint.

    The checkpoint stores a PretrainModel state dict where encoder keys live
    under ``main_task.encoder.*``.  We strip that prefix to load into a
    standalone DualStreamEncoder.
    """
    mc = cfg["model"]
    encoder = DualStreamEncoder(
        n_channels_per_hemi=mc["n_channels_per_hemi"],
        n_bands=mc["n_bands"],
        d_model=mc["d_model"],
        cnn_channels=tuple(mc["cnn_channels"]),
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"],
    )

    ckpt_path = cfg["finetune"]["pretrained_ckpt"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    full_sd = ckpt["model_state_dict"]

    # Extract encoder keys: main_task.encoder.xxx → encoder.xxx → xxx
    prefix = "main_task.encoder."
    enc_sd = {
        k[len(prefix):]: v
        for k, v in full_sd.items()
        if k.startswith(prefix)
    }

    encoder.load_state_dict(enc_sd)
    print(f"Loaded pretrained encoder from {ckpt_path} "
          f"(epoch {ckpt.get('epoch', '?')})")
    return encoder


# ── Single fold ──────────────────────────────────────────────────────────────

def run_fold(
    cfg: dict,
    test_subj: int,
    all_subjects: list[int],
    device: torch.device,
    pretrained_encoder: DualStreamEncoder,
) -> dict[str, float]:
    """Train on all subjects except *test_subj*, evaluate on *test_subj*."""
    fc = cfg["finetune"]
    dc = cfg["data"]
    mc = cfg["model"]

    train_subjects = [s for s in all_subjects if s != test_subj]

    # ── Datasets ──────────────────────────────────────────────────────────
    ds_kwargs = dict(
        root=dc["root"],
        sessions=dc.get("sessions"),
        segment_length=dc["segment_length"],
        step=dc["step"],
        n_fft=dc["n_fft"],
        hop_length=dc["hop_length"],
    )
    train_ds = SEEDDataset(subjects=train_subjects, **ds_kwargs)
    test_ds  = SEEDDataset(subjects=[test_subj],    **ds_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=fc["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=fc["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    encoder = copy.deepcopy(pretrained_encoder)
    model = AsymmetryFusionClassifier(
        encoder=encoder,
        d_model=mc["d_model"],
        n_classes=fc["n_classes"],
        dropout=fc["dropout"],
        freeze_encoder=fc["freeze_encoder"],
    ).to(device)

    # ── Optimizer: differential LR ────────────────────────────────────────
    if fc["freeze_encoder"]:
        params = model.head.parameters()
    else:
        params = [
            {"params": model.encoder.parameters(), "lr": fc["encoder_lr"]},
            {"params": model.head.parameters(),     "lr": fc["lr"]},
        ]
    optimizer = AdamW(params, lr=fc["lr"], weight_decay=fc["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=fc["epochs"], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # ── Training ──────────────────────────────────────────────────────────
    for epoch in range(fc["epochs"]):
        model.train()
        for batch in train_loader:
            left  = batch["left"].to(device)
            right = batch["right"].to(device)
            label = batch["label"].to(device)

            logits = model(left, right)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

    # ── Evaluation ────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            left  = batch["left"].to(device)
            right = batch["right"].to(device)
            label = batch["label"]

            logits = model(left, right)
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(label)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")

    return {"accuracy": acc, "f1_macro": f1}


# ── LOSO cross-validation ───────────────────────────────────────────────────

def loso_cv(cfg: dict, wandb_project: str | None = None) -> None:
    fc = cfg["finetune"]
    device = torch.device(fc["device"] if torch.cuda.is_available() else "cpu")
    all_subjects = list(range(1, 16))  # SEED: 15 subjects

    # Load pretrained encoder once
    pretrained_encoder = load_pretrained_encoder(cfg)

    # ── wandb ─────────────────────────────────────────────────────────────
    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config=cfg)

    # ── Run folds ─────────────────────────────────────────────────────────
    results = defaultdict(list)

    for subj in all_subjects:
        t0 = time.time()
        fold_metrics = run_fold(
            cfg, test_subj=subj, all_subjects=all_subjects,
            device=device, pretrained_encoder=pretrained_encoder,
        )
        dt = time.time() - t0

        results["accuracy"].append(fold_metrics["accuracy"])
        results["f1_macro"].append(fold_metrics["f1_macro"])

        print(
            f"[Subject {subj:>2}]  "
            f"acc={fold_metrics['accuracy']:.4f}  "
            f"F1={fold_metrics['f1_macro']:.4f}  "
            f"({dt:.1f}s)"
        )

        if use_wandb:
            wandb.log({
                "fold/subject":  subj,
                "fold/accuracy": fold_metrics["accuracy"],
                "fold/f1_macro": fold_metrics["f1_macro"],
                "fold/time_s":   dt,
            })

    # ── Summary ───────────────────────────────────────────────────────────
    acc_arr = np.array(results["accuracy"])
    f1_arr  = np.array(results["f1_macro"])

    print("\n" + "=" * 60)
    print("LOSO Cross-Validation Results")
    print("=" * 60)
    print(f"  Accuracy : {acc_arr.mean():.4f} ± {acc_arr.std():.4f}")
    print(f"  Macro-F1 : {f1_arr.mean():.4f} ± {f1_arr.std():.4f}")
    print("=" * 60)

    # Per-subject table
    print(f"\n{'Subject':>8}  {'Accuracy':>8}  {'F1':>8}")
    print("-" * 28)
    for i, subj in enumerate(all_subjects):
        print(f"{subj:>8}  {acc_arr[i]:>8.4f}  {f1_arr[i]:>8.4f}")
    print("-" * 28)
    print(f"{'Mean':>8}  {acc_arr.mean():>8.4f}  {f1_arr.mean():>8.4f}")
    print(f"{'Std':>8}  {acc_arr.std():>8.4f}  {f1_arr.std():>8.4f}")

    if use_wandb:
        wandb.log({
            "summary/accuracy_mean": acc_arr.mean(),
            "summary/accuracy_std":  acc_arr.std(),
            "summary/f1_macro_mean": f1_arr.mean(),
            "summary/f1_macro_std":  f1_arr.std(),
        })
        # Log per-subject table as wandb.Table
        table = wandb.Table(
            columns=["subject", "accuracy", "f1_macro"],
            data=[[s, float(acc_arr[i]), float(f1_arr[i])]
                  for i, s in enumerate(all_subjects)],
        )
        wandb.log({"results/per_subject": table})
        wandb.finish()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SSL-BCIFM Fine-tuning (LOSO)")
    parser.add_argument("--config", type=str, default="configs/seed.yaml")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project name (omit to disable logging)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    loso_cv(cfg, wandb_project=args.wandb_project)


if __name__ == "__main__":
    main()
