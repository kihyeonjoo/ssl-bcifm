"""
Fine-tuning on SEED with the LaBraM + AsymmetryAdapter pipeline.

Loads a pretrained adapter (from ``pretrain_labram.py``) on top of a frozen
LaBraM backbone and fine-tunes on the 3-class emotion task.

Supports LOSO and subject-dependent evaluation via ``--eval_mode``.
"""

from __future__ import annotations

import argparse
import copy
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
from models.labram_wrapper import LaBraMWrapper
from models.asymmetry_adapter import AsymmetryAdapter
from models.classifier import AsymmetryFusionClassifier


# ── Adapter loader ──────────────────────────────────────────────────────────

def load_pretrained_adapter(cfg: dict, device: torch.device) -> AsymmetryAdapter:
    """Build LaBraM + AsymmetryAdapter and optionally load pretrained weights."""
    mc = cfg["model"]
    lc = cfg["labram"]
    fc = cfg["finetune"]

    labram = LaBraMWrapper(
        labram_repo=lc["repo_path"],
        ckpt_path=lc["ckpt_path"],
        ch_names=SEED_CH_NAMES,
        model_name=lc.get("model_name", "labram_base_patch200_200"),
        freeze=True,
    )

    adapter = AsymmetryAdapter(
        labram=labram,
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"],
    ).to(device)

    ckpt_path = fc.get("pretrained_ckpt")
    skip_adapter = fc.get("skip_adapter_pretrain", False)

    if skip_adapter or not ckpt_path:
        print("Adapter: random init (no pretraining loaded)")
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        adapter.adapter.load_state_dict(ckpt["adapter_state_dict"])
        print(f"Loaded adapter from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return adapter


# ── Classifier wrapper that forwards from raw EEG ──────────────────────────

class RawEEGClassifier(nn.Module):
    """Wraps AsymmetryFusionClassifier to accept raw EEG via the adapter.

    The existing AsymmetryFusionClassifier expects ``(left, right)`` hemisphere
    tensors. Here we call the adapter (LaBraM + hemisphere split +
    HemisphereAdapter) on raw EEG to produce ``(z_L, z_R)`` and feed them
    into the same fusion head.
    """

    def __init__(
        self,
        adapter: AsymmetryAdapter,
        d_model: int,
        n_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        # Reuse the fusion head from the original classifier (just the head)
        self.head = AsymmetryFusionClassifier(
            encoder=adapter,                # not used inside __init__ beyond param freeze
            d_model=d_model,
            n_classes=n_classes,
            dropout=dropout,
            freeze_encoder=False,           # we manage freezing via the adapter itself
            use_lora=False,
        ).head

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        z_L, z_R, _ = self.adapter(eeg)
        z_diff = z_L - z_R
        z_prod = z_L * z_R
        z_fused = torch.cat([z_L, z_R, z_diff, z_prod], dim=-1)
        return self.head(z_fused)


# ── Shared train/eval ───────────────────────────────────────────────────────

def _evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            label = batch["label"]
            logits = model(eeg)
            preds.append(logits.argmax(-1).cpu())
            labels.append(label)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def _train_and_eval(cfg, train_ds, test_ds, device, pretrained_adapter):
    fc = cfg["finetune"]
    mc = cfg["model"]

    train_loader = DataLoader(
        train_ds, batch_size=fc["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=fc["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Deepcopy adapter for fold independence (LaBraM weights are shared refs inside)
    adapter = copy.deepcopy(pretrained_adapter)
    # Re-freeze LaBraM after deepcopy (requires_grad is preserved, but be safe)
    for p in adapter.labram.parameters():
        p.requires_grad = False
    adapter.labram.frozen = True
    adapter.labram.model.eval()

    model = RawEEGClassifier(
        adapter=adapter,
        d_model=mc["d_model"],
        n_classes=fc["n_classes"],
        dropout=fc["dropout"],
    ).to(device)

    # Trainable params: adapter transformer + fusion head
    # LaBraM is frozen so it won't appear in grad flow anyway
    adapter_params = [p for p in model.adapter.adapter.parameters() if p.requires_grad]
    head_params    = list(model.head.parameters())

    params = [
        {"params": adapter_params, "lr": fc.get("adapter_lr", fc["lr"])},
        {"params": head_params,    "lr": fc["lr"]},
    ]
    optimizer = AdamW(params, lr=fc["lr"], weight_decay=fc["weight_decay"])

    total_epochs  = fc["epochs"]
    warmup_epochs = fc.get("warmup_epochs", 0)
    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=fc.get("label_smoothing", 0.0),
    )

    verbose    = fc.get("verbose", True)
    eval_every = fc.get("eval_every_epoch", False)
    best_acc, best_f1 = 0.0, 0.0

    for epoch in range(total_epochs):
        model.train()
        # Keep LaBraM in eval mode (BN-free but safe)
        model.adapter.labram.model.eval()

        epoch_loss, n_correct, n_total = 0.0, 0, 0
        for batch in train_loader:
            eeg   = batch["eeg"].to(device)
            label = batch["label"].to(device)

            logits = model(eeg)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
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
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "best_accuracy": max(best_acc, acc),
        "best_f1_macro": max(best_f1, f1),
    }


# ── LOSO + Subject-dependent ────────────────────────────────────────────────

def run_loso(cfg, wandb_project):
    fc = cfg["finetune"]
    dc = cfg["data"]
    device = torch.device(fc["device"] if torch.cuda.is_available() else "cpu")
    all_subjects = dc.get("subjects") or list(range(1, 16))

    pretrained_adapter = load_pretrained_adapter(cfg, device)

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
        train_subjects = [s for s in all_subjects if s != subj]
        train_ds = SEEDRawDataset(subjects=train_subjects, **ds_kwargs)
        test_ds  = SEEDRawDataset(subjects=[subj],         **ds_kwargs)

        m = _train_and_eval(cfg, train_ds, test_ds, device, pretrained_adapter)
        dt = time.time() - t0

        results["accuracy"].append(m["accuracy"])
        results["f1_macro"].append(m["f1_macro"])
        print(
            f"[Subject {subj:>2}]  acc={m['accuracy']:.4f}  "
            f"F1={m['f1_macro']:.4f}  ({dt:.1f}s)"
        )
        if use_wandb:
            wandb.log({
                "fold/subject":  subj,
                "fold/accuracy": m["accuracy"],
                "fold/f1_macro": m["f1_macro"],
                "fold/time_s":   dt,
            })

    _summary("LOSO", results, all_subjects, use_wandb)


def run_subject_dependent(cfg, wandb_project):
    fc = cfg["finetune"]
    dc = cfg["data"]
    device = torch.device(fc["device"] if torch.cuda.is_available() else "cpu")
    all_subjects = dc.get("subjects") or list(range(1, 16))

    pretrained_adapter = load_pretrained_adapter(cfg, device)

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
    # SEED official subject-dependent protocol: inside each session, film
    # clips 1..9 train and 10..15 test, averaged over the three sessions.
    # (The previous split — global trials <30 vs >=30 — was really a
    # cross-session split and is not comparable to published numbers.)
    n_train_clips = fc.get("sd_train_per_session", SD_TRAIN_CLIPS)
    sessions = dc.get("sessions") or [1, 2, 3]
    results = defaultdict(list)

    for subj in all_subjects:
        t0 = time.time()
        session_accs, session_f1s = [], []

        for sess in sessions:
            sess_ds = SEEDRawDataset(
                subjects=[subj], **{**ds_kwargs, "sessions": [sess]}
            )
            train_idx, test_idx = sess_ds.sd_split(n_train_clips)
            if not train_idx or not test_idx:
                continue

            sess_m = _train_and_eval(
                cfg, Subset(sess_ds, train_idx), Subset(sess_ds, test_idx),
                device, pretrained_adapter,
            )
            session_accs.append(sess_m["accuracy"])
            session_f1s.append(sess_m["f1_macro"])
            print(
                f"  [Subject {subj:>2} / Session {sess}]  "
                f"acc={sess_m['accuracy']:.4f}  F1={sess_m['f1_macro']:.4f}",
                flush=True,
            )

        m = {
            "accuracy": float(np.mean(session_accs)) if session_accs else 0.0,
            "f1_macro": float(np.mean(session_f1s))  if session_f1s  else 0.0,
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
                "fold/subject":  subj,
                "fold/accuracy": m["accuracy"],
                "fold/f1_macro": m["f1_macro"],
                "fold/time_s":   dt,
            })

    _summary("Subject-Dependent", results, all_subjects, use_wandb)


def _summary(name, results, subjects, use_wandb):
    acc_arr = np.array(results["accuracy"])
    f1_arr  = np.array(results["f1_macro"])

    print("\n" + "=" * 60)
    print(f"{name} Results")
    print("=" * 60)
    print(f"  Accuracy : {acc_arr.mean():.4f} ± {acc_arr.std():.4f}")
    print(f"  Macro-F1 : {f1_arr.mean():.4f} ± {f1_arr.std():.4f}")
    print("=" * 60)
    print(f"\n{'Subject':>8}  {'Accuracy':>8}  {'F1':>8}")
    print("-" * 28)
    for i, subj in enumerate(subjects):
        print(f"{subj:>8}  {acc_arr[i]:>8.4f}  {f1_arr[i]:>8.4f}")
    print("-" * 28)
    print(f"{'Mean':>8}  {acc_arr.mean():>8.4f}  {f1_arr.mean():>8.4f}")

    if use_wandb:
        import wandb
        wandb.log({
            "summary/accuracy_mean": acc_arr.mean(),
            "summary/accuracy_std":  acc_arr.std(),
            "summary/f1_macro_mean": f1_arr.mean(),
            "summary/f1_macro_std":  f1_arr.std(),
        })
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="SSL-BCIFM LaBraM Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/seed_labram.yaml")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument(
        "--eval_mode", type=str, default="loso",
        choices=["loso", "subject_dependent", "both"],
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.eval_mode in ("loso", "both"):
        print("\n" + "#" * 60)
        print("#  LaBraM LOSO Cross-Validation")
        print("#" * 60)
        run_loso(cfg, wandb_project=args.wandb_project)

    if args.eval_mode in ("subject_dependent", "both"):
        print("\n" + "#" * 60)
        print("#  LaBraM Subject-Dependent Cross-Validation")
        print("#" * 60)
        run_subject_dependent(cfg, wandb_project=args.wandb_project)


if __name__ == "__main__":
    main()
