"""
LaBraM 62ch fine-tuning + hemisphere asymmetry auxiliary loss.
==============================================================

Architecture
------------
    Raw EEG (B, 62, n_patches, 200)
       │
       ▼
    LaBraM 62ch (LoRA fine-tune)
       │
       ├──► CLS token (B, 200) ──► main_head → logits_main (B, 3)
       │
       └──► patch tokens (B, 62*P, 200)
            │
            ▼ reshape → (B, 62, P, 200)
            ├─ LEFT_IDX  → mean → z_L (B, 200)
            └─ RIGHT_IDX → mean → z_R (B, 200)
                │
                ▼
            [z_L ; z_R ; z_L-z_R ; z_L⊙z_R] (B, 800)
                │
                ▼
            asym_head → logits_asym (B, 3)

    Loss = CE(logits_main) + λ · CE(logits_asym)

Key: LaBraM runs ONCE on all 62 channels (as designed), but we extract
hemisphere-specific features from its internal patch tokens as an
auxiliary signal. This preserves LaBraM's architecture while adding
asymmetry inductive bias.

Usage
-----
    python finetune_labram_hemi_aux.py --config configs/seed_labram.yaml \\
        --eval_mode subject_dependent --wandb_project ssl-bcifm-hemiaux
"""

from __future__ import annotations

import argparse
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
from data.preprocessing import LEFT_IDX, RIGHT_IDX


# ── LaBraM setup helpers ────────────────────────────────────────────────────

def _register_labram(repo: str):
    repo = os.path.abspath(os.path.expanduser(repo))
    if repo not in sys.path:
        sys.path.insert(0, repo)

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


# ── Model ────────────────────────────────────────────────────────────────────

class LaBraMHemiAuxClassifier(nn.Module):
    """LaBraM 62ch + hemisphere asymmetry auxiliary classifier.

    LaBraM processes all 62 channels together (as designed).
    Two classification heads:
      - main_head: CLS token → 3 classes (standard)
      - asym_head: [z_L; z_R; z_L-z_R; z_L⊙z_R] → 3 classes (auxiliary)

    Parameters
    ----------
    labram_repo, ckpt_path : LaBraM paths
    n_classes   : int
    dropout     : float
    lora_r, lora_alpha : LoRA config
    lambda_asym : float — weight for auxiliary asymmetry loss
    """

    def __init__(
        self,
        labram_repo: str,
        ckpt_path: str,
        n_classes: int = 3,
        dropout: float = 0.2,
        lambda_asym: float = 0.5,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
    ) -> None:
        super().__init__()
        _register_labram(labram_repo)

        import modeling_finetune  # noqa
        from timm.models import create_model

        # Build LaBraM (62ch, standard config — matches official)
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
            sd = {k[len("student."):]: v for k, v in sd.items()
                  if k.startswith("student.")}
        drop = ("head.", "lm_head.", "projection_head.", "mask_token", "logit_scale")
        sd = {k: v for k, v in sd.items()
              if not any(k.startswith(p) for p in drop)}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[LaBraM] loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        self.labram = model
        self.embed_dim = model.embed_dim  # 200
        self.lambda_asym = lambda_asym
        self.use_lora = use_lora

        if use_lora:
            from models.lora import inject_lora, count_trainable_parameters
            for p in self.labram.parameters():
                p.requires_grad = False
            n_wrapped = inject_lora(
                self.labram, target_names=("fc1", "fc2"),
                r=lora_r, alpha=lora_alpha,
            )
            trainable, total = count_trainable_parameters(self.labram)
            print(f"[LaBraM] LoRA: {n_wrapped} layers (r={lora_r}).  "
                  f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        else:
            # Full fine-tune — all parameters trainable
            trainable = sum(p.numel() for p in self.labram.parameters())
            print(f"[LaBraM] Full fine-tune: {trainable:,} params all trainable")

        # 62-ch input_chans for LaBraM positional embedding
        self.input_chans = _get_input_chans(SEED_CH_NAMES)

        # Hemisphere channel indices (into the 62-ch ordering)
        self.register_buffer(
            "left_idx", torch.tensor(LEFT_IDX, dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "right_idx", torch.tensor(RIGHT_IDX, dtype=torch.long), persistent=False,
        )

        d = self.embed_dim  # 200

        # Main head: CLS token → emotion class
        self.main_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, n_classes),
        )

        # Asymmetry auxiliary head: [z_L; z_R; z_L-z_R; z_L⊙z_R] → emotion
        self.asym_head = nn.Sequential(
            nn.LayerNorm(4 * d),
            nn.Linear(4 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, n_classes),
        )

    def forward(self, eeg: torch.Tensor):
        """
        Parameters
        ----------
        eeg : (B, 62, n_patches, 200)

        Returns
        -------
        dict with keys: logits_main, logits_asym, z_L, z_R
        """
        B, C, P, T = eeg.shape

        # LaBraM forward — get ALL tokens (CLS + patch tokens)
        all_tokens = self.labram.forward_features(
            eeg,
            input_chans=self.input_chans,
            return_all_tokens=True,
        )  # (B, 1 + C*P, embed_dim)

        cls_token = all_tokens[:, 0]          # (B, d) — CLS
        patch_tokens = all_tokens[:, 1:]      # (B, C*P, d)

        # Main classification from CLS
        logits_main = self.main_head(cls_token)

        # Hemisphere features from patch tokens
        patch_tokens = patch_tokens.reshape(B, C, P, self.embed_dim)
        # Average over patches within each channel → (B, C, d)
        channel_feats = patch_tokens.mean(dim=2)

        z_L = channel_feats[:, self.left_idx].mean(dim=1)   # (B, d)
        z_R = channel_feats[:, self.right_idx].mean(dim=1)  # (B, d)

        # 4-way asymmetry fusion
        z_fused = torch.cat([z_L, z_R, z_L - z_R, z_L * z_R], dim=-1)
        logits_asym = self.asym_head(z_fused)

        return {
            "logits_main": logits_main,
            "logits_asym": logits_asym,
            "z_L": z_L,
            "z_R": z_R,
        }

    def compute_loss(self, out: dict, label: torch.Tensor, criterion):
        loss_main = criterion(out["logits_main"], label)
        loss_asym = criterion(out["logits_asym"], label)
        loss = loss_main + self.lambda_asym * loss_asym
        return loss, loss_main, loss_asym


# ── Train / eval ─────────────────────────────────────────────────────────────

# ── Layer-wise LR decay (matches LaBraM official) ───────────────────────────

def _get_layer_id(name: str, num_layers: int = 12) -> int:
    """Assign a depth index for layer-wise LR decay.

    Layer 0 = patch_embed / cls_token / pos_embed / time_embed  (lowest LR)
    Layer 1..12 = transformer blocks
    Layer 13 = norm / heads                                     (highest LR)
    """
    if name.startswith("labram.patch_embed") or any(
        name.endswith(s) for s in ("cls_token", "pos_embed", "time_embed")
    ):
        return 0
    if "labram.blocks." in name:
        block_id = int(name.split("labram.blocks.")[1].split(".")[0])
        return block_id + 1
    return num_layers + 1


def build_layer_decay_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    layer_decay: float,
    num_layers: int = 12,
) -> AdamW:
    """Build AdamW with layer-wise LR decay (LaBraM official recipe)."""
    param_groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _get_layer_id(name, num_layers)
        lr_scale = layer_decay ** (num_layers + 1 - layer_id)
        wd = 0.0 if (param.ndim == 1 or name.endswith(".bias")) else weight_decay
        group_key = f"layer_{layer_id}_wd_{wd}"
        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [], "lr": lr * lr_scale, "weight_decay": wd,
            }
        param_groups[group_key]["params"].append(param)

    print(f"[Optimizer] {len(param_groups)} param groups, "
          f"lr range [{lr * layer_decay**(num_layers+1):.2e}, {lr:.2e}]")
    return AdamW(list(param_groups.values()))


def build_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Cosine schedule with linear warmup (per-step, LaBraM official)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _amp_ctx(device, enabled: bool):
    """bf16 autocast, matching LaBraM official (engine_for_finetuning.py).

    bf16 keeps FP32's exponent range, so no GradScaler / loss scaling is
    needed — unlike fp16.
    """
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def _evaluate(model, loader, device, amp: bool = False):
    model.eval()
    preds_main, preds_asym, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            with _amp_ctx(device, amp):
                out = model(eeg)
            out = {k: v.float() for k, v in out.items()
                   if isinstance(v, torch.Tensor)}
            # Ensemble: average main + asym logits
            logits = out["logits_main"] + model.lambda_asym * out["logits_asym"]
            preds_main.append(logits.argmax(-1).cpu())
            labels.append(batch["label"])
    y_pred = torch.cat(preds_main).numpy()
    y_true = torch.cat(labels).numpy()
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def _train_and_eval(cfg, train_ds, test_ds, device, val_ds=None):
    """Train once and report test metrics.

    When ``val_ds`` is given the reported numbers are the test metrics at the
    epoch with the best VALIDATION macro-F1 — the test set never influences
    model selection.  Without it, the last epoch is reported.
    """
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
    val_loader = None if val_ds is None else DataLoader(
        val_ds, batch_size=fc["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    use_lora = fc.get("use_lora", False)
    model = LaBraMHemiAuxClassifier(
        labram_repo=lc["repo_path"],
        ckpt_path=lc["ckpt_path"],
        n_classes=fc["n_classes"],
        dropout=fc["dropout"],
        lambda_asym=fc.get("lambda_asym", 0.0),
        use_lora=use_lora,
        lora_r=fc.get("lora_r", 8),
        lora_alpha=fc.get("lora_alpha", 16.0),
    ).to(device)

    total_epochs = fc["epochs"]
    warmup_epochs = fc.get("warmup_epochs", 5)

    if use_lora:
        # LoRA mode: simple param groups
        lora_params = [p for p in model.labram.parameters() if p.requires_grad]
        head_params = list(model.main_head.parameters()) + list(model.asym_head.parameters())
        optimizer = AdamW([
            {"params": lora_params, "lr": fc.get("lora_lr", 2e-4)},
            {"params": head_params, "lr": fc["lr"]},
        ], lr=fc["lr"], weight_decay=fc["weight_decay"])
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        step_scheduler = False
    else:
        # Full FT mode: layer-wise LR decay (LaBraM official recipe)
        optimizer = build_layer_decay_optimizer(
            model,
            lr=fc["lr"],
            weight_decay=fc["weight_decay"],
            layer_decay=fc.get("layer_decay", 0.65),
        )
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        scheduler = build_cosine_scheduler(optimizer, total_steps, warmup_steps)
        step_scheduler = True  # step per iteration, not per epoch

    criterion = nn.CrossEntropyLoss(label_smoothing=fc.get("label_smoothing", 0.0))
    verbose = fc.get("verbose", True)
    eval_every = fc.get("eval_every_epoch", False)
    amp = fc.get("amp", True)
    best_acc = 0.0
    # model selection state (validation-driven)
    best_val_f1 = -1.0
    sel = None            # (test_acc, test_f1, epoch) at best val F1

    for epoch in range(total_epochs):
        model.train()
        epoch_loss, n_correct, n_total = 0.0, 0, 0

        for batch in train_loader:
            eeg   = batch["eeg"].to(device)
            label = batch["label"].to(device)

            with _amp_ctx(device, amp):
                out = model(eeg)
                loss, loss_m, loss_a = model.compute_loss(out, label, criterion)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=fc.get("clip_grad", 3.0),
            )
            optimizer.step()
            if step_scheduler:
                scheduler.step()

            epoch_loss += loss.item() * label.size(0)
            # Use ensembled prediction for train acc tracking
            logits = (out["logits_main"] + model.lambda_asym * out["logits_asym"]).float()
            n_correct += (logits.argmax(-1) == label).sum().item()
            n_total   += label.size(0)

        if not step_scheduler:
            scheduler.step()
        train_loss = epoch_loss / max(n_total, 1)
        train_acc  = n_correct / max(n_total, 1)

        lr_now = optimizer.param_groups[-1]["lr"]
        msg = (
            f"    epoch {epoch+1:>2}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"lr={lr_now:.2e}"
        )

        if val_loader is not None:
            # Select on validation only; test is evaluated for logging but
            # never used to choose the reported epoch.
            val_acc, val_f1 = _evaluate(model, val_loader, device, amp)
            te_acc, te_f1 = _evaluate(model, test_loader, device, amp)
            msg += f"  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
            msg += f"  test_acc={te_acc:.4f}  test_f1={te_f1:.4f}"
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                sel = (te_acc, te_f1, epoch + 1)
                msg += "  *"
        elif eval_every:
            eval_acc, eval_f1 = _evaluate(model, test_loader, device, amp)
            if eval_acc > best_acc:
                best_acc = eval_acc
            msg += f"  test_acc={eval_acc:.4f}  test_f1={eval_f1:.4f}"

        if verbose:
            print(msg, flush=True)

    acc, f1 = _evaluate(model, test_loader, device, amp)
    if sel is not None:
        te_acc, te_f1, ep = sel
        return {
            "accuracy": te_acc, "f1_macro": te_f1,     # test @ best-val epoch
            "selected_epoch": ep, "val_f1": best_val_f1,
            "final_accuracy": acc, "final_f1_macro": f1,
            "best_accuracy": te_acc,
        }
    return {"accuracy": acc, "f1_macro": f1, "best_accuracy": max(best_acc, acc)}


# ── All-subject cross-session (LaBraM official protocol) ─────────────────────

def run_all_subject_cross_session(cfg, wandb_project):
    """LaBraM official SEED protocol:
    Train = all 15 subjects × session 1
    Val   = all 15 subjects × session 2
    Test  = all 15 subjects × session 3
    Single model, single run.
    """
    fc = cfg["finetune"]
    dc = cfg["data"]
    device = torch.device(fc["device"] if torch.cuda.is_available() else "cpu")

    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config=cfg, reinit=True)

    ds_kwargs = dict(
        root=dc["root"],
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

    train_ds = SEEDRawDataset(subjects=list(range(1, 16)), sessions=[1], **ds_kwargs)
    val_ds   = SEEDRawDataset(subjects=list(range(1, 16)), sessions=[2], **ds_kwargs)
    test_ds  = SEEDRawDataset(subjects=list(range(1, 16)), sessions=[3], **ds_kwargs)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    lc = cfg["labram"]
    train_loader = DataLoader(train_ds, batch_size=fc["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=fc["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=fc["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    use_lora = fc.get("use_lora", False)
    model = LaBraMHemiAuxClassifier(
        labram_repo=lc["repo_path"],
        ckpt_path=lc["ckpt_path"],
        n_classes=fc["n_classes"],
        dropout=fc["dropout"],
        lambda_asym=fc.get("lambda_asym", 0.0),
        use_lora=use_lora,
        lora_r=fc.get("lora_r", 8),
        lora_alpha=fc.get("lora_alpha", 16.0),
    ).to(device)

    # Optimizer
    if use_lora:
        lora_params = [p for p in model.labram.parameters() if p.requires_grad]
        head_params = list(model.main_head.parameters()) + list(model.asym_head.parameters())
        optimizer = AdamW([
            {"params": lora_params, "lr": fc.get("lora_lr", 2e-4)},
            {"params": head_params, "lr": fc["lr"]},
        ], lr=fc["lr"], weight_decay=fc["weight_decay"])
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=fc.get("warmup_epochs", 5))
        cosine = CosineAnnealingLR(optimizer,
                                    T_max=max(fc["epochs"] - fc.get("warmup_epochs", 5), 1),
                                    eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                  milestones=[fc.get("warmup_epochs", 5)])
        step_scheduler = False
    else:
        optimizer = build_layer_decay_optimizer(
            model, lr=fc["lr"], weight_decay=fc["weight_decay"],
            layer_decay=fc.get("layer_decay", 0.65),
        )
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = fc["epochs"] * steps_per_epoch
        warmup_steps = fc.get("warmup_epochs", 5) * steps_per_epoch
        scheduler = build_cosine_scheduler(optimizer, total_steps, warmup_steps)
        step_scheduler = True

    criterion = nn.CrossEntropyLoss(label_smoothing=fc.get("label_smoothing", 0.1))
    best_val_acc, best_test_acc, best_test_f1 = 0.0, 0.0, 0.0

    for epoch in range(fc["epochs"]):
        model.train()
        epoch_loss, n_correct, n_total = 0.0, 0, 0

        for batch in train_loader:
            eeg   = batch["eeg"].to(device)
            label = batch["label"].to(device)

            out = model(eeg)
            loss, _, _ = model.compute_loss(out, label, criterion)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=fc.get("clip_grad", 3.0),
            )
            optimizer.step()
            if step_scheduler:
                scheduler.step()

            epoch_loss += loss.item() * label.size(0)
            logits = out["logits_main"] + model.lambda_asym * out["logits_asym"]
            n_correct += (logits.argmax(-1) == label).sum().item()
            n_total   += label.size(0)

        if not step_scheduler:
            scheduler.step()

        train_loss = epoch_loss / max(n_total, 1)
        train_acc  = n_correct / max(n_total, 1)
        val_acc, val_f1     = _evaluate(model, val_loader, device)
        test_acc, test_f1   = _evaluate(model, test_loader, device)

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            best_test_f1  = test_f1

        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"  epoch {epoch+1:>2}/{fc['epochs']}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  "
            f"lr={lr_now:.2e}",
            flush=True,
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_acc": val_acc, "val_f1": val_f1,
                "test_acc": test_acc, "test_f1": test_f1,
            })

    print(f"\n{'='*60}")
    print(f"All-Subject Cross-Session Results")
    print(f"{'='*60}")
    print(f"  Best val acc:  {best_val_acc:.4f}")
    print(f"  Best test acc: {best_test_acc:.4f}  (at best val)")
    print(f"  Best test F1:  {best_test_f1:.4f}")
    print(f"  Final test acc: {test_acc:.4f}")
    print(f"  Final test F1:  {test_f1:.4f}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.log({
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "best_test_f1": best_test_f1,
        })
        wandb.finish()


# ── Per-subject CV loops ─────────────────────────────────────────────────────

def run_cv(cfg, mode, wandb_project, folds=None):
    fc = cfg["finetune"]
    dc = cfg["data"]
    device = torch.device(fc["device"] if torch.cuda.is_available() else "cpu")
    all_subjects = dc.get("subjects") or list(range(1, 16))
    # ``folds`` picks which subjects are held out as TEST.  The train/val pool
    # is always ``all_subjects``, so sharding folds across GPUs does not change
    # any fold's training data.
    fold_subjects = folds or all_subjects
    unknown = [s for s in fold_subjects if s not in all_subjects]
    assert not unknown, f"fold subjects not in subject pool: {unknown}"

    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config=cfg, reinit=True)

    ds_kwargs = dict(
        root=dc["root"],
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
    for subj in fold_subjects:
        t0 = time.time()

        if mode == "loso":
            pool = [s for s in all_subjects if s != subj]
            n_val = fc.get("n_val_subjects", 2)
            if n_val > 0 and len(pool) > n_val:
                # Deterministic rotation: validation subjects follow the test
                # subject in cyclic order, so every subject serves as
                # validation equally often and no RNG is involved.
                i = all_subjects.index(subj)
                val_subjects = [
                    all_subjects[(i + k) % len(all_subjects)]
                    for k in range(1, n_val + 1)
                ]
                train_subjects = [s for s in pool if s not in val_subjects]
            else:
                val_subjects, train_subjects = [], pool

            print(f"  train={train_subjects}  val={val_subjects}  test=[{subj}]",
                  flush=True)
            train_ds = SEEDRawDataset(subjects=train_subjects, sessions=[1, 2, 3], **ds_kwargs)
            test_ds  = SEEDRawDataset(subjects=[subj],         sessions=[1, 2, 3], **ds_kwargs)
            val_ds   = (SEEDRawDataset(subjects=val_subjects, sessions=[1, 2, 3], **ds_kwargs)
                        if val_subjects else None)
            m = _train_and_eval(cfg, train_ds, test_ds, device, val_ds=val_ds)
            if "selected_epoch" in m:
                print(f"  -> selected epoch {m['selected_epoch']} "
                      f"(val_f1={m['val_f1']:.4f}); last-epoch test "
                      f"acc={m['final_accuracy']:.4f}", flush=True)

        elif mode == "cross_session":
            # LaBraM official protocol: session 1 train, session 2 val, session 3 test
            # But per-subject, all subjects together
            train_ds = SEEDRawDataset(subjects=[subj], sessions=[1], **ds_kwargs)
            test_ds  = SEEDRawDataset(subjects=[subj], sessions=[3], **ds_kwargs)
            m = _train_and_eval(cfg, train_ds, test_ds, device)

        else:  # subject_dependent (within-session)
            sd_train = fc.get("sd_train_per_session", SD_TRAIN_CLIPS)
            session_accs, session_f1s = [], []
            for sess in [1, 2, 3]:
                sess_ds = SEEDRawDataset(subjects=[subj], sessions=[sess], **ds_kwargs)
                # Split on the film-clip index, not on load order
                train_idx, test_idx = sess_ds.sd_split(sd_train)
                if not train_idx or not test_idx:
                    continue
                sess_m = _train_and_eval(cfg, Subset(sess_ds, train_idx), Subset(sess_ds, test_idx), device)
                session_accs.append(sess_m["accuracy"])
                session_f1s.append(sess_m["f1_macro"])
                print(f"  [S{subj}/Sess{sess}] acc={sess_m['accuracy']:.4f} F1={sess_m['f1_macro']:.4f} best={sess_m['best_accuracy']:.4f}")
            m = {
                "accuracy": float(np.mean(session_accs)) if session_accs else 0.0,
                "f1_macro": float(np.mean(session_f1s)) if session_f1s else 0.0,
                "best_accuracy": 0.0,
            }

        dt = time.time() - t0
        results["accuracy"].append(m["accuracy"])
        results["f1_macro"].append(m["f1_macro"])
        print(f"[Subject {subj:>2}]  acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  ({dt:.1f}s)")
        if use_wandb:
            wandb.log({"fold/subject": subj, "fold/accuracy": m["accuracy"], "fold/f1_macro": m["f1_macro"]})

    acc_arr, f1_arr = np.array(results["accuracy"]), np.array(results["f1_macro"])
    print(f"\n{'='*60}\n{mode} Results\n{'='*60}")
    print(f"  Accuracy : {acc_arr.mean():.4f} ± {acc_arr.std():.4f}")
    print(f"  Macro-F1 : {f1_arr.mean():.4f} ± {f1_arr.std():.4f}")
    print(f"{'='*60}")
    for i, s in enumerate(fold_subjects):
        print(f"  Subject {s:>2}: acc={acc_arr[i]:.4f}  F1={f1_arr[i]:.4f}")

    if use_wandb:
        wandb.log({"summary/acc_mean": acc_arr.mean(), "summary/acc_std": acc_arr.std(),
                    "summary/f1_mean": f1_arr.mean(), "summary/f1_std": f1_arr.std()})
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="LaBraM 62ch + Hemisphere Aux")
    parser.add_argument("--config", type=str, default="configs/seed_labram.yaml")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--eval_mode", type=str, default="all_subject_cross_session",
                        choices=["loso", "cross_session", "subject_dependent", "both",
                                 "all_subject_cross_session"])
    parser.add_argument("--folds", type=str, default=None,
                        help="comma-separated TEST subjects to run, e.g. 1,2,3. "
                             "Train/val pool stays the full subject list, so "
                             "folds can be sharded across GPUs.")
    args = parser.parse_args()
    folds = ([int(x) for x in args.folds.split(",")] if args.folds else None)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.eval_mode == "all_subject_cross_session":
        print(f"\n{'#'*60}\n#  LaBraM 62ch: All-Subject Cross-Session (official protocol)\n{'#'*60}")
        run_all_subject_cross_session(cfg, args.wandb_project)
    else:
        modes = [args.eval_mode] if args.eval_mode != "both" else ["cross_session", "subject_dependent"]
        for mode in modes:
            print(f"\n{'#'*60}\n#  LaBraM 62ch + HemiAux: {mode}"
                  f"{'  folds=' + str(folds) if folds else ''}\n{'#'*60}")
            run_cv(cfg, mode, args.wandb_project, folds=folds)


if __name__ == "__main__":
    main()
