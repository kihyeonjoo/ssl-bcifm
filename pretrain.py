"""
Pretraining loop: ℒ_total = ℒ_main + λ(t)·ℒ_aux
=================================================

- ℒ_main : Cross-Hemisphere Masked Band Prediction
- ℒ_aux  : Temporal Delta Asymmetry Prediction
- λ(t)   : linear warm-up from λ_init → λ_final over warmup epochs
- Optimizer : AdamW
- Scheduler : CosineAnnealingLR
- Logging   : Weights & Biases

Usage
-----
    python pretrain.py --config configs/seed.yaml [--wandb_project ssl-bcifm]
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import yaml

from data.seed_dataset import SEEDDataset
from data.temporal_pair_dataset import TemporalPairDataset
from models.encoder import DualStreamEncoder
from tasks.cross_hemisphere import CrossHemisphereMaskedPrediction
from tasks.temporal_delta_asymmetry import TemporalDeltaAsymmetry


# ── λ warm-up schedule ──────────────────────────────────────────────────────

def lambda_schedule(epoch: int, warmup_epochs: int, init: float, final: float) -> float:
    """Linear warm-up from *init* to *final* over *warmup_epochs*, then constant."""
    if epoch >= warmup_epochs:
        return final
    return init + (final - init) * epoch / warmup_epochs


# ── Build components from config ─────────────────────────────────────────────

def build_from_config(cfg: dict) -> dict:
    """Instantiate encoder, tasks, dataset, dataloader, optimizer, scheduler."""
    mc = cfg["model"]
    tc = cfg["training"]
    dc = cfg["data"]
    device = torch.device(tc["device"] if torch.cuda.is_available() else "cpu")

    # ── Encoder E_φ (shared) ──────────────────────────────────────────────
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

    # ── Main task ─────────────────────────────────────────────────────────
    main_task = CrossHemisphereMaskedPrediction(
        encoder=encoder,
        n_channels=mc["n_channels_per_hemi"],
        n_bands=mc["n_bands"],
        d_model=mc["d_model"],
        dec_heads=mc["n_heads"],
        dec_layers=2,
        mask_sigma=tc.get("mask_sigma", 0.3),
        dropout=mc["dropout"],
    )

    # ── Auxiliary task ────────────────────────────────────────────────────
    aux_task = TemporalDeltaAsymmetry(
        encoder=encoder,  # same instance — shared weights
        d_model=mc["d_model"],
        lambda_cos=tc["lambda_cos"],
    )

    # Wrap into a single nn.Module so .parameters() covers everything
    model = PretrainModel(main_task, aux_task).to(device)

    # ── Data ──────────────────────────────────────────────────────────────
    base_dataset = SEEDDataset(
        root=dc["root"],
        subjects=dc.get("subjects"),
        sessions=dc.get("sessions"),
        segment_length=dc["segment_length"],
        step=dc["step"],
        n_fft=dc["n_fft"],
        hop_length=dc["hop_length"],
    )
    pair_dataset = TemporalPairDataset(base_dataset)
    dataloader = DataLoader(
        pair_dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"], eta_min=1e-6)

    return {
        "model": model,
        "dataloader": dataloader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
    }


# ── Thin wrapper to unify parameters ─────────────────────────────────────────

class PretrainModel(nn.Module):
    """Groups main + aux task modules so optimizer.parameters() covers both.

    The encoder is shared — it lives inside main_task.encoder and aux_task
    holds a reference to the *same* object, so no parameter duplication.
    """

    def __init__(
        self,
        main_task: CrossHemisphereMaskedPrediction,
        aux_task: TemporalDeltaAsymmetry,
    ) -> None:
        super().__init__()
        self.main_task = main_task
        # Register only the predictor head (encoder already registered via main_task)
        self.aux_predictor = aux_task.predictor
        self._aux_task = aux_task  # keep reference but don't re-register encoder

    def forward_main(self, left: torch.Tensor, right: torch.Tensor) -> dict:
        return self.main_task(left, right)

    def forward_aux(
        self, left_t1: torch.Tensor, right_t1: torch.Tensor,
        left_t2: torch.Tensor, right_t2: torch.Tensor,
    ) -> dict:
        return self._aux_task(left_t1, right_t1, left_t2, right_t2)


# ── Training loop ────────────────────────────────────────────────────────────

def train(cfg: dict, wandb_project: str | None = None) -> None:
    tc = cfg["training"]
    comps = build_from_config(cfg)
    model      = comps["model"]
    dataloader = comps["dataloader"]
    optimizer  = comps["optimizer"]
    scheduler  = comps["scheduler"]
    device     = comps["device"]

    epochs          = tc["epochs"]
    log_every       = tc.get("log_every", 10)
    save_every      = tc.get("save_every", 10)
    ckpt_dir        = tc.get("checkpoint_dir", "checkpoints")
    warmup_epochs   = tc["lambda_warmup_epochs"]
    lambda_init     = tc["lambda_aux_init"]
    lambda_final    = tc["lambda_aux"]

    os.makedirs(ckpt_dir, exist_ok=True)

    # ── wandb ─────────────────────────────────────────────────────────────
    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config=cfg)
        wandb.watch(model, log="gradients", log_freq=log_every * 5)

    global_step = 0

    for epoch in range(epochs):
        model.train()
        lam = lambda_schedule(epoch, warmup_epochs, lambda_init, lambda_final)

        epoch_metrics = {
            "loss_main": 0.0, "loss_aux": 0.0, "loss_total": 0.0,
            "aux_mse": 0.0, "aux_cos": 0.0,
        }
        n_batches = 0
        t0 = time.time()

        for batch in dataloader:
            left_t1  = batch["left_t1"].to(device)
            right_t1 = batch["right_t1"].to(device)
            left_t2  = batch["left_t2"].to(device)
            right_t2 = batch["right_t2"].to(device)

            # ── ℒ_main: use t₁ pair ──────────────────────────────────────
            main_out = model.forward_main(left_t1, right_t1)
            loss_main = main_out["loss"]

            # ── ℒ_aux: use both t₁ and t₂ ────────────────────────────────
            aux_out = model.forward_aux(left_t1, right_t1, left_t2, right_t2)
            loss_aux = aux_out["loss"]

            # ── ℒ_total ──────────────────────────────────────────────────
            loss_total = loss_main + lam * loss_aux

            optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ── Accumulate ────────────────────────────────────────────────
            epoch_metrics["loss_main"]  += loss_main.item()
            epoch_metrics["loss_aux"]   += loss_aux.item()
            epoch_metrics["loss_total"] += loss_total.item()
            epoch_metrics["aux_mse"]    += aux_out["loss_mse"].item()
            epoch_metrics["aux_cos"]    += aux_out["loss_cos"].item()
            n_batches += 1
            global_step += 1

            # ── Step-level logging ────────────────────────────────────────
            if use_wandb and global_step % log_every == 0:
                wandb.log({
                    "step/loss_main":  loss_main.item(),
                    "step/loss_aux":   loss_aux.item(),
                    "step/loss_total": loss_total.item(),
                    "step/aux_mse":    aux_out["loss_mse"].item(),
                    "step/aux_cos":    aux_out["loss_cos"].item(),
                    "step/lambda_aux": lam,
                    "step/lr":         optimizer.param_groups[0]["lr"],
                }, step=global_step)

        # ── Epoch-level ──────────────────────────────────────────────────
        scheduler.step()
        dt = time.time() - t0

        avg = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
        print(
            f"[epoch {epoch+1:>3}/{epochs}]  "
            f"main={avg['loss_main']:.4f}  aux={avg['loss_aux']:.4f}  "
            f"total={avg['loss_total']:.4f}  λ={lam:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"({dt:.1f}s)"
        )

        if use_wandb:
            wandb.log({
                "epoch/loss_main":  avg["loss_main"],
                "epoch/loss_aux":   avg["loss_aux"],
                "epoch/loss_total": avg["loss_total"],
                "epoch/aux_mse":    avg["aux_mse"],
                "epoch/aux_cos":    avg["aux_cos"],
                "epoch/lambda_aux": lam,
                "epoch/lr":         optimizer.param_groups[0]["lr"],
                "epoch/epoch":      epoch + 1,
                "epoch/time_s":     dt,
            }, step=global_step)

        # ── Checkpoint ────────────────────────────────────────────────────
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"pretrain_epoch{epoch+1:03d}.pt")
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"  → saved {ckpt_path}")

    if use_wandb:
        wandb.finish()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SSL-BCIFM Pretraining")
    parser.add_argument("--config", type=str, default="configs/seed.yaml")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project name (omit to disable logging)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, wandb_project=args.wandb_project)


if __name__ == "__main__":
    main()
