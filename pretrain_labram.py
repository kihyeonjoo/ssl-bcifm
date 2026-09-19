"""
Pretraining with a frozen LaBraM backbone + AsymmetryAdapter.

ℒ_total = ℒ_main + λ(t) · ℒ_aux

  ℒ_main : cross-hemisphere *channel* masked prediction on LaBraM features
  ℒ_aux  : temporal delta asymmetry (same as the STFT pipeline)
  λ(t)   : linear warmup 0.1 → 0.5 over 20 epochs

Only the AsymmetryAdapter + decoder + aux predictor are trained.
LaBraM stays frozen throughout.

Usage
-----
    python pretrain_labram.py --config configs/seed_labram.yaml \\
           [--wandb_project ssl-bcifm-labram]
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

from data.seed_raw_dataset import SEEDRawDataset, SEEDRawPairDataset, SEED_CH_NAMES
from models.labram_wrapper import LaBraMWrapper
from models.asymmetry_adapter import AsymmetryAdapter
from tasks.cross_hemisphere_channel import CrossHemisphereChannelMaskedPrediction
from tasks.temporal_delta_adapter import TemporalDeltaAsymmetryAdapter


# ── λ warm-up schedule ──────────────────────────────────────────────────────

def lambda_schedule(epoch: int, warmup_epochs: int, init: float, final: float) -> float:
    if epoch >= warmup_epochs:
        return final
    return init + (final - init) * epoch / warmup_epochs


# ── Thin wrapper to unify trainable parameters ──────────────────────────────

class LaBraMPretrainModel(nn.Module):
    """Holds the shared adapter + main/aux task heads as trainable modules.

    LaBraM lives inside ``adapter.labram`` and is frozen — it is still
    registered as a submodule so ``.to(device)`` moves it, but its
    parameters don't appear in ``parameters()`` because requires_grad=False.
    """

    def __init__(
        self,
        adapter: AsymmetryAdapter,
        main_task: CrossHemisphereChannelMaskedPrediction,
        aux_task: TemporalDeltaAsymmetryAdapter,
    ) -> None:
        super().__init__()
        self.adapter   = adapter            # contains frozen LaBraM + shared HemisphereAdapter
        self.main_head = main_task.decoder  # decoder is trainable
        self.aux_head  = aux_task.predictor # predictor is trainable
        self._main_task = main_task
        self._aux_task  = aux_task

    def forward_main(self, eeg: torch.Tensor) -> dict:
        return self._main_task(eeg)

    def forward_aux(self, eeg_t1: torch.Tensor, eeg_t2: torch.Tensor) -> dict:
        return self._aux_task(eeg_t1, eeg_t2)


# ── Build components from config ────────────────────────────────────────────

def build(cfg: dict):
    mc = cfg["model"]
    tc = cfg["training"]
    dc = cfg["data"]
    lc = cfg["labram"]
    device = torch.device(tc["device"] if torch.cuda.is_available() else "cpu")

    # LaBraM (frozen)
    labram = LaBraMWrapper(
        labram_repo=lc["repo_path"],
        ckpt_path=lc["ckpt_path"],
        ch_names=SEED_CH_NAMES,
        model_name=lc.get("model_name", "labram_base_patch200_200"),
        freeze=True,
    )

    # Adapter
    adapter = AsymmetryAdapter(
        labram=labram,
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"],
    )

    # Main task
    main_task = CrossHemisphereChannelMaskedPrediction(
        adapter=adapter,
        d_model=mc["d_model"],
        out_dim=labram.embed_dim,
        n_masked_ch=mc.get("n_masked_ch", 5),
        n_patches=dc["segment_length"] // dc["patch_size"],
        dec_heads=mc["n_heads"],
        dec_layers=2,
        dropout=mc["dropout"],
    )

    # Aux task
    aux_task = TemporalDeltaAsymmetryAdapter(
        adapter=adapter,
        d_model=mc["d_model"],
        lambda_cos=tc["lambda_cos"],
    )

    model = LaBraMPretrainModel(adapter, main_task, aux_task).to(device)

    # ── Data ────────────────────────────────────────────────────────────
    base_dataset = SEEDRawDataset(
        root=dc["root"],
        subjects=dc.get("subjects"),
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
    pair_dataset = SEEDRawPairDataset(base_dataset)

    dataloader = DataLoader(
        pair_dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer & scheduler ──────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"], eta_min=1e-6)

    n_train = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {n_train:,} / {n_total:,} "
        f"({100 * n_train / n_total:.2f}%)"
    )

    return {
        "model": model,
        "dataloader": dataloader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
    }


# ── Training loop ───────────────────────────────────────────────────────────

def train(cfg: dict, wandb_project: str | None = None) -> None:
    tc = cfg["training"]
    comps = build(cfg)
    model      = comps["model"]
    dataloader = comps["dataloader"]
    optimizer  = comps["optimizer"]
    scheduler  = comps["scheduler"]
    device     = comps["device"]

    epochs        = tc["epochs"]
    log_every     = tc.get("log_every", 10)
    save_every    = tc.get("save_every", 10)
    ckpt_dir      = tc.get("checkpoint_dir", "checkpoints_labram")
    warmup_epochs = tc["lambda_warmup_epochs"]
    lambda_init   = tc["lambda_aux_init"]
    lambda_final  = tc["lambda_aux"]

    os.makedirs(ckpt_dir, exist_ok=True)

    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config=cfg)

    global_step = 0

    for epoch in range(epochs):
        model.train()
        lam = lambda_schedule(epoch, warmup_epochs, lambda_init, lambda_final)

        metrics = {
            "loss_main": 0.0, "loss_aux": 0.0, "loss_total": 0.0,
            "aux_mse":  0.0, "aux_cos": 0.0,
        }
        n_batches = 0
        t0 = time.time()

        for batch in dataloader:
            eeg_t1 = batch["eeg_t1"].to(device)
            eeg_t2 = batch["eeg_t2"].to(device)

            main_out = model.forward_main(eeg_t1)
            loss_main = main_out["loss"]

            aux_out = model.forward_aux(eeg_t1, eeg_t2)
            loss_aux = aux_out["loss"]

            loss_total = loss_main + lam * loss_aux

            optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            metrics["loss_main"]  += loss_main.item()
            metrics["loss_aux"]   += loss_aux.item()
            metrics["loss_total"] += loss_total.item()
            metrics["aux_mse"]    += aux_out["loss_mse"].item()
            metrics["aux_cos"]    += aux_out["loss_cos"].item()
            n_batches += 1
            global_step += 1

            if use_wandb and global_step % log_every == 0:
                wandb.log({
                    "step/loss_main":  loss_main.item(),
                    "step/loss_aux":   loss_aux.item(),
                    "step/loss_total": loss_total.item(),
                    "step/lambda_aux": lam,
                    "step/lr":         optimizer.param_groups[0]["lr"],
                }, step=global_step)

        scheduler.step()
        dt = time.time() - t0
        avg = {k: v / max(n_batches, 1) for k, v in metrics.items()}
        print(
            f"[epoch {epoch+1:>3}/{epochs}]  "
            f"main={avg['loss_main']:.4f}  aux={avg['loss_aux']:.4f}  "
            f"total={avg['loss_total']:.4f}  λ={lam:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  ({dt:.1f}s)",
            flush=True,
        )

        if use_wandb:
            wandb.log({
                "epoch/loss_main":  avg["loss_main"],
                "epoch/loss_aux":   avg["loss_aux"],
                "epoch/loss_total": avg["loss_total"],
                "epoch/lambda_aux": lam,
                "epoch/lr":         optimizer.param_groups[0]["lr"],
                "epoch/epoch":      epoch + 1,
                "epoch/time_s":     dt,
            }, step=global_step)

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"pretrain_epoch{epoch+1:03d}.pt")
            # Save only the trainable parts — LaBraM weights stay in their own file
            torch.save({
                "epoch": epoch + 1,
                "adapter_state_dict":  model.adapter.adapter.state_dict(),
                "main_decoder_state_dict": model.main_head.state_dict(),
                "aux_predictor_state_dict": model.aux_head.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"  → saved {ckpt_path}")

    if use_wandb:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="SSL-BCIFM LaBraM Pretraining")
    parser.add_argument("--config", type=str, default="configs/seed_labram.yaml")
    parser.add_argument("--wandb_project", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, wandb_project=args.wandb_project)


if __name__ == "__main__":
    main()
