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
import random
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
from data.seedv_raw_dataset import SEEDVRawDataset
from data.seediv_raw_dataset import SEEDIVRawDataset
from data.deap_raw_dataset import DEAPRawDataset

# 데이터셋은 설정의 ``data.loader`` 로 고른다.  SEED-V 는 클래스가 5개이고 피험자가
# 16명이지만 로더 인터페이스가 같으므로, 여기만 바꾸면 LOSO 경로가 그대로 돈다.
_LOADERS = {"SEEDRawDataset": SEEDRawDataset, "SEEDVRawDataset": SEEDVRawDataset,
            "SEEDIVRawDataset": SEEDIVRawDataset, "DEAPRawDataset": DEAPRawDataset}
from data.preprocessing import LEFT_IDX, RIGHT_IDX, LEFT_CH, RIGHT_CH


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Seed every RNG this pipeline draws from.

    Without this the run-to-run spread swamps the effects we are trying to
    measure: the diag-EA gain over no-EA is +0.064 while the fold-to-fold std
    is 0.068, so a single unseeded run per fold cannot separate a real change
    from noise.

    This does NOT make the run bit-exact.  cuDNN picks algorithms by
    benchmarking and several CUDA kernels reduce in nondeterministic order, so
    two seeded runs still differ slightly.  Turning that off costs speed, so it
    is left to the caller (see ``--cudnn_deterministic``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Re-seed a DataLoader worker from the base seed torch gives it.

    Workers fork with their own torch seed but numpy's and python's global
    RNGs are NOT reseeded, so every worker would otherwise draw the same
    numbers.  Nothing in this dataset samples randomly today, but
    ``refit_alignment_limited(how='random')`` does, and a future sampler will.
    """
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws)
    random.seed(ws)


def _loader_generator(seed: int) -> torch.Generator:
    """Generator that fixes DataLoader's shuffling order."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


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
        lora_targets: tuple = ("fc1", "fc2"),
        ch_names: "list | None" = None,
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
        # E6: set by attach_subject_discriminator(); adv_lambda is the GRL
        # strength and is updated per epoch by the training loop.
        self.subject_disc = None
        self.adv_lambda = 0.0
        self.use_lora = use_lora

        if use_lora:
            from models.lora import inject_lora, count_trainable_parameters
            for p in self.labram.parameters():
                p.requires_grad = False
            # LaBraM blocks expose qkv / proj (attention) and fc1 / fc2 (MLP).
            # `qkv` cannot be wrapped: Attention.forward reads
            # ``self.qkv.weight`` directly (F.linear, to splice in its
            # separate q_bias/v_bias) instead of calling the module, so a
            # LoRALinear there raises AttributeError — and giving the wrapper
            # a `.weight` would be worse, silently bypassing the LoRA path.
            # `proj` is a normal module call and wraps fine.
            if "qkv" in lora_targets:
                raise ValueError(
                    "lora_targets cannot include 'qkv': LaBraM's attention "
                    "bypasses the module and reads qkv.weight directly. "
                    "Use ('proj', 'fc1', 'fc2') to adapt attention output "
                    "plus the MLP."
                )
            n_wrapped = inject_lora(
                self.labram, target_names=tuple(lora_targets),
                r=lora_r, alpha=lora_alpha,
            )
            print(f"[LaBraM] LoRA targets: {tuple(lora_targets)}")
            trainable, total = count_trainable_parameters(self.labram)
            print(f"[LaBraM] LoRA: {n_wrapped} layers (r={lora_r}).  "
                  f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        else:
            # Full fine-tune — all parameters trainable
            trainable = sum(p.numel() for p in self.labram.parameters())
            print(f"[LaBraM] Full fine-tune: {trainable:,} params all trainable")

        # input_chans for LaBraM's channel embedding.  Default: the 62-ch SEED
        # layout (unchanged for SEED / SEED-V / SEED-IV).  Other datasets pass
        # their own 10-20 names (DEAP: 32 ch, 2026-10-05).
        ch_names = list(ch_names) if ch_names is not None else list(SEED_CH_NAMES)
        self.ch_names = ch_names
        self.input_chans = _get_input_chans(ch_names)

        # Hemisphere channel indices into this dataset's channel ordering.
        # For the SEED layout these are exactly LEFT_IDX / RIGHT_IDX; for other
        # layouts they are the channels that appear in LEFT_CH / RIGHT_CH.
        if ch_names == list(SEED_CH_NAMES):
            left, right = LEFT_IDX, RIGHT_IDX
        else:
            left = [i for i, c in enumerate(ch_names) if c in LEFT_CH]
            right = [i for i, c in enumerate(ch_names) if c in RIGHT_CH]
            if not left or not right:
                raise ValueError(f"no left/right hemisphere channels in {ch_names}")
        self.register_buffer(
            "left_idx", torch.tensor(left, dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "right_idx", torch.tensor(right, dtype=torch.long), persistent=False,
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

    def forward(self, eeg: torch.Tensor, center: "torch.Tensor | None" = None):
        """
        Parameters
        ----------
        eeg    : (B, 62, n_patches, 200)
        center : (B, embed_dim) or None — per-sample vector subtracted from the
                 CLS token before the classification head.  It carries the
                 recording's own mean, estimated without labels, so the head
                 sees each domain at the same location.  ``None`` leaves the
                 forward pass bit-identical to the uncentered pipeline.

        Returns
        -------
        dict with keys: logits_main, logits_asym, z_L, z_R, z
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

        # Main classification from CLS.  The asymmetry branch below is left on
        # the raw token: lambda_asym is 0 in every current config, and changing
        # it would alter a path that is not being tested.
        head_in = cls_token if center is None else cls_token - center
        logits_main = self.main_head(head_in)

        # Hemisphere features from patch tokens
        patch_tokens = patch_tokens.reshape(B, C, P, self.embed_dim)
        # Average over patches within each channel → (B, C, d)
        channel_feats = patch_tokens.mean(dim=2)

        z_L = channel_feats[:, self.left_idx].mean(dim=1)   # (B, d)
        z_R = channel_feats[:, self.right_idx].mean(dim=1)  # (B, d)

        # 4-way asymmetry fusion
        z_fused = torch.cat([z_L, z_R, z_L - z_R, z_L * z_R], dim=-1)
        logits_asym = self.asym_head(z_fused)

        out = {
            "logits_main": logits_main,
            "logits_asym": logits_asym,
            "z_L": z_L,
            "z_R": z_R,
            "z": cls_token,          # the shared representation E6 attacks
            "z_centered": head_in,   # what the head actually read
        }
        if self.subject_disc is not None:
            out["logits_subj"] = self.subject_disc(cls_token, self.adv_lambda)
        return out

    def compute_loss(self, out: dict, label: torch.Tensor, criterion,
                     subject_idx: "torch.Tensor | None" = None):
        loss_main = criterion(out["logits_main"], label)
        loss_asym = criterion(out["logits_asym"], label)
        loss = loss_main + self.lambda_asym * loss_asym
        if "logits_subj" in out and subject_idx is not None:
            # The reversal lives in the layer, so this term is ADDED: the
            # discriminator minimises it while the backbone maximises it.
            loss = loss + criterion(out["logits_subj"], subject_idx)
        return loss, loss_main, loss_asym

    def attach_subject_discriminator(self, n_subjects: int, dropout: float = 0.1):
        from models.adversarial import SubjectDiscriminator
        self.subject_disc = SubjectDiscriminator(
            self.embed_dim, n_subjects, dropout=dropout,
        ).to(next(self.parameters()).device)
        return self.subject_disc


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


def _evaluate(model, loader, device, amp: bool = False, return_logits: bool = False,
              center_table=None):
    model.eval()
    preds_main, preds_asym, labels = [], [], []
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            ctr = None if center_table is None else lookup_center(center_table, batch, device)
            with _amp_ctx(device, amp):
                out = model(eeg, center=ctr)
            out = {k: v.float() for k, v in out.items()
                   if isinstance(v, torch.Tensor)}
            # Ensemble: average main + asym logits
            logits = out["logits_main"] + model.lambda_asym * out["logits_asym"]
            preds_main.append(logits.argmax(-1).cpu())
            labels.append(batch["label"])
            if return_logits:
                all_logits.append(logits.cpu())
    y_pred = torch.cat(preds_main).numpy()
    y_true = torch.cat(labels).numpy()
    acc, f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")
    if return_logits:
        return acc, f1, torch.cat(all_logits).numpy()
    return acc, f1


# ── domain centering ────────────────────────────────────────────────────────

MAX_SUBJ, MAX_SESS = 20, 5          # table bounds, SEED uses 15 x 3


@torch.no_grad()
def domain_mean_table(model, dataset, device, batch_size, amp=True):
    """Mean CLS token per (subject, session), computed WITHOUT labels.

    Recomputed from a full forward pass at the start of every epoch rather than
    tracked as an EMA.  Two reasons.  The test-time procedure is exactly this —
    one pass over that session with the current weights — so a full pass makes
    training and evaluation estimate the same quantity the same way, while an
    EMA would always lag the model it is centering.  And a full pass depends on
    no batch order, so the run stays bit-reproducible; an EMA would make the
    mean a function of the shuffle.

    The cost is one forward-only pass per epoch, reported in the log.

    -> FloatTensor (MAX_SUBJ+1, MAX_SESS+1, embed_dim) on ``device``
    """
    model.eval()
    d = model.embed_dim
    tot = torch.zeros(MAX_SUBJ + 1, MAX_SESS + 1, d, device=device)
    cnt = torch.zeros(MAX_SUBJ + 1, MAX_SESS + 1, 1, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
    for b in loader:
        eeg = b["eeg"].to(device, non_blocking=True)
        with _amp_ctx(device, amp):
            allt = model.labram.forward_features(
                eeg, input_chans=model.input_chans, return_all_tokens=True)
        z = allt[:, 0].float()
        sj = b["subject"].to(device).long()
        se = b["session"].to(device).long()
        tot.index_put_((sj, se), z, accumulate=True)
        cnt.index_put_((sj, se), torch.ones_like(z[:, :1]), accumulate=True)
    return tot / cnt.clamp(min=1.0)


def lookup_center(table, batch, device):
    """Per-sample centring vector for one batch, from a domain-mean table.

    The table comes out of a ``@torch.no_grad()`` function, so it carries no
    grad_fn and the subtraction treats it as a constant — the centring vector
    is never differentiated through.
    """
    return table[batch["subject"].to(device).long(),
                 batch["session"].to(device).long()]


def relative_offsets(table, mask):
    """Each domain's mean minus the mean of all domains.

    The quantity that matters for staleness is not how far the domain means
    travel together — a shift shared by every domain cancels in the
    subtraction the head sees — but how their positions move relative to one
    another.  This strips the common part.
    """
    mus = table[mask]
    return mus - mus.mean(0, keepdim=True)


def sample_indices_per_domain(dataset, n_per_domain, seed=0):
    """A fixed sample of window indices per (subject, session).

    Fixed, because the staleness measurement compares means at three points in
    an epoch: if the windows differed between them, sampling noise would be
    read as model drift.
    """
    rng = np.random.default_rng(seed)
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    rows = dataset.indices if isinstance(dataset, Subset) else range(len(base))
    by = defaultdict(list)
    for i in rows:
        s, se, _c = base._seg_meta[i]
        by[(s, se)].append(i)
    out = []
    for k in sorted(by):
        idx = by[k]
        take = rng.choice(len(idx), size=min(n_per_domain, len(idx)), replace=False)
        out.extend(idx[t] for t in take)
    return Subset(base, sorted(out))


def _clip_level_metrics(logits, y, session, clip, n_classes):
    """Average the softmax over each film clip and score one decision per clip.

    SEED's label is constant for a whole ~4 min clip, so the ~56 windows inside
    one clip are near-duplicates of each other.  A clip-level score has
    15 clips x 3 sessions = 45 decisions per subject: it moves in ~2.2% steps
    and its fold-to-fold variance is larger than the window-level number's.
    Report both, never one in place of the other.

    The softmax and balanced-accuracy definitions come from aggregate_logits so
    the number printed here and the number that script prints cannot drift.
    """
    from aggregate_logits import softmax, balanced

    P = softmax(np.asarray(logits, dtype=np.float64))
    session, clip, y = np.asarray(session), np.asarray(clip), np.asarray(y)
    cp, ct = [], []
    for key in sorted(set(zip(session.tolist(), clip.tolist()))):
        m = (session == key[0]) & (clip == key[1])
        cp.append(P[m].mean(0).argmax())
        ct.append(y[m][0])
    cp, ct = np.array(cp), np.array(ct)
    return {
        "clip_accuracy": float((cp == ct).mean()),
        "clip_f1_macro": float(f1_score(ct, cp, average="macro")),
        "clip_balanced": balanced(cp, ct, n_classes),
        "window_balanced": balanced(P.argmax(1), y, n_classes),
        "n_clips": int(len(ct)),
        # Raw decisions, so a caller that runs SEVERAL splits per subject
        # (the subject_dependent path) can pool them and compute macro-F1 and
        # balanced accuracy ONCE.  Those two are non-linear, so the mean of
        # per-split values is not the pooled value — and with one clip per
        # class in a split, per-split macro-F1 is nearly degenerate.
        # Underscored so the CSV row builder's explicit key list ignores them.
        "_clip_pred": cp, "_clip_true": ct,
        "_win_pred": P.argmax(1), "_win_true": np.asarray(y),
    }


def _train_and_eval(cfg, train_ds, test_ds, device, val_ds=None, fold_tag="",
                    seed=0):
    """Train once and report test metrics.

    When ``val_ds`` is given the reported numbers are the test metrics at the
    epoch with the best VALIDATION macro-F1 — the test set never influences
    model selection.  Without it, the last epoch is reported.

    ``seed`` fixes weight init, dropout, drop-path and the shuffling order, so
    repeating a fold at several seeds measures the run-to-run noise directly
    instead of leaving it confounded with whatever is being compared.
    """
    fc = cfg["finetune"]
    dc = cfg["data"]
    lc = cfg["labram"]

    # Seed BEFORE the model is built: head init and drop-path draw here.
    set_seed(seed)

    # CAFT (캘리브레이션 인지 파인튜닝, 2026-10-05): 같은 세션의 피험자 K 명 × 같은 (클립, 시점) M 곳으로 배치를 짠다.
    # 꺼져 있으면 (기본) 기존 로더와 bit 단위로 같다.
    caft = bool(fc.get("caft", False))
    caft_K = int(fc.get("caft_subjects_per_batch", 4))
    caft_M = int(fc.get("caft_windows_per_subject", 15))
    caft_align = float(fc.get("caft_align", 0.0))
    # ⓑ0 대조 (2026-10-06): 구조 배치는 그대로 두고 배치 안 중심화만 끈다 — 이득이 배치 구성이 아니라 평균 빼기에서
    # 오는지 가른다.  기본 (켬) 이면 ⓑ · ⓒ 와 같다.
    caft_center = bool(fc.get("caft_center", True))
    if caft:
        from caft import CAFTBatchSampler, in_batch_center, stim_align_loss
        if isinstance(train_ds, Subset):
            raise ValueError("CAFT 는 Subset 이 아닌 원본 데이터셋에서만 쓴다")
        caft_sampler = CAFTBatchSampler(train_ds, n_classes=fc["n_classes"], subjects_per_batch=caft_K,
                                        windows_per_subject=caft_M, seed=seed)
        train_loader = DataLoader(train_ds, batch_sampler=caft_sampler, num_workers=4, pin_memory=True,
                                  worker_init_fn=seed_worker)
        print(f"[CAFT] 배치 = 피험자 {caft_K} × 위치 {caft_M} = {caft_K * caft_M} 창, 정렬 손실 가중 {caft_align}, "
              f"배치 안 중심화 {'켬' if caft_center else '끔 (ⓑ0 대조)'}, epoch 당 {len(caft_sampler)} 배치", flush=True)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=fc["batch_size"],
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
            generator=_loader_generator(seed), worker_init_fn=seed_worker,
        )
    test_loader = DataLoader(
        test_ds, batch_size=fc["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker,
    )
    val_loader = None if val_ds is None else DataLoader(
        val_ds, batch_size=fc["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker,
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
        lora_targets=fc.get("lora_targets", ("fc1", "fc2")),
        ch_names=getattr(getattr(train_ds, "dataset", train_ds), "ch_names", None),   # Subset 이면 원본에서
    ).to(device)

    # ── E6: subject-adversarial head ─────────────────────────────────────
    # Registered over the TRAINING subjects only; the held-out subject never
    # reaches the discriminator, so this stays calibration-free.
    lambda_adv = float(fc.get("lambda_adv", 0.0))
    subj_index = None
    best_backbone = None
    best_adv_lambda = 0.0
    best_test_logits = None
    best_state = None
    if lambda_adv > 0:
        from models.adversarial import SubjectIndexer
        base = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        if isinstance(train_ds, Subset):
            train_subj = [base.subjects_of[i] for i in train_ds.indices]
        else:
            train_subj = list(base.subjects_of)
        subj_index = SubjectIndexer(train_subj)
        model.attach_subject_discriminator(len(subj_index), dropout=fc.get("dropout", 0.1))
        print(f"[E6] subject-adversarial: {len(subj_index)} train subjects "
              f"{subj_index.ids}  lambda_adv={lambda_adv}", flush=True)

    total_epochs = fc["epochs"]
    warmup_epochs = fc.get("warmup_epochs", 5)

    if use_lora:
        # LoRA mode: simple param groups
        lora_params = [p for p in model.labram.parameters() if p.requires_grad]
        head_params = list(model.main_head.parameters()) + list(model.asym_head.parameters())
        if model.subject_disc is not None:
            head_params += list(model.subject_disc.parameters())
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

    # ── domain centering (off by default; false must be bit-identical) ───
    dom_center = bool(fc.get("domain_center", False))
    center_diag = bool(fc.get("center_diag", False))
    n_probe = int(fc.get("center_diag_windows", 64))
    # Refresh the TRAIN table every K optimiser steps.  0 = only at epoch start.
    # ``center_refresh_windows`` is windows per domain; 0 means every window.
    refresh_k = int(fc.get("center_refresh_steps", 0))
    refresh_m = int(fc.get("center_refresh_windows", 0))
    refresh_ds = None
    # Recompute val/test tables just before scoring rather than at epoch start:
    # evaluation happens at the END of the epoch, so an epoch-start table is as
    # stale as the training one, and the offline comparison it is measured
    # against used a table from the very weights being scored.
    fresh_eval = bool(fc.get("center_fresh_eval", False))
    tbl_tr = tbl_va = tbl_te = None
    prev_tbl = prev_rel = probe_ds = probe_start = None
    stale_mid = float("nan")
    if dom_center:
        print("[center] domain centering ON — head input is z - mu(subject,session); "
              "mu is recomputed each epoch by a label-free forward pass", flush=True)
        if refresh_k > 0:
            refresh_ds = (train_ds if refresh_m <= 0
                          else sample_indices_per_domain(train_ds, refresh_m, seed=seed))
            print(f"[center] train table refreshed every {refresh_k} steps from "
                  f"{len(refresh_ds)} windows "
                  f"({'ALL' if refresh_m <= 0 else str(refresh_m) + '/domain'})",
                  flush=True)
        if fresh_eval:
            print("[center] val/test tables recomputed immediately before scoring",
                  flush=True)
        if center_diag:
            # n_probe <= 0 means "every training window".  A sampled probe is
            # itself noisy — a 64-window domain mean sits about 64% of |rel|
            # away from the full-domain mean — so a small sample cannot tell
            # real drift from its own sampling error.
            probe_ds = (train_ds if n_probe <= 0
                        else sample_indices_per_domain(train_ds, n_probe))
            print(f"[center] staleness probe: {len(probe_ds)} windows "
                  f"({'ALL' if n_probe <= 0 else str(n_probe) + '/domain'}), "
                  f"re-measured mid- and end-epoch", flush=True)

    for epoch in range(total_epochs):
        if dom_center and not caft:
            t_c = time.time()
            tbl_tr = domain_mean_table(model, train_ds, device, fc["batch_size"], amp)
            if not fresh_eval:
                tbl_te = domain_mean_table(model, test_ds, device, fc["batch_size"], amp)
                tbl_va = (domain_mean_table(model, val_ds, device, fc["batch_size"], amp)
                          if val_ds is not None else None)
            nz = tbl_tr.abs().sum(-1) > 0
            rel = relative_offsets(tbl_tr, nz)
            rel_size = float(rel.norm(dim=-1).mean())
            abs_shift = (float("nan") if prev_tbl is None else
                         float((tbl_tr[nz] - prev_tbl[nz]).norm(dim=-1).mean()))
            rel_shift = (float("nan") if prev_rel is None else
                         float((rel - prev_rel).norm(dim=-1).mean()))
            print(f"    [center] epoch {epoch+1}: {int(nz.sum())} domains  "
                  f"|mu| {float(tbl_tr[nz].norm(dim=-1).mean()):.3f}  "
                  f"|rel| {rel_size:.3f}  abs_shift {abs_shift:.3f}  "
                  f"rel_shift {rel_shift:.3f}  {time.time()-t_c:.0f}s", flush=True)
            prev_tbl, prev_rel = tbl_tr.clone(), rel.clone()
            if center_diag:
                # the SAME fixed windows at all three points, so any change is
                # model drift rather than sampling noise
                probe_start = relative_offsets(
                    domain_mean_table(model, probe_ds, device, fc["batch_size"], amp), nz)

        model.train()
        epoch_loss, n_correct, n_total = 0.0, 0, 0
        subj_correct = 0
        n_seen = 0

        if subj_index is not None:
            from models.adversarial import dann_lambda
            # Ramp the reversal in: at full strength from step 0 the
            # discriminator wins before the representation means anything.
            model.adv_lambda = dann_lambda(epoch, total_epochs, lambda_adv)

        for batch in train_loader:
            eeg   = batch["eeg"].to(device)
            label = batch["label"].to(device)
            subj_idx = (subj_index(batch["subject"]).to(device)
                        if subj_index is not None else None)

            if caft:
                # ① 배치 안 즉석 중심화 → head,  ② (선택) 같은 위치의 사람 간 정렬 손실
                with _amp_ctx(device, amp):
                    out = model(eeg)
                    zc = in_batch_center(out["z"], caft_K, caft_M) if caft_center else out["z"]
                    logits_c = model.main_head(zc)
                    loss = criterion(logits_c, label)
                    if caft_align > 0:
                        loss = loss + caft_align * stim_align_loss(zc, caft_K, caft_M)
                out = {**out, "logits_main": logits_c}       # 학습 정확도 집계용
            else:
                ctr = None if tbl_tr is None else lookup_center(tbl_tr, batch, device)
                with _amp_ctx(device, amp):
                    out = model(eeg, center=ctr)
                    loss, loss_m, loss_a = model.compute_loss(
                        out, label, criterion, subject_idx=subj_idx)
            if subj_idx is not None:
                subj_correct += (out["logits_subj"].argmax(-1) == subj_idx).sum().item()

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
            n_seen += 1
            if refresh_ds is not None and n_seen % refresh_k == 0:
                tbl_tr = domain_mean_table(model, refresh_ds, device,
                                           fc["batch_size"], amp)
                model.train()
            if probe_start is not None and n_seen == len(train_loader) // 2:
                stale_mid = float((relative_offsets(
                    domain_mean_table(model, probe_ds, device, fc["batch_size"], amp), nz)
                    - probe_start).norm(dim=-1).mean())
                model.train()

        if probe_start is not None:
            stale_end = float((relative_offsets(
                domain_mean_table(model, probe_ds, device, fc["batch_size"], amp), nz)
                - probe_start).norm(dim=-1).mean())
            print(f"    [stale] epoch {epoch+1}: |rel| {rel_size:.3f}   "
                  f"mid {stale_mid:.3f} ({100*stale_mid/max(rel_size,1e-9):.1f}%)   "
                  f"end {stale_end:.3f} ({100*stale_end/max(rel_size,1e-9):.1f}%)",
                  flush=True)
            model.train()

        if not step_scheduler:
            scheduler.step()
        train_loss = epoch_loss / max(n_total, 1)
        subj_acc = subj_correct / max(n_total, 1) if subj_index is not None else None
        train_acc  = n_correct / max(n_total, 1)

        lr_now = optimizer.param_groups[-1]["lr"]
        msg = (
            f"    epoch {epoch+1:>2}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"lr={lr_now:.2e}"
        )
        if subj_acc is not None:
            # subj_acc falling toward chance is the adversary working.
            msg += f"  adv_l={model.adv_lambda:.3f}  subj_acc={subj_acc:.4f}"

        if dom_center and fresh_eval:
            t_f = time.time()
            tbl_te = domain_mean_table(model, test_ds, device, fc["batch_size"], amp)
            tbl_va = (domain_mean_table(model, val_ds, device, fc["batch_size"], amp)
                      if val_ds is not None else None)
            if epoch == 0:
                print(f"    [center] fresh eval tables {time.time()-t_f:.1f}s", flush=True)

        if val_loader is not None:
            # Select on validation only; test is evaluated for logging but
            # never used to choose the reported epoch.
            val_acc, val_f1 = _evaluate(model, val_loader, device, amp,
                                        center_table=tbl_va)
            te_acc, te_f1, te_logits = _evaluate(model, test_loader, device, amp,
                                                 return_logits=True,
                                                 center_table=tbl_te)
            msg += f"  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
            msg += f"  test_acc={te_acc:.4f}  test_f1={te_f1:.4f}"
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                sel = (te_acc, te_f1, epoch + 1)
                best_test_logits = te_logits
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                if fc.get("save_backbone"):
                    # Snapshot HERE, not at the last epoch: the probe must
                    # audit the same model whose accuracy gets reported, and
                    # with an adversary the two epochs sit at different lambda.
                    best_backbone = {k: v.detach().cpu().clone()
                                     for k, v in model.labram.state_dict().items()}
                    best_adv_lambda = model.adv_lambda
                msg += "  *"
        elif eval_every:
            eval_acc, eval_f1 = _evaluate(model, test_loader, device, amp,
                                          center_table=tbl_te)
            if eval_acc > best_acc:
                best_acc = eval_acc
            msg += f"  test_acc={eval_acc:.4f}  test_f1={eval_f1:.4f}"

        if verbose:
            print(msg, flush=True)

    # Optionally keep the fine-tuned backbone so a frozen-feature probe can ask
    # whether the adversary really removed subject identity, or only fooled the
    # one discriminator it was trained against.
    # ── calibration-length sweep ──────────────────────────────────────────
    # EA as published needs the whole test recording.  Refit the test
    # subject's transform from only the first N seconds and re-score, to see
    # how short a calibration can be.  Scoring excludes the calibration
    # segments themselves.  Training is untouched; only evaluation repeats.
    calib = fc.get("calib_seconds")
    if calib:
        base_te = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
        if best_state is not None:
            model.load_state_dict(best_state)
        for how in fc.get("calib_modes", ["prefix"]):
            for sec in calib:
                keep, n_seg = base_te.refit_alignment_limited(
                    sec, how=how,
                    scale=dc.get("ea_scale", 0.2), trim=dc.get("ea_trim", 0.05),
                    eps=dc.get("ea_eps", 1e-6),
                )
                ld = DataLoader(Subset(base_te, keep), batch_size=fc["batch_size"],
                                shuffle=False, num_workers=4, pin_memory=True)
                a, f = _evaluate(model, ld, device, amp, center_table=tbl_te)
                print(f"    [calib] {how:6s} {sec:>5.0f}s ({n_seg} seg/session)  "
                      f"n_eval={len(keep)}  acc={a:.4f}  f1={f:.4f}", flush=True)
        # Restore the full-session transform so anything after this is unaffected.
        base_te._fit_alignment(dc.get("ea_scale", 0.2), dc.get("ea_trim", 0.05),
                               dc.get("ea_eps", 1e-6))

    def _path(tmpl):
        """Fill {fold}/{seed} and make the directory.  Both are in the name so
        one fold's seeds never overwrite each other."""
        out = tmpl.format(fold=fold_tag, seed=seed)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        return out

    # Window-level logits at the reported epoch, with provenance, so a
    # clip-level (or causal, delayed) vote can be computed afterwards without
    # retraining.  The loader is unshuffled, so row i is test_ds[i]; segments
    # are appended in time order within each clip.
    clip_metrics = {}
    base = test_ds.dataset if isinstance(test_ds, Subset) else test_ds
    rows = test_ds.indices if isinstance(test_ds, Subset) else list(range(len(base)))
    if best_test_logits is not None:
        meta = np.asarray([base._seg_meta[i] for i in rows])
        y_te = np.asarray([base._segments[i][1] for i in rows])
        clip_metrics = _clip_level_metrics(
            best_test_logits, y_te, meta[:, 1], meta[:, 2], fc["n_classes"])
        logit_path = fc.get("save_logits")
        if logit_path:
            logit_path = _path(logit_path)
            np.savez(logit_path,
                     logits=best_test_logits,
                     label=y_te,
                     subject=meta[:, 0], session=meta[:, 1], clip=meta[:, 2],
                     epoch=sel[2] if sel else -1, seed=seed)
            print(f"[logits] saved {best_test_logits.shape} -> {logit_path}", flush=True)

    # Full model at the reported epoch — backbone AND both heads, so the model
    # whose accuracy is reported can be reloaded exactly for a feature probe.
    # ``save_backbone`` keeps only model.labram and is left as it was.
    state_path = fc.get("save_state")
    if state_path and best_state is not None:
        state_path = _path(state_path)
        torch.save({"model": best_state,
                    "epoch": sel[2] if sel else total_epochs,
                    "val_f1": best_val_f1,
                    "seed": seed, "fold": fold_tag,
                    "lora": use_lora,
                    "note": "full model (labram + main_head + asym_head) "
                            "at the reported best-val epoch"},
                   state_path)
        print(f"[state] saved full model -> {state_path}  "
              f"(epoch {sel[2] if sel else total_epochs}, seed {seed})", flush=True)

    ckpt_path = fc.get("save_backbone")
    if ckpt_path:
        ckpt_path = _path(ckpt_path)
        if best_backbone is not None:
            payload, ep_note, eff = best_backbone, sel[2], best_adv_lambda
        else:
            payload, ep_note, eff = model.labram.state_dict(), total_epochs, model.adv_lambda
        torch.save({"labram": payload,
                    "lambda_adv": lambda_adv,
                    "effective_lambda": eff,
                    "epoch": ep_note,
                    "lora": use_lora,
                    "note": "backbone at the reported (best-val) epoch"},
                   ckpt_path)
        print(f"[ckpt] saved backbone -> {ckpt_path}  "
              f"(epoch {ep_note}, effective lambda={eff:.3f})", flush=True)

    acc, f1 = _evaluate(model, test_loader, device, amp, center_table=tbl_te)
    if sel is not None:
        te_acc, te_f1, ep = sel
        out = {
            "accuracy": te_acc, "f1_macro": te_f1,     # test @ best-val epoch
            "selected_epoch": ep, "val_f1": best_val_f1,
            "final_accuracy": acc, "final_f1_macro": f1,
            "best_accuracy": te_acc, "seed": seed,
        }
        out.update(clip_metrics)
        return out
    return {"accuracy": acc, "f1_macro": f1, "best_accuracy": max(best_acc, acc),
            "seed": seed, **clip_metrics}


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
        ea_mode=dc.get("ea_mode", "full"),
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
        lora_targets=fc.get("lora_targets", ("fc1", "fc2")),
        ch_names=getattr(getattr(train_ds, "dataset", train_ds), "ch_names", None),   # Subset 이면 원본에서
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
    DS = _LOADERS[dc.get("loader", "SEEDRawDataset")]
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
        ea_mode=dc.get("ea_mode", "full"),
    )

    # One run per (fold, seed).  Several seeds per fold is what makes a later
    # comparison interpretable: without it the run-to-run spread is confounded
    # with whatever is being compared.
    seeds = fc.get("seeds") or [fc.get("seed", 0)]
    seeds = [int(x) for x in seeds]
    print(f"  seeds={seeds}  ({len(fold_subjects)} folds x {len(seeds)} seeds "
          f"= {len(fold_subjects) * len(seeds)} runs)", flush=True)

    results = defaultdict(list)
    rows = []          # one dict per (fold, seed), written out at the end
    for subj in fold_subjects:
      for seed in seeds:
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
            sess = dc.get("sessions") or [1, 2, 3]
            train_ds = DS(subjects=train_subjects, sessions=sess, **ds_kwargs)
            test_ds  = DS(subjects=[subj],         sessions=sess, **ds_kwargs)
            val_ds   = (DS(subjects=val_subjects, sessions=sess, **ds_kwargs)
                        if val_subjects else None)
            m = _train_and_eval(cfg, train_ds, test_ds, device, val_ds=val_ds,
                                fold_tag=str(subj), seed=seed)
            if "selected_epoch" in m:
                print(f"  -> selected epoch {m['selected_epoch']} "
                      f"(val_f1={m['val_f1']:.4f}); last-epoch test "
                      f"acc={m['final_accuracy']:.4f}", flush=True)

        elif mode == "cross_session":
            # LaBraM official protocol: session 1 train, session 2 val, session 3 test
            # But per-subject, all subjects together
            train_ds = SEEDRawDataset(subjects=[subj], sessions=[1], **ds_kwargs)
            test_ds  = SEEDRawDataset(subjects=[subj], sessions=[3], **ds_kwargs)
            m = _train_and_eval(cfg, train_ds, test_ds, device,
                                fold_tag=str(subj), seed=seed)

        else:  # subject_dependent
            # Two conventions, one per dataset, both taken from the literature
            # rather than invented here (checked 2026-10-03):
            #
            #   SEED    per SESSION: clips 1-9 train, 10-15 test, results
            #           averaged over the three sessions.
            #   SEED-V  3 folds over clip GROUPS (1-5 / 6-10 / 11-15) with the
            #           three sessions POOLED — each fold trains on that
            #           group's clips from all three sessions.
            #
            # The asymmetry is theirs, not ours; reports/EVAL_PROTOCOL.md says
            # so, because it looks like an inconsistency otherwise.  Pooling
            # means the SEED-V number includes within-subject across-session
            # generalisation (cf. S15), and that has to be stated.
            #
            # Validation cannot be a held-out SUBJECT the way the LOSO path
            # does it — this protocol lives inside one subject.  ``sd_val_frac``
            # takes the tail of every training clip instead; it is correlated
            # with training and is used only to pick the epoch.  Setting it to 0
            # reports the last epoch, and then ``_train_and_eval`` keeps no
            # logits, so there are NO clip-level metrics.
            sd_train = fc.get("sd_train_per_session", SD_TRAIN_CLIPS)
            val_frac = fc.get("sd_val_frac", 0.2)
            n_folds = fc.get("sd_n_folds", 0)
            pool = fc.get("sd_pool_sessions", False)
            sess = dc.get("sessions") or [1, 2, 3]
            groups = [list(sess)] if pool else [[ss] for ss in sess]
            acc = defaultdict(list)
            pooled = defaultdict(list)
            for g in groups:
                g_ds = DS(subjects=[subj], sessions=g, **ds_kwargs)
                # Split on the film-clip index, not on load order.  With the
                # sessions pooled, clip 1-5 means those clips in EVERY session.
                splits = (g_ds.sd_folds(n_folds) if n_folds
                          else [g_ds.sd_split(sd_train)])
                for fi, (train_idx, test_idx) in enumerate(splits, 1):
                    if not train_idx or not test_idx:
                        continue
                    val_ds = None
                    if val_frac:
                        train_idx, val_idx = g_ds.sd_val_split(
                            train_idx, val_frac)
                        if val_idx:
                            val_ds = Subset(g_ds, val_idx)
                    who = "all" if pool else f"s{g[0]}"
                    tag = f"{subj}{who}" + (f"f{fi}" if n_folds else "")
                    sess_m = _train_and_eval(cfg, Subset(g_ds, train_idx),
                                             Subset(g_ds, test_idx), device,
                                             val_ds=val_ds, fold_tag=tag,
                                             seed=seed)
                    # Diagnostics stay as per-split values (meaned below).
                    for k in ("selected_epoch", "val_f1"):
                        if k in sess_m:
                            acc[k].append(sess_m[k])
                    # Decisions are pooled, not averaged.
                    for k in ("_clip_pred", "_clip_true",
                              "_win_pred", "_win_true"):
                        if k in sess_m:
                            pooled[k].append(np.asarray(sess_m[k]))
                    if "_clip_pred" not in sess_m:
                        # No validation set -> no logits -> no decisions to
                        # pool.  Fall back so the run still reports something.
                        for k in ("accuracy", "f1_macro"):
                            if k in sess_m:
                                acc[k].append(sess_m[k])
                    extra = (f" clip_acc={sess_m['clip_accuracy']:.4f}"
                             if "clip_accuracy" in sess_m else "")
                    print(f"  [S{subj}/{who}"
                          f"{f'/F{fi}' if n_folds else ''}] "
                          f"acc={sess_m['accuracy']:.4f} "
                          f"F1={sess_m['f1_macro']:.4f}{extra}", flush=True)
            # Pool every split's decisions, then score ONCE.  Averaging
            # per-split macro-F1 or balanced accuracy is not the pooled value
            # (both are non-linear), and the splits do not even hold the same
            # number of windows — SEED-V's clips differ in length, so the three
            # folds carried 532 / 667 / 624 test windows for subject 1.
            from aggregate_logits import balanced as _bal
            m = {k: float(np.mean(v)) for k, v in acc.items() if v}
            if pooled.get("_clip_pred"):
                n_cls = fc["n_classes"]
                cp = np.concatenate(pooled["_clip_pred"])
                ct = np.concatenate(pooled["_clip_true"])
                wp = np.concatenate(pooled["_win_pred"])
                wt = np.concatenate(pooled["_win_true"])
                m.update({
                    "accuracy": float((wp == wt).mean()),
                    "f1_macro": float(f1_score(wt, wp, average="macro")),
                    "window_balanced": _bal(wp, wt, n_cls),
                    "clip_accuracy": float((cp == ct).mean()),
                    "clip_f1_macro": float(f1_score(ct, cp, average="macro")),
                    "clip_balanced": _bal(cp, ct, n_cls),
                    "n_clips": int(len(ct)),
                })
            m.setdefault("accuracy", 0.0)
            m.setdefault("f1_macro", 0.0)
            m["best_accuracy"] = 0.0

        dt = time.time() - t0
        results["accuracy"].append(m["accuracy"])
        results["f1_macro"].append(m["f1_macro"])
        row = {"subject": subj, "seed": seed, "seconds": round(dt, 1),
               "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}
        for k in ("clip_accuracy", "clip_f1_macro", "clip_balanced",
                  "window_balanced", "n_clips", "selected_epoch", "val_f1"):
            if k in m:
                row[k] = m[k]
        rows.append(row)
        # Rewrite the CSV after every run, not once at the end: a 2-3 day
        # sweep that dies at fold 12 must not lose the eleven folds before it.
        _write_csv(rows, fc.get("results_csv"))
        extra = ""
        if "clip_accuracy" in m:
            extra = (f"  clip_acc={m['clip_accuracy']:.4f}"
                     f"  clip_bacc={m['clip_balanced']:.4f}"
                     f"  win_bacc={m['window_balanced']:.4f}")
        print(f"[Subject {subj:>2} seed {seed}]  acc={m['accuracy']:.4f}  "
              f"F1={m['f1_macro']:.4f}{extra}  ({dt:.1f}s)", flush=True)
        if use_wandb:
            wandb.log({"fold/subject": subj, "fold/seed": seed,
                       "fold/accuracy": m["accuracy"], "fold/f1_macro": m["f1_macro"]})

    _report_cv(mode, rows, fold_subjects, seeds, fc.get("results_csv"))

    if use_wandb:
        acc_arr = np.array(results["accuracy"])
        f1_arr = np.array(results["f1_macro"])
        wandb.log({"summary/acc_mean": acc_arr.mean(), "summary/acc_std": acc_arr.std(),
                    "summary/f1_mean": f1_arr.mean(), "summary/f1_std": f1_arr.std()})
        wandb.finish()


def _report_cv(mode, rows, fold_subjects, seeds, csv_path=None):
    """Print the fold x seed table and the seed-averaged summary.

    Two different means are reported and they answer different questions:
    averaging over seeds first gives one number per subject (what a comparison
    between conditions should be paired on), while the spread ACROSS seeds
    within a subject is the run-to-run noise floor that any claimed
    improvement has to clear.
    """
    if not rows:
        print("no results")
        return
    has_clip = "clip_accuracy" in rows[0]
    by = {}
    for r in rows:
        by.setdefault(r["subject"], {})[r["seed"]] = r

    print(f"\n{'='*78}\n{mode} Results — {len(rows)} runs "
          f"({len(fold_subjects)} folds x {len(seeds)} seeds)\n{'='*78}")

    hdr = f"{'subj':>4} {'seed':>4} {'win_acc':>8} {'win_f1':>8}"
    if has_clip:
        hdr += f" {'win_bacc':>9} {'clip_acc':>9} {'clip_f1':>8} {'clip_bacc':>10}"
    print(hdr)
    for subj in fold_subjects:
        for seed in seeds:
            r = by.get(subj, {}).get(seed)
            if r is None:
                continue
            line = (f"{subj:>4} {seed:>4} {r['accuracy']:>8.4f} "
                    f"{r['f1_macro']:>8.4f}")
            if has_clip:
                line += (f" {r['window_balanced']:>9.4f} {r['clip_accuracy']:>9.4f}"
                         f" {r['clip_f1_macro']:>8.4f} {r['clip_balanced']:>10.4f}")
            print(line)

    # per-subject seed mean and the seed spread (the noise floor)
    print(f"\n{'subj':>4} {'win_acc mean':>13} {'± seed sd':>10}", end="")
    if has_clip:
        print(f" {'clip_acc mean':>14} {'± seed sd':>10}", end="")
    print()
    means_w, means_c = [], []
    for subj in fold_subjects:
        rs = [by[subj][s] for s in seeds if s in by.get(subj, {})]
        if not rs:
            continue
        w = np.array([r["accuracy"] for r in rs])
        means_w.append(w.mean())
        sd_w = w.std(ddof=1) if len(w) > 1 else float("nan")
        print(f"{subj:>4} {w.mean():>13.4f} {sd_w:>10.4f}", end="")
        if has_clip:
            c = np.array([r["clip_accuracy"] for r in rs])
            means_c.append(c.mean())
            sd_c = c.std(ddof=1) if len(c) > 1 else float("nan")
            print(f" {c.mean():>14.4f} {sd_c:>10.4f}", end="")
        print()

    mw = np.array(means_w)
    print(f"\n  window acc (subject means): {mw.mean():.4f} "
          f"± {mw.std(ddof=1) if len(mw) > 1 else 0:.4f}  (n={len(mw)} subjects)")
    if has_clip and means_c:
        mc = np.array(means_c)
        print(f"  clip   acc (subject means): {mc.mean():.4f} "
              f"± {mc.std(ddof=1) if len(mc) > 1 else 0:.4f}")
    if len(seeds) > 1:
        sds = []
        for subj in fold_subjects:
            rs = [by[subj][s] for s in seeds if s in by.get(subj, {})]
            if len(rs) > 1:
                sds.append(np.std([r["accuracy"] for r in rs], ddof=1))
        if sds:
            print(f"  run-to-run sd within a subject: mean {np.mean(sds):.4f}  "
                  f"max {np.max(sds):.4f}   <- any claimed gain must clear this")
    print("=" * 78)

    if csv_path:
        _write_csv(rows, csv_path)
        print(f"[csv] {len(rows)} rows -> {csv_path}")


def _write_csv(rows, csv_path):
    """Dump every completed run.  Called after each one so a killed sweep
    still leaves the folds it finished."""
    if not csv_path or not rows:
        return
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


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
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seeds, overriding finetune.seeds "
                             "in the config (e.g. 0,1,2). One run per "
                             "(fold, seed).")
    parser.add_argument("--cudnn_deterministic", action="store_true",
                        help="force deterministic cuDNN/cuBLAS kernels. Makes "
                             "two runs at the same seed bit-identical, at a "
                             "speed cost. OFF by default — seeding alone "
                             "leaves a small residual spread; measure it "
                             "before deciding.")
    args = parser.parse_args()
    folds = ([int(x) for x in args.folds.split(",")] if args.folds else None)

    if args.cudnn_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[repro] cuDNN deterministic ON (slower)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seeds:
        cfg["finetune"]["seeds"] = [int(x) for x in args.seeds.split(",")]

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
