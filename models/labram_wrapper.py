"""
LaBraM wrapper — loads the pretrained Large Brain Model and exposes a clean
feature-extraction interface for our asymmetry adapter pipeline.

Usage
-----
    wrapper = LaBraMWrapper(
        labram_repo="/home/kihyeonjoo/LaBraM",
        ckpt_path="/home/kihyeonjoo/LaBraM/checkpoints/labram-base.pth",
        ch_names=SEED_CH_NAMES,      # 62-channel list
        freeze=True,                  # freeze all LaBraM params
    )

    # x: (B, 62, n_patches, 200)  raw EEG in LaBraM format
    feats = wrapper(x)               # (B, 62, n_patches, embed_dim=200)

The wrapper:
  1. Dynamically imports LaBraM from the local clone (not a pip package).
  2. Loads pretrained weights once.
  3. Applies ``forward_features`` with ``return_patch_tokens=True`` to get
     per-(channel, patch) token embeddings.
  4. Reshapes the flat token sequence back into (B, C, T_patches, D).

Channel indexing
----------------
LaBraM uses a learned positional embedding indexed over ~128 electrodes in
the standard 10-20 system. We map our SEED channel names to these indices
via ``utils.get_input_chans`` from the LaBraM repo.
"""

from __future__ import annotations

import os
import sys
from typing import List

import torch
import torch.nn as nn


def _register_labram_on_path(labram_repo: str) -> None:
    """Add LaBraM repo to ``sys.path`` so we can import its modules."""
    labram_repo = os.path.abspath(os.path.expanduser(labram_repo))
    if labram_repo not in sys.path:
        sys.path.insert(0, labram_repo)


# LaBraM's learned channel positional embedding is indexed over this 10-20
# electrode list (copied from LaBraM/utils.py to avoid an h5py dependency).
_LABRAM_STANDARD_1020: List[str] = [
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


def _get_input_chans(ch_names: List[str]) -> List[int]:
    """Local reimplementation of LaBraM's utils.get_input_chans (no h5py import)."""
    input_chans = [0]  # CLS token
    for ch_name in ch_names:
        input_chans.append(_LABRAM_STANDARD_1020.index(ch_name) + 1)
    return input_chans


class LaBraMWrapper(nn.Module):
    """Frozen LaBraM backbone → per-channel per-patch feature tensor.

    Parameters
    ----------
    labram_repo  : str         — path to the LaBraM git clone
    ckpt_path    : str         — path to ``labram-base.pth`` (or large/huge)
    ch_names     : list[str]   — channel names matching LaBraM's standard_1020
    model_name   : str         — LaBraM model variant to instantiate
    freeze       : bool        — freeze all LaBraM parameters (default True)

    Output shape
    ------------
    (B, n_channels, n_patches, embed_dim)
    For labram-base: embed_dim = 200.
    """

    def __init__(
        self,
        labram_repo: str,
        ckpt_path: str,
        ch_names: List[str],
        model_name: str = "labram_base_patch200_200",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        _register_labram_on_path(labram_repo)

        # Import LaBraM lazily so that environments without it can still
        # import the rest of our codebase.
        import modeling_finetune  # noqa: F401  (registers model)
        from timm.models import create_model

        # Build model and load pretrained weights.
        # NOTE: use_mean_pooling=False — matches the labram-base.pth layout
        # (checkpoint has ``norm.{weight,bias}``, not ``fc_norm.*``).
        model = create_model(
            model_name,
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

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)

        # labram-base.pth comes from a student–teacher pretraining stage —
        # the backbone weights are under the ``student.`` prefix. Strip it.
        if any(k.startswith("student.") for k in sd.keys()):
            sd = {
                k[len("student."):]: v
                for k, v in sd.items()
                if k.startswith("student.")
            }

        # Drop pretraining-only heads that don't exist on our model.
        drop_prefixes = (
            "head.", "lm_head.", "projection_head.",
            "mask_token", "logit_scale",
        )
        sd = {
            k: v for k, v in sd.items()
            if not any(k.startswith(p) for p in drop_prefixes)
        }

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[LaBraM] loaded {model_name} from {os.path.basename(ckpt_path)}  "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

        self.model = model
        self.embed_dim = model.embed_dim
        self.n_channels = len(ch_names)

        # Precompute input_chans for LaBraM pos-embedding indexing
        # (leading 0 is the CLS token slot).
        input_chans = _get_input_chans(ch_names)
        self.register_buffer(
            "input_chans",
            torch.tensor(input_chans, dtype=torch.long),
            persistent=False,
        )

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.frozen = freeze

    def train(self, mode: bool = True):  # type: ignore[override]
        """Keep the backbone in eval mode even when the outer model trains."""
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg : (B, n_channels, n_patches, patch_size=200)

        Returns
        -------
        feats : (B, n_channels, n_patches, embed_dim)
        """
        B, C, P, T = eeg.shape
        assert C == self.n_channels, (
            f"expected {self.n_channels} channels, got {C}"
        )

        # LaBraM's forward_features expects (B, N_ch, n_patches, patch_size).
        # With return_patch_tokens=True it returns (B, N_ch * n_patches, D).
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            tokens = self.model.forward_features(
                eeg,
                input_chans=self.input_chans.tolist(),
                return_patch_tokens=True,
            )

        # Reshape to (B, C, P, D) — LaBraM flattens channels first, then patches
        feats = tokens.reshape(B, C, P, self.embed_dim)
        return feats
