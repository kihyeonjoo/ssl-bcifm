"""
AsymmetryAdapter — lightweight transformer that sits on top of a frozen EEG
foundation model (LaBraM) and produces per-hemisphere embeddings for our
asymmetry-based pretext tasks and downstream classifier.

Pipeline
--------
    Raw EEG (B, 62, P, 200)
        │
        ▼  LaBraMWrapper (frozen)
    feats (B, 62, P, D_labram)
        │
        ▼  hemisphere split (LEFT_IDX / RIGHT_IDX)
    feats_L (B, 27, P, D_labram)    feats_R (B, 27, P, D_labram)
        │
        ▼  AsymmetryAdapter (same instance, called twice)
    z_L (B, d_model)                z_R (B, d_model)

The adapter treats (channel × patch) as a flat token sequence and processes
it with a small transformer. A learned CLS token produces the hemisphere
summary.  Both hemispheres pass through the *same* adapter instance → true
weight sharing, so ``z_L - z_R`` encodes asymmetry cleanly.

This module is intentionally shape-compatible with ``DualStreamEncoder``
(it returns ``(z_L, z_R, z_joint)``) so the existing pretext task modules
and the classifier head can be reused without changes.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from data.preprocessing import LEFT_IDX, RIGHT_IDX


# ── Positional encoding (sinusoidal, learnable-free) ─────────────────────────

class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ── Single-hemisphere adapter ────────────────────────────────────────────────

class HemisphereAdapter(nn.Module):
    """Lightweight transformer over (channel × patch) tokens from one hemisphere.

    Input  : (B, C_h, P, D_labram)
    Output : CLS (B, d_model)  or  (CLS, seq) when return_sequence=True
    """

    def __init__(
        self,
        in_dim: int = 200,         # LaBraM base embed_dim
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = _SinusoidalPE(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        feats: torch.Tensor,
        return_sequence: bool = False,
    ):
        """
        Parameters
        ----------
        feats : (B, C_h, P, D_labram)
        """
        B, C, P, D = feats.shape
        x = feats.reshape(B, C * P, D)            # flatten (channel, patch)
        x = self.proj(x)                           # (B, C*P, d_model)
        x = self.norm_in(x)

        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)             # (B, 1 + C*P, d_model)
        x = self.pos_enc(x)

        x = self.transformer(x)
        x = self.norm(x)

        if return_sequence:
            return x[:, 0], x[:, 1:]               # CLS, seq
        return x[:, 0]                              # CLS only


# ── Dual-stream asymmetry adapter (shape-compatible with DualStreamEncoder) ─

class AsymmetryAdapter(nn.Module):
    """Dual-hemisphere adapter over frozen LaBraM features.

    Shares a single HemisphereAdapter across left and right → real weight
    sharing.  The forward signature matches ``DualStreamEncoder`` so that
    downstream modules (classifier, pretext tasks) can treat it identically.

    Parameters
    ----------
    labram        : LaBraMWrapper  — the frozen foundation model
    d_model       : int            — adapter embedding dimension
    n_heads, n_layers, dim_feedforward, dropout
                                    — adapter transformer config
    left_idx, right_idx : list[int]
                                    — channel indices into LaBraM output
                                      (defaults to SEED LEFT_IDX / RIGHT_IDX)

    Forward
    -------
    eeg : (B, 62, n_patches, 200)   — raw EEG in LaBraM format

    Returns
    -------
    z_left  : (B, d_model)
    z_right : (B, d_model)
    z_joint : (B, 2 * d_model)      — concat for downstream fusion head
    """

    def __init__(
        self,
        labram,                                          # LaBraMWrapper
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        left_idx: List[int] = None,
        right_idx: List[int] = None,
    ) -> None:
        super().__init__()
        self.labram = labram
        self.d_model = d_model

        self.left_idx = torch.tensor(
            left_idx if left_idx is not None else LEFT_IDX, dtype=torch.long
        )
        self.right_idx = torch.tensor(
            right_idx if right_idx is not None else RIGHT_IDX, dtype=torch.long
        )

        # Single shared adapter for both hemispheres
        self.adapter = HemisphereAdapter(
            in_dim=labram.embed_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    # The original DualStreamEncoder had an ``encoder`` attribute used by
    # pretext tasks.  Expose the same attribute name so we can reuse
    # ``CrossHemisphereMaskedPrediction`` / ``TemporalDeltaAsymmetry`` with
    # minimal changes — but note that those tasks call
    # ``self.encoder.encoder(x, return_sequence=True)`` which expects the
    # inner single-hemi encoder.  Here we expose the adapter directly.
    @property
    def encoder(self):
        return self.adapter

    def _split(self, feats: torch.Tensor):
        """Split LaBraM features into left and right hemispheres."""
        left_idx = self.left_idx.to(feats.device)
        right_idx = self.right_idx.to(feats.device)
        return feats.index_select(1, left_idx), feats.index_select(1, right_idx)

    def encode_hemi(
        self,
        hemi_feats: torch.Tensor,
        return_sequence: bool = False,
    ):
        """Encode one hemisphere's LaBraM features through the shared adapter."""
        return self.adapter(hemi_feats, return_sequence=return_sequence)

    def forward(
        self,
        eeg: torch.Tensor,
    ):
        """
        Parameters
        ----------
        eeg : (B, 62, n_patches, patch_size)

        Returns
        -------
        z_left, z_right, z_joint
        """
        feats = self.labram(eeg)                   # (B, 62, P, D_labram)
        feats_L, feats_R = self._split(feats)       # (B, 27, P, D)

        z_left  = self.adapter(feats_L)             # (B, d_model)
        z_right = self.adapter(feats_R)             # (B, d_model)
        z_joint = torch.cat([z_left, z_right], dim=-1)
        return z_left, z_right, z_joint
