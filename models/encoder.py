"""
Dual-stream shared encoder  E_φ
================================
Architecture
------------
  SpectroSpatialCNN  →  TemporalTransformer  →  (B, d_model)

Both left- and right-hemisphere streams share identical weights by reusing
the same HemisphereEncoder module — no parameter duplication.

Shapes (default config, SEED)
------------------------------
  input per hemisphere : (B, C_h=27, 4, T=16)
  after SpectroSpatialCNN : (B, T=16, d_model)
  after TemporalTransformer (CLS) : (B, d_model)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── 1. Spectro-spatial CNN ────────────────────────────────────────────────────

class SpectroSpatialCNN(nn.Module):
    """Extract spectro-spatial features from a single time frame.

    The (C_h × n_bands) map is treated as a 2-D image:

      Stage 1 – spectral fusion   : Conv2d kernel (1, n_bands) collapses the
                                    frequency axis entirely, fusing all bands
                                    into a per-channel feature vector.
      Stage 2+ – spatial encoding : Stacked 1-D-spatial Conv2d kernels (k, 1)
                                    capture neighbouring channel interactions.
      Final pooling               : AdaptiveAvgPool across channels → scalar
                                    feature per filter.

    Parameters
    ----------
    n_channels : int
        EEG channels per hemisphere (27 for SEED).
    n_bands : int
        Number of frequency bands (4).
    d_model : int
        Output feature dimension per time frame.
    cnn_channels : tuple[int, ...]
        Hidden channel sizes for the intermediate spatial conv stages.
        Length determines depth.  Default (32, 64, 128).

    Input  : (B, C_h, n_bands, T)
    Output : (B, T, d_model)
    """

    def __init__(
        self,
        n_channels: int = 27,
        n_bands: int = 4,
        d_model: int = 256,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # ── Stage 1: spectral fusion ──────────────────────────────────────────
        # kernel (1, n_bands) covers all frequency bands at once per channel
        layers += [
            nn.Conv2d(1, cnn_channels[0], kernel_size=(1, n_bands), bias=False),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.GELU(),
        ]

        # ── Stages 2+: spatial (channel-axis) encoding ───────────────────────
        # After stage 1 the freq axis is collapsed to 1, so kernels are (k, 1)
        in_ch = cnn_channels[0]
        for out_ch in cnn_channels[1:]:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch

        # ── Final projection to d_model ───────────────────────────────────────
        layers += [
            nn.Conv2d(in_ch, d_model, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        ]

        self.conv_net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # global spatial pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape

        # (B, C, F, T) → (B, T, C, F) → (B*T, 1, C, F)
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, C, F)

        x = self.conv_net(x)         # (B*T, d_model, C', 1)
        x = self.pool(x).flatten(1)  # (B*T, d_model)

        return x.reshape(B, T, -1)   # (B, T, d_model)


# ── 2. Sinusoidal positional encoding ────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
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
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ── 3. Temporal Transformer ───────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """Transformer encoder over the time axis with a learnable [CLS] token.

    A learnable [CLS] token is prepended; its final-layer representation
    is taken as the global temporal embedding — the same design as BERT /
    ViT, well-suited for masked prediction pretext tasks.

    Parameters
    ----------
    d_model : int
        Model dimension (must equal SpectroSpatialCNN output dim).
    n_heads : int
        Number of attention heads.  Must divide d_model evenly.
    n_layers : int
        Number of TransformerEncoderLayer blocks.
    dim_feedforward : int
        Inner dimension of the FFN sublayer.
    dropout : float
        Dropout rate applied in attention, FFN, and positional encoding.

    Input  : (B, T, d_model)
    Output : (B, d_model)                       — [CLS] token (default)
           | ((B, d_model), (B, T, d_model))    — (CLS, sequence) when return_sequence=True
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)

        # Pre-LayerNorm (norm_first=True) for training stability
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
        self, x: torch.Tensor, return_sequence: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d_model)
        B = x.size(0)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)          # (B, T+1, d_model)
        x = self.pos_enc(x)

        x = self.transformer(x)  # (B, T+1, d_model)
        x = self.norm(x)         # normalise all tokens (CLS + sequence)

        if return_sequence:
            return x[:, 0], x[:, 1:]  # (B, d_model), (B, T, d_model)
        return x[:, 0]                 # (B, d_model)  — CLS only


# ── 4. Single-hemisphere encoder ─────────────────────────────────────────────

class HemisphereEncoder(nn.Module):
    """Full single-stream pipeline: SpectroSpatialCNN → TemporalTransformer.

    Input  : (B, C_h, n_bands, T)
    Output : (B, d_model)
    """

    def __init__(
        self,
        n_channels: int = 27,
        n_bands: int = 4,
        d_model: int = 256,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spectro_spatial = SpectroSpatialCNN(
            n_channels=n_channels,
            n_bands=n_bands,
            d_model=d_model,
            cnn_channels=cnn_channels,
        )
        self.temporal = TemporalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, return_sequence: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        seq = self.spectro_spatial(x)                          # (B, T, d_model)
        return self.temporal(seq, return_sequence=return_sequence)


# ── 5. Dual-stream shared encoder E_φ ────────────────────────────────────────

class DualStreamEncoder(nn.Module):
    """Dual-stream encoder E_φ with weight sharing across hemispheres.

    Both the left and right hemisphere streams pass through the *same*
    HemisphereEncoder instance — there is literally one copy of every
    parameter.  This is the standard PyTorch way to share weights: call
    the same module twice.

    Parameters
    ----------
    n_channels_per_hemi : int
        EEG channels per hemisphere after midline removal (27 for SEED).
    n_bands : int
        Frequency bands (4 : theta, alpha, beta, gamma).
    d_model : int
        Embedding dimension for each hemisphere.
    cnn_channels : tuple[int, ...]
        Hidden channels for SpectroSpatialCNN intermediate stages.
    n_heads : int
        Attention heads for TemporalTransformer.
    n_layers : int
        Transformer depth.
    dim_feedforward : int
        FFN inner dimension.
    dropout : float

    Forward
    -------
    left, right : Tensor (B, C_h, n_bands, T)

    Returns
    -------
    z_left  : Tensor (B, d_model)
    z_right : Tensor (B, d_model)
    z_joint : Tensor (B, 2 * d_model)   — concat; used by downstream heads
    """

    def __init__(
        self,
        n_channels_per_hemi: int = 27,
        n_bands: int = 4,
        d_model: int = 256,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # One encoder — shared by both hemisphere streams
        self.encoder = HemisphereEncoder(
            n_channels=n_channels_per_hemi,
            n_bands=n_bands,
            d_model=d_model,
            cnn_channels=cnn_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single hemisphere.

        Parameters
        ----------
        x : Tensor (B, C_h, n_bands, T)

        Returns
        -------
        Tensor (B, d_model)
        """
        return self.encoder(x)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_left  = self.encoder(left)                       # (B, d_model)
        z_right = self.encoder(right)                      # (B, d_model)
        z_joint = torch.cat([z_left, z_right], dim=-1)    # (B, 2*d_model)
        return z_left, z_right, z_joint
