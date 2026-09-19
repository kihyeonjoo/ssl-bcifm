"""
Cross-Hemisphere *Channel* Masked Prediction (adapter version)
==============================================================

Adapter-space counterpart of ``tasks/cross_hemisphere.py``. Since LaBraM
features are per-(channel, patch) and no longer have a frequency band
axis, we mask a random subset of *channel features* of one hemisphere
and ask the decoder to reconstruct them using the intact hemisphere's
features via cross-attention.

Procedure
---------
1. Pass raw EEG through the frozen LaBraMWrapper → (B, 62, P, D_labram)
2. Split hemispheres → (B, 27, P, D) each
3. Random hemi_side ∈ {L, R}, random subset of k channels to mask
4. Zero out the k masked channel-feature rows in the masked hemisphere
5. Encode both hemispheres with the shared adapter, return_sequence=True
   → z_masked_seq, z_intact_seq  (each (B, 1+C*P, d_model))
6. CrossHemisphereDecoder (same module as in the STFT pipeline):
       Query   = z_masked_seq[:, 1:]   # drop CLS, keep tokens
       Key/Val = z_intact_seq[:, 1:]
   → recon (B, n_tokens, d_model) → Linear → (B, n_tokens, D_labram)
7. ℒ_main = MSE(recon[masked_token_positions],
                target_feats[masked_token_positions])

The target is the *original* LaBraM features for the masked channels —
no normalisation, preserving magnitude (avoids the trivial-zero collapse
we hit earlier in the STFT pipeline).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMaskDecoder(nn.Module):
    """Cross-attention decoder that predicts per-token LaBraM features.

    Architecture mirrors the STFT pipeline's CrossHemisphereDecoder but
    with an output projection matching LaBraM's embedding dim.
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(
        self,
        z_masked_seq: torch.Tensor,
        z_intact_seq: torch.Tensor,
    ) -> torch.Tensor:
        x = self.transformer_decoder(tgt=z_masked_seq, memory=z_intact_seq)
        x = self.norm(x)
        return self.out_proj(x)              # (B, T, out_dim)


class CrossHemisphereChannelMaskedPrediction(nn.Module):
    """Main pretext task: masked-channel reconstruction on LaBraM features.

    Parameters
    ----------
    adapter     : AsymmetryAdapter     — must expose ``labram`` and
                                        ``encode_hemi`` (shared across hemi)
    d_model     : int                  — adapter hidden dim
    out_dim     : int                  — LaBraM embed dim (default 200)
    n_masked_ch : int                  — how many channels to mask (default 5)
    n_patches   : int                  — patches per segment (default 4 for 4 s)
    dec_heads   : int
    dec_layers  : int
    dropout     : float
    """

    def __init__(
        self,
        adapter,
        d_model: int,
        out_dim: int,
        n_masked_ch: int = 5,
        n_patches: int = 4,
        dec_heads: int = 8,
        dec_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.n_masked_ch = n_masked_ch
        self.n_patches = n_patches
        self.decoder = ChannelMaskDecoder(
            d_model=d_model,
            out_dim=out_dim,
            n_heads=dec_heads,
            n_layers=dec_layers,
            dropout=dropout,
        )

    def forward(self, eeg: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        eeg : (B, 62, n_patches, patch_size)  raw LaBraM-format input
        """
        # 1. LaBraM features (frozen)
        feats = self.adapter.labram(eeg)           # (B, 62, P, D_labram)
        feats_L, feats_R = self.adapter._split(feats)  # (B, 27, P, D)

        B, C_h, P, D = feats_L.shape

        # 2. Randomly choose hemisphere & channel indices to mask
        hemi_side = int(torch.randint(0, 2, ()).item())
        masked_ch = torch.randperm(C_h, device=eeg.device)[: self.n_masked_ch]

        if hemi_side == 0:
            orig = feats_L.clone()
            intact = feats_R
            masked = feats_L.clone()
            masked[:, masked_ch] = 0.0
        else:
            orig = feats_R.clone()
            intact = feats_L
            masked = feats_R.clone()
            masked[:, masked_ch] = 0.0

        # 3. Dual encoding via shared adapter (return full sequence)
        _, z_masked_seq = self.adapter.encode_hemi(masked, return_sequence=True)
        _, z_intact_seq = self.adapter.encode_hemi(intact, return_sequence=True)
        # z_*_seq shape: (B, C_h * P, d_model)

        # 4. Decode (cross-attention) and project to LaBraM embedding space
        recon = self.decoder(z_masked_seq, z_intact_seq)  # (B, C_h*P, D)

        # 5. Reshape recon and target to (B, C_h, P, D) and select masked channels
        recon = recon.reshape(B, C_h, P, D)
        target = orig                                      # raw LaBraM features

        # Build mask over channel axis
        recon_masked = recon[:, masked_ch]                 # (B, n_masked_ch, P, D)
        target_masked = target[:, masked_ch]

        loss = F.mse_loss(recon_masked, target_masked)

        # Also expose z_L, z_R for convenience (CLS tokens)
        z_L = self.adapter.encode_hemi(feats_L)
        z_R = self.adapter.encode_hemi(feats_R)

        return {
            "loss":       loss,
            "z_left":     z_L,
            "z_right":    z_R,
            "hemi_side":  hemi_side,
            "masked_ch":  masked_ch.detach().cpu().tolist(),
        }
