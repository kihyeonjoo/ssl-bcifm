"""
Main pretext task: Cross-Hemisphere Masked Band Prediction
==========================================================

Procedure per forward pass
───────────────────────────
1. BandSelectiveMasking   — randomly choose one hemisphere (L or R) and one
                            frequency band (θ/α/β/γ); apply Gaussian-smoothed
                            soft mask so band boundaries taper smoothly,
                            preventing spectral leakage artifacts.

2. Dual encoding (E_φ)    — both the masked and intact hemisphere pass through
                            the shared HemisphereEncoder, returning a CLS
                            embedding *and* the post-Transformer token sequence.

3. Decoding (D_ψ)         — CrossHemisphereDecoder reconstructs the masked band
                            using cross-attention:
                              Query   = masked-hemisphere sequence  z̃_seq
                              Key/Val = intact-hemisphere sequence  z_seq
                            This forces inter-hemisphere information flow.

4. ℒ_main                 — MSE between decoded output and the *original*
                            (pre-masking) values at the selected band only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import DualStreamEncoder


# ── 1. Gaussian-smoothed band mask ───────────────────────────────────────────

def _gaussian_band_weights(
    n_bands: int,
    b_masked: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Soft mask weights over the band dimension.

    Returns a (n_bands,) tensor where:
      - weight ≈ 0  at b_masked   (fully suppressed)
      - weight → 1  farther away  (gradually unmasked)

    The Gaussian taper at band boundaries prevents the model from exploiting
    sharp spectral transitions caused by the rectangular band-extraction
    windows in the upstream STFT.
    """
    idx = torch.arange(n_bands, dtype=torch.float32, device=device)
    # Gaussian centered at b_masked; peak = 1 at center
    gauss = torch.exp(-(idx - b_masked).pow(2) / (2.0 * sigma ** 2))
    return 1.0 - gauss  # invert: 0 = masked, 1 = untouched


class BandSelectiveMasking(nn.Module):
    """Randomly mask one band of one hemisphere with a Gaussian-smoothed boundary.

    On each call the module:
      1. Samples a hemisphere side  hemi_side ~ Uniform{0=left, 1=right}
      2. Samples a band index       b_masked  ~ Uniform{0,1,2,3}
      3. Builds a (n_bands,) soft mask via an inverted Gaussian centred at
         b_masked and multiplies it into the selected hemisphere's feature
         tensor.

    The Gaussian spread σ controls leakage suppression:
      - σ → 0 : hard binary mask (no boundary smoothing)
      - σ = 0.7 (default): adjacent band retains ~36 % energy; two steps
        away retains ~98 % — smooth rolloff without over-corrupting context.

    Parameters
    ----------
    n_bands : int    number of frequency bands (4)
    sigma   : float  Gaussian σ in band-index units (default 0.7)
    """

    def __init__(self, n_bands: int = 4, sigma: float = 0.7) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.sigma = sigma

    def forward(
        self,
        left: torch.Tensor,   # (B, C_h, n_bands, T)
        right: torch.Tensor,  # (B, C_h, n_bands, T)
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        """
        Returns
        -------
        x_masked     : (B, C_h, n_bands, T)  — hemisphere with masked band
        x_intact     : (B, C_h, n_bands, T)  — other hemisphere (untouched)
        b_masked     : int                    — masked band index
        hemi_side    : int                    — 0 = left masked, 1 = right
        mask_weights : (n_bands,)             — per-band Gaussian weights
                                                (0 at b_masked, ~1 far away)
        """
        device = left.device
        b_masked  = int(torch.randint(0, self.n_bands, ()).item())
        hemi_side = int(torch.randint(0, 2,            ()).item())

        weights   = _gaussian_band_weights(self.n_bands, b_masked, self.sigma, device)
        soft_mask = weights[None, None, :, None]  # broadcast → (1, 1, n_bands, 1)

        if hemi_side == 0:
            return left * soft_mask, right, b_masked, hemi_side, weights
        else:
            return right * soft_mask, left, b_masked, hemi_side, weights


# ── 2. Cross-hemisphere decoder D_ψ ─────────────────────────────────────────

class CrossHemisphereDecoder(nn.Module):
    """Reconstruct a masked frequency band via cross-attention.

    Architecture
    ------------
    Stacked TransformerDecoderLayers (Pre-LN, GELU):
      tgt    = z̃_seq  — masked-hemisphere token sequence  (B, T, d_model) [Query]
      memory = z_seq   — intact-hemisphere token sequence  (B, T, d_model) [Key / Value]

    Each layer performs:
      1. Masked self-attention on the query sequence
      2. Cross-attention from query to memory (intact hemisphere)
      3. Feed-forward sublayer

    A linear projection maps each output token to C_h values (one per channel),
    reconstructing the band-power envelope at every time frame.

    Parameters
    ----------
    d_model    : int   encoder embedding dimension
    n_channels : int   EEG channels per hemisphere (27)
    n_heads    : int   cross-attention heads (must divide d_model)
    n_layers   : int   decoder depth
    dropout    : float

    Input  : z̃_seq (B, T, d),  z_seq (B, T, d)
    Output : (B, C_h, T)  — reconstructed band-power values
    """

    def __init__(
        self,
        d_model: int = 256,
        n_channels: int = 27,
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
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )
        self.norm     = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_channels)

    def forward(
        self,
        z_masked_seq: torch.Tensor,  # (B, T, d_model) — Query
        z_intact_seq: torch.Tensor,  # (B, T, d_model) — Key / Value
    ) -> torch.Tensor:
        x = self.transformer_decoder(
            tgt=z_masked_seq,       # Query  : masked hemisphere
            memory=z_intact_seq,    # K / V  : intact hemisphere
        )                                  # (B, T, d_model)
        x = self.norm(x)
        x = self.out_proj(x)               # (B, T, C_h)
        return x.permute(0, 2, 1)          # (B, C_h, T)


# ── 3. Main pretext task ──────────────────────────────────────────────────────

class CrossHemisphereMaskedPrediction(nn.Module):
    """Main pretext task: Cross-Hemisphere Masked Band Prediction.

    Wires together BandSelectiveMasking → E_φ (shared encoder) → D_ψ (decoder)
    and computes ℒ_main.

    Loss
    ----
    ℒ_main = MSE( D_ψ(z̃_seq, z_seq),  x_orig[:, :, b_masked, :] )

    The MSE is computed *only* over the fully-masked band index b_masked,
    isolating the reconstruction signal from the Gaussian-tapered context
    bands that were only partially suppressed.

    Parameters
    ----------
    encoder    : DualStreamEncoder   E_φ with shared weights (already constructed)
    n_channels : int                 channels per hemisphere (27 for SEED)
    n_bands    : int                 number of frequency bands (4)
    d_model    : int                 encoder embedding dimension
    dec_heads  : int                 decoder cross-attention heads
    dec_layers : int                 decoder depth (lightweight: 2 layers)
    mask_sigma : float               Gaussian boundary smoothing (0.7)
    dropout    : float

    Forward
    -------
    left, right : Tensor (B, C_h, n_bands, T)

    Returns  (dict)
    -------
    loss      : scalar Tensor       ℒ_main (MSE, band-averaged over C_h × T)
    z_left    : (B, d_model)        left-hemisphere CLS embedding
    z_right   : (B, d_model)        right-hemisphere CLS embedding
    b_masked  : int                 which band was masked this step
    hemi_side : int                 0 = left was masked, 1 = right
    """

    def __init__(
        self,
        encoder: DualStreamEncoder,
        n_channels: int = 27,
        n_bands: int = 4,
        d_model: int = 256,
        dec_heads: int = 8,
        dec_layers: int = 2,
        mask_sigma: float = 0.7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.masking = BandSelectiveMasking(n_bands=n_bands, sigma=mask_sigma)
        self.decoder = CrossHemisphereDecoder(
            d_model=d_model,
            n_channels=n_channels,
            n_heads=dec_heads,
            n_layers=dec_layers,
            dropout=dropout,
        )

    def forward(
        self,
        left: torch.Tensor,   # (B, C_h, n_bands, T)
        right: torch.Tensor,  # (B, C_h, n_bands, T)
    ) -> dict[str, object]:

        # ── 1. Band-selective masking with Gaussian boundary smoothing ─────────
        x_masked, x_intact, b_masked, hemi_side, _ = self.masking(left, right)

        # Ground-truth band from the original (pre-masking) hemisphere
        orig = left if hemi_side == 0 else right
        target_band = orig[:, :, b_masked, :].contiguous()  # (B, C_h, T)

        # ── 2. Dual encoding  (shared E_φ — one parameter set, two forward passes)
        # return_sequence=True exposes the post-Transformer token sequence
        # alongside the CLS embedding needed for downstream tasks.
        z_masked_cls, z_masked_seq = self.encoder.encoder(x_masked, return_sequence=True)
        z_intact_cls, z_intact_seq = self.encoder.encoder(x_intact, return_sequence=True)
        # shapes: cls (B, d),   seq (B, T, d)

        # Re-assign to named left/right embeddings for caller convenience
        if hemi_side == 0:
            z_left, z_right = z_masked_cls, z_intact_cls
        else:
            z_left, z_right = z_intact_cls, z_masked_cls

        # ── 3. Cross-hemisphere decoding ───────────────────────────────────────
        # D_ψ: masked sequence = Query, intact sequence = Key / Value
        recon_band = self.decoder(z_masked_seq, z_intact_seq)  # (B, C_h, T)

        # ── 4. ℒ_main : MSE on the masked band only ───────────────────────────
        loss = F.mse_loss(recon_band, target_band)

        return {
            "loss":      loss,
            "z_left":    z_left,   # (B, d_model)
            "z_right":   z_right,  # (B, d_model)
            "b_masked":  b_masked,
            "hemi_side": hemi_side,
        }
