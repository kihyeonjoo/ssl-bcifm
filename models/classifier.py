"""
Downstream emotion classifier with hemisphere-asymmetry fusion.

Fusion
------
  z = [ z_L ; z_R ; z_L − z_R ; z_L ⊙ z_R ]  ∈  R^{4·d_model}

The four components capture complementary information:
  - z_L, z_R          : hemisphere-specific representations
  - z_L − z_R         : signed asymmetry (lateralisation)
  - z_L ⊙ z_R        : interaction / co-activation patterns
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.encoder import DualStreamEncoder
from models.lora import (
    inject_lora,
    count_trainable_parameters,
    disable_transformer_fast_path,
)


class AsymmetryFusionClassifier(nn.Module):
    """Pretrained encoder E_φ → fusion → MLP → emotion class logits.

    Parameters
    ----------
    encoder   : DualStreamEncoder  — pretrained, optionally frozen
    d_model   : int                — per-hemisphere embedding dimension
    n_classes : int                — number of emotion classes (3 for SEED)
    dropout   : float              — classifier head dropout
    freeze_encoder : bool          — if True, encoder weights are frozen

    Forward
    -------
    left, right : (B, C_h, n_bands, T)

    Returns
    -------
    logits : (B, n_classes)
    """

    def __init__(
        self,
        encoder: DualStreamEncoder,
        d_model: int = 256,
        n_classes: int = 3,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.use_lora = use_lora

        if use_lora:
            # Freeze all pretrained encoder params, then inject LoRA adapters.
            # Only LoRA A/B and the classification head will be trainable.
            for p in self.encoder.parameters():
                p.requires_grad = False
            n_wrapped = inject_lora(
                self.encoder,
                target_names=("linear1", "linear2"),
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            # Force slow-path forward so LoRA-wrapped linears are actually called.
            n_patched = disable_transformer_fast_path(self.encoder)
            trainable, total = count_trainable_parameters(self.encoder)
            print(
                f"  → LoRA injected: {n_wrapped} Linear layers wrapped "
                f"(r={lora_r}, α={lora_alpha}).  "
                f"Fast path disabled on {n_patched} TransformerEncoderLayers.  "
                f"Encoder trainable: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)"
            )
        elif freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Fusion dimension: z_L | z_R | z_L-z_R | z_L⊙z_R → 4·d_model
        # Compact 2-layer head (smaller random-init footprint for stable fine-tuning)
        fusion_dim = 4 * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        z_L, z_R, _ = self.encoder(left, right)  # (B, d_model) each

        # ── Asymmetry-aware fusion ────────────────────────────────────────
        z_diff = z_L - z_R        # signed asymmetry
        z_prod = z_L * z_R        # element-wise interaction
        z_fused = torch.cat([z_L, z_R, z_diff, z_prod], dim=-1)  # (B, 4·d_model)

        return self.head(z_fused)  # (B, n_classes)
