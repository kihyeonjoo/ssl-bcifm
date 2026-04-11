"""
Auxiliary pretext task: Temporal Delta Asymmetry Prediction (ℒ_aux)
===================================================================

Procedure
---------
1. Sample two time segments (t₁, t₂) from the same trial.
2. Encode both with the shared encoder E_φ to get z_L(t), z_R(t).
3. Compute asymmetry vectors:  a(t) = z_L(t) − z_R(t)
4. Compute temporal delta:     Δa   = a(t₂) − a(t₁)
5. Predictor P_θ maps the t₁ representation to Δâ.
6. Loss:
     ℒ_aux = MSE(Δa, Δâ) + λ_cos · (1 − cos_sim(Δa, Δâ))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import DualStreamEncoder


class AsymmetryDeltaPredictor(nn.Module):
    """MLP predictor P_θ:  z_joint(t₁) → Δâ.

    Input  : z_joint(t₁) = [z_L(t₁); z_R(t₁)]  ∈ R^{2·d_model}
    Output : Δâ  ∈ R^{d_model}
    """

    def __init__(self, d_model: int = 256, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, z_joint_t1: torch.Tensor) -> torch.Tensor:
        return self.net(z_joint_t1)  # (B, d_model)


class TemporalDeltaAsymmetry(nn.Module):
    """Auxiliary task: predict how hemispheric asymmetry changes across time.

    Parameters
    ----------
    encoder     : DualStreamEncoder  — shared E_φ (same instance as main task)
    d_model     : int                — encoder embedding dimension
    hidden_dim  : int | None         — predictor hidden size (default 2·d_model)
    lambda_cos  : float              — weight for cosine similarity loss term

    Forward
    -------
    left_t1, right_t1 : (B, C_h, n_bands, T)   — hemisphere pair at time t₁
    left_t2, right_t2 : (B, C_h, n_bands, T)   — hemisphere pair at time t₂

    Returns (dict)
    -------
    loss       : scalar Tensor        — ℒ_aux
    loss_mse   : scalar Tensor        — MSE component
    loss_cos   : scalar Tensor        — cosine similarity component
    delta_a    : (B, d_model)         — ground-truth Δa
    delta_a_hat: (B, d_model)         — predicted Δâ
    """

    def __init__(
        self,
        encoder: DualStreamEncoder,
        d_model: int = 256,
        hidden_dim: int | None = None,
        lambda_cos: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = AsymmetryDeltaPredictor(d_model=d_model, hidden_dim=hidden_dim)
        self.lambda_cos = lambda_cos

    def forward(
        self,
        left_t1: torch.Tensor,
        right_t1: torch.Tensor,
        left_t2: torch.Tensor,
        right_t2: torch.Tensor,
    ) -> dict[str, object]:

        # ── 1. Encode both time segments with shared E_φ ──────────────────────
        z_L_t1, z_R_t1, z_joint_t1 = self.encoder(left_t1, right_t1)
        z_L_t2, z_R_t2, _          = self.encoder(left_t2, right_t2)
        # shapes: z_L, z_R (B, d_model),  z_joint (B, 2·d_model)

        # ── 2. Asymmetry vectors & temporal delta ─────────────────────────────
        a_t1 = z_L_t1 - z_R_t1              # (B, d_model)
        a_t2 = z_L_t2 - z_R_t2              # (B, d_model)
        delta_a = a_t2 - a_t1               # (B, d_model)  — ground truth

        # ── 3. Predict Δa from t₁ representation ─────────────────────────────
        delta_a_hat = self.predictor(z_joint_t1)  # (B, d_model)

        # ── 4. ℒ_aux = MSE + λ_cos · (1 − cos_sim) ──────────────────────────
        loss_mse = F.mse_loss(delta_a_hat, delta_a.detach())
        loss_cos = 1.0 - F.cosine_similarity(delta_a_hat, delta_a.detach(), dim=-1).mean()
        loss = loss_mse + self.lambda_cos * loss_cos

        return {
            "loss":        loss,
            "loss_mse":    loss_mse,
            "loss_cos":    loss_cos,
            "delta_a":     delta_a,
            "delta_a_hat": delta_a_hat,
        }
