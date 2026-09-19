"""
Temporal Delta Asymmetry (adapter version)
===========================================

Same formulation as ``tasks/temporal_delta_asymmetry.py`` but the adapter
pipeline receives *raw EEG* at t₁ and t₂ (LaBraM-format) instead of
pre-computed STFT hemispheres. The adapter internally runs LaBraM →
hemisphere split → shared transformer, producing z_L(t), z_R(t).

Loss
----
    a(t)   = z_L(t) - z_R(t)
    Δa     = a(t₂) - a(t₁)             (detached ground truth)
    Δâ     = P_θ([z_L(t₁); z_R(t₁)])
    ℒ_aux  = MSE(Δâ, Δa) + λ_cos · (1 - cos_sim(Δâ, Δa))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetryDeltaPredictor(nn.Module):
    """MLP predictor P_θ : z_joint(t₁) → Δâ."""

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
        return self.net(z_joint_t1)


class TemporalDeltaAsymmetryAdapter(nn.Module):
    """Auxiliary task using the LaBraM-based AsymmetryAdapter.

    Parameters
    ----------
    adapter     : AsymmetryAdapter   — dual-stream adapter (shared weights)
    d_model     : int
    hidden_dim  : int | None
    lambda_cos  : float
    """

    def __init__(
        self,
        adapter,
        d_model: int = 256,
        hidden_dim: int | None = None,
        lambda_cos: float = 0.5,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.predictor = AsymmetryDeltaPredictor(d_model=d_model, hidden_dim=hidden_dim)
        self.lambda_cos = lambda_cos

    def forward(self, eeg_t1: torch.Tensor, eeg_t2: torch.Tensor) -> dict:
        z_L_t1, z_R_t1, z_joint_t1 = self.adapter(eeg_t1)
        z_L_t2, z_R_t2, _          = self.adapter(eeg_t2)

        a_t1 = z_L_t1 - z_R_t1
        a_t2 = z_L_t2 - z_R_t2
        delta_a = a_t2 - a_t1                          # ground truth

        delta_a_hat = self.predictor(z_joint_t1)

        loss_mse = F.mse_loss(delta_a_hat, delta_a.detach())
        loss_cos = 1.0 - F.cosine_similarity(
            delta_a_hat, delta_a.detach(), dim=-1
        ).mean()
        loss = loss_mse + self.lambda_cos * loss_cos

        return {
            "loss":        loss,
            "loss_mse":    loss_mse,
            "loss_cos":    loss_cos,
            "delta_a":     delta_a,
            "delta_a_hat": delta_a_hat,
        }
