"""
LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Reference
---------
Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.

Idea
----
For a pretrained linear layer  y = W x + b,  freeze W and b and learn a
low-rank update instead:

    y = W x + b + (α / r) · B A x

where A ∈ R^{r × d_in}  (Gaussian init)
      B ∈ R^{d_out × r} (zero init  →  no effect at t=0)
      r ≪ min(d_in, d_out)

Only A and B are trained, so the trainable parameter count drops from
d_in × d_out to r × (d_in + d_out).  The low-rank path is initialised
to zero, so fine-tuning starts from the exact pretrained behaviour.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing ``nn.Linear`` with a trainable low-rank side path.

    Parameters
    ----------
    base     : nn.Linear  — the pretrained linear layer (frozen)
    r        : int        — rank of the adaptation  (typical: 4, 8, 16)
    alpha    : float      — scaling factor  (update magnitude ≈ α/r)
    dropout  : float      — dropout on the LoRA input
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")

        self.base = base
        # Freeze the pretrained weight and bias
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base.in_features
        out_features = base.out_features

        # A: (r, in) — Gaussian init ;  B: (out, r) — zero init
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)                                    # (..., out)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T # (..., out)
        return base_out + self.scaling * lora_out


def inject_lora(
    module: nn.Module,
    target_names: tuple[str, ...] = ("linear1", "linear2"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """Recursively replace matching ``nn.Linear`` submodules with ``LoRALinear``.

    Parameters
    ----------
    module       : nn.Module           — model to modify in place
    target_names : tuple[str, ...]     — attribute names to target
    r, alpha     : LoRA hyperparameters
    dropout      : dropout on LoRA input

    Returns
    -------
    n_replaced : int  — number of Linear layers wrapped with LoRA

    Notes
    -----
    Default targets ``linear1`` / ``linear2`` — the two ``nn.Linear`` layers
    of the FFN inside each TransformerEncoderLayer.  These make up the
    majority of the transformer parameters and are safe to wrap.

    ``self_attn.out_proj`` is NOT wrapped by default: PyTorch's
    ``nn.MultiheadAttention`` accesses ``self.out_proj.weight`` directly
    via its fast path, so replacing it with a wrapper breaks the forward.

    Q/K/V projections live inside a combined ``in_proj_weight`` Parameter
    (not an ``nn.Linear``) and are also not reachable by name.
    """
    n_replaced = 0
    for name, child in list(module.named_children()):
        if name in target_names and isinstance(child, nn.Linear):
            new_child = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, new_child)
            n_replaced += 1
        else:
            n_replaced += inject_lora(child, target_names, r, alpha, dropout)
    return n_replaced


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


# ── Transformer fast-path bypass ────────────────────────────────────────────
#
# PyTorch's ``nn.TransformerEncoderLayer`` has a fused fast path that directly
# accesses ``self.linear1.weight``, ``self.linear2.weight``, etc. as tensors
# (not as Module calls). When we wrap these Linear layers with LoRALinear,
# the fast path crashes with ``AttributeError: 'LoRALinear' object has no
# attribute 'weight'``.  The slow path calls ``self.linear1(x)`` which routes
# through ``LoRALinear.forward`` correctly.
#
# The helper below replaces each layer's forward with an explicit slow-path
# implementation so LoRA-wrapped submodules are always called via Python.


def _forced_slow_forward(
    self,
    src: torch.Tensor,
    src_mask: torch.Tensor | None = None,
    src_key_padding_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Slow-path forward for ``nn.TransformerEncoderLayer`` (Pre-LN / Post-LN)."""
    x = src
    if self.norm_first:
        x = x + self._sa_block(
            self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
        )
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(
            x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
        )
        x = self.norm2(x + self._ff_block(x))
    return x


def disable_transformer_fast_path(module: nn.Module) -> int:
    """Replace the ``forward`` of every ``nn.TransformerEncoderLayer`` inside
    *module* with a slow-path implementation compatible with LoRA wrappers.

    Returns
    -------
    n_patched : int  — number of layers patched
    """
    import types

    n_patched = 0
    for m in module.modules():
        if isinstance(m, nn.TransformerEncoderLayer):
            m.forward = types.MethodType(_forced_slow_forward, m)
            n_patched += 1
    return n_patched
