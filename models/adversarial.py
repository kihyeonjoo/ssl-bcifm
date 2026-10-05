"""
Subject-adversarial head (E6) — does enforcing subject-invariance help?

The probe in ``probe_subject_invariance.py`` established a correlation: across
representations, the ones that give up subject identity also give up emotion.
A correlation over six non-independent points cannot settle the direction, so
this module supplies the intervention.  A discriminator is asked to name the
subject from the shared representation, and a gradient reversal layer makes the
backbone fight it.  Turning ``lambda_adv`` up is literally "enforce subject
invariance harder"; the emotion accuracy curve against it is the answer.

One design choice separates this from DANN / BiDANN / RGNN: the discriminator
sees **only the training subjects**, never the held-out one.  Those methods put
the test subject's unlabelled data into the discriminator, which makes them
transductive and mixes two effects — enforcing invariance, and adapting to the
target.  Restricting the discriminator to training subjects isolates the first
and keeps the setting calibration-free.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class _GradientReversal(torch.autograd.Function):
    """Identity forwards, negated gradient backwards."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


class SubjectDiscriminator(nn.Module):
    """Predicts which training subject a representation came from.

    Parameters
    ----------
    d_model      : representation width
    n_subjects   : number of *training* subjects (the held-out one is excluded)
    hidden       : width of the single hidden layer
    """

    def __init__(self, d_model: int, n_subjects: int,
                 hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_subjects),
        )

    def forward(self, z: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(grad_reverse(z, lambd))


class SubjectIndexer:
    """Maps subject ids to 0..n-1 over the training subjects only.

    A held-out subject must never reach the discriminator, so an id that was
    not registered raises instead of silently folding into some class.
    """

    def __init__(self, train_subjects: List[int]) -> None:
        self.ids = sorted(set(int(s) for s in train_subjects))
        self._to_idx: Dict[int, int] = {s: i for i, s in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def __call__(self, subjects: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(subjects)
        for i, s in enumerate(subjects.tolist()):
            idx = self._to_idx.get(int(s))
            if idx is None:
                raise KeyError(
                    f"subject {s} is not a training subject — the discriminator "
                    f"must not see held-out data (known: {self.ids})")
            out[i] = idx
        return out


def dann_lambda(epoch: int, total_epochs: int, max_lambda: float,
                warmup_frac: float = 0.2) -> float:
    """DANN's 2/(1+exp(-10p)) - 1 ramp, scaled to ``max_lambda``.

    Starting at full strength lets the discriminator dominate before the
    representation carries anything worth discriminating, so the weight is
    ramped in.  ``warmup_frac`` is the portion of training spent ramping.

    It must be a *small* portion.  DANN's original schedule ramps across the
    whole run, but the reported epoch here is chosen by validation F1 and that
    lands early, so a full-length ramp means the model that gets reported never
    saw the nominal lambda — a nominal 1.0 arm ran at an effective 0.20 on one
    fold, inside the 0.3 arm's range.  The sweep's x-axis stops meaning
    anything.  Ramping over the first fifth keeps lambda at nominal for the
    epochs that can actually be selected.
    """
    if max_lambda <= 0:
        return 0.0
    span = max(1.0, total_epochs * warmup_frac)
    p = min(1.0, epoch / span)
    return float(max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0))
