"""
Channel layout and STFT-based frequency band feature extraction for SEED EEG.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# ── SEED 62-channel order (10-20 system) ─────────────────────────────────────
SEED_CH_NAMES: List[str] = [
    "FP1",  "FPZ",  "FP2",
    "AF3",  "AF4",
    "F7",  "F5",  "F3",  "F1",  "FZ",  "F2",  "F4",  "F6",  "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7",  "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",  "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7",  "P5",  "P3",  "P1",  "PZ",  "P2",  "P4",  "P6",  "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1",  "OZ",  "O2",  "CB2",
]  # 62 channels

# ── Hemisphere partitioning ───────────────────────────────────────────────────
# Midline electrodes (z-suffix) are discarded for asymmetry-based pretraining.
MIDLINE_CH: frozenset = frozenset([
    "FPZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ",
])

LEFT_CH: List[str] = [
    "FP1", "AF3",
    "F7",  "F5",  "F3",  "F1",
    "FT7", "FC5", "FC3", "FC1",
    "T7",  "C5",  "C3",  "C1",
    "TP7", "CP5", "CP3", "CP1",
    "P7",  "P5",  "P3",  "P1",
    "PO7", "PO5", "PO3",
    "CB1", "O1",
]  # 27 channels

RIGHT_CH: List[str] = [
    "FP2", "AF4",
    "F8",  "F6",  "F4",  "F2",
    "FT8", "FC6", "FC4", "FC2",
    "T8",  "C6",  "C4",  "C2",
    "TP8", "CP6", "CP4", "CP2",
    "P8",  "P6",  "P4",  "P2",
    "PO8", "PO6", "PO4",
    "CB2", "O2",
]  # 27 channels — mirror of LEFT_CH

assert len(LEFT_CH) == len(RIGHT_CH) == 27
assert len(LEFT_CH) + len(RIGHT_CH) + len(MIDLINE_CH) == 62

# Pre-compute integer indices into the raw (62,) channel axis
_ch_to_idx: Dict[str, int] = {ch: i for i, ch in enumerate(SEED_CH_NAMES)}
LEFT_IDX:  List[int] = [_ch_to_idx[ch] for ch in LEFT_CH]
RIGHT_IDX: List[int] = [_ch_to_idx[ch] for ch in RIGHT_CH]

# ── Frequency band definitions (Hz) ──────────────────────────────────────────
# lo inclusive, hi exclusive  →  matches neuroscience convention
BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
BAND_ORDER: List[str] = ["theta", "alpha", "beta", "gamma"]  # fixed index 0-3


class BandSTFT(nn.Module):
    """Convert a raw EEG segment into per-band temporal power envelopes.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.  SEED uses 200 Hz.
    n_fft : int
        FFT window length (samples).
        With fs=200 and n_fft=200 we get exactly 1 Hz per frequency bin,
        making band boundary mapping trivial.
    hop_length : int
        STFT hop in samples.  Default 40 → 0.2 s time step (80 % overlap).

    Input
    -----
    eeg : Tensor (C, L)   — float32, C channels, L time samples

    Output
    ------
    Tensor (C, 4, T_frames)
        axis 0 : channels  (same C as input)
        axis 1 : bands     [theta, alpha, beta, gamma]
        axis 2 : time frames after STFT hop
    """

    def __init__(self, fs: int = 200, n_fft: int = 200, hop_length: int = 40):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Hann window — register so .to(device) moves it automatically
        self.register_buffer("window", torch.hann_window(n_fft))

        # Frequency axis: 0, fs/n_fft, 2*fs/n_fft, …, fs/2
        # Shape: (n_fft // 2 + 1,)  →  101 bins for n_fft=200, fs=200
        freqs = torch.linspace(0, fs / 2, n_fft // 2 + 1)

        # Boolean mask per band — register as non-trainable buffers
        for name, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            self.register_buffer(f"mask_{name}", mask)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg : Tensor (C, L)

        Returns
        -------
        Tensor (C, 4, T_frames)
        """
        C, L = eeg.shape

        # Vectorised STFT over all C channels at once
        # torch.stft supports batch dim (C, L) since PyTorch ≥ 1.9
        spec = torch.stft(
            eeg,                          # (C, L)
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )  # → (C, F, T_frames),  F = n_fft // 2 + 1

        power = torch.log1p(spec.abs().pow(2))  # log(1+|X|²) — stabilise scale

        # Average power within each band  →  stack to (C, 4, T_frames)
        band_tensors = []
        for name in BAND_ORDER:
            mask = getattr(self, f"mask_{name}")          # (F,)
            band_power = power[:, mask, :].mean(dim=1)   # (C, T_frames)
            band_tensors.append(band_power)

        return torch.stack(band_tensors, dim=1)  # (C, 4, T_frames)
