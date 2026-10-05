"""
손설계 대조군: 로그 밴드 파워(DE 대용) 캐시.

미분 엔트로피(DE)는 가우시안 가정에서 0.5*log(2*pi*e*sigma^2) 이므로 **로그 밴드
파워의 단조 변환**이다.  SEED 문헌의 표준 기준선이 DE 이므로, 여기서는 밴드별
로그 파워를 그대로 쓴다 (상수배·상수합은 선형 분류기에 영향이 없다).

밴드는 `data/preprocessing.BANDS` 와 같다: theta 4-8, alpha 8-13, beta 13-30,
gamma 30-45.  창마다 (62채널 x 4밴드) = 248 차원.

입력은 EA 없이 `norm=scale100` — 손설계 특징에 EA 를 걸면 대조군의 의미가 흐려진다.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from data.preprocessing import BANDS

torch.set_num_threads(8)


def log_band_power(x, fs=200, n_fft=200, hop=40, eps=1e-10):
    """(n, C, L) -> (n, C*4) 로그 밴드 파워."""
    n, C, L = x.shape
    win = torch.hann_window(n_fft)
    z = torch.stft(x.reshape(-1, L), n_fft=n_fft, hop_length=hop,
                   window=win, return_complex=True, center=False)
    p = (z.real ** 2 + z.imag ** 2).mean(-1)          # (n*C, freq)
    freqs = torch.linspace(0, fs / 2, n_fft // 2 + 1)
    out = []
    for lo, hi in BANDS.values():
        m = (freqs >= lo) & (freqs < hi)
        out.append(torch.log(p[:, m].mean(-1) + eps))
    return torch.stack(out, -1).reshape(n, C * len(BANDS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache_de.npz")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    from data.seed_raw_dataset import SEEDRawDataset
    t0 = time.time()
    from dataset_config import get as _cfg
    _c = _cfg()
    ds = SEEDRawDataset(root=_c.root,
                        subjects=_c.subjects, sessions=_c.sessions,
                        segment_length=800, step=800, patch_size=200,
                        norm="scale100", ea=False)
    print(f"[data] {len(ds)} 창, 로드 {time.time()-t0:.1f}s", flush=True)

    raw = np.stack([s[0] for s in ds._segments]).astype(np.float32) / 100.0
    meta = np.array([[m[0], m[1], m[2]] for m in ds._seg_meta], np.int16)
    lab = np.array([s[1] for s in ds._segments], np.int8)

    feats = []
    t0 = time.time()
    for i in range(0, len(raw), args.batch):
        feats.append(log_band_power(torch.from_numpy(raw[i:i + args.batch])).numpy())
        if i % (args.batch * 20) == 0:
            print(f"  {i}/{len(raw)}  {time.time()-t0:.0f}s", flush=True)
    F = np.concatenate(feats).astype(np.float32)
    np.savez(args.out, cls=F, meta=meta, lab=lab)
    print(f"[저장] {args.out}  {F.shape}  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
