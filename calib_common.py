"""
캘리브레이션 프로토콜 공용 부품.

S8·S9 와 이후의 M1·M2 가 **같은 창 선택 규칙과 같은 클립 추출**을 쓰도록 한곳에
모아 둔다.  여기가 갈라지면 방법 간 차이와 프로토콜 간 차이가 섞인다.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from dataset_config import get as _cfg

N_CLS = _cfg().n_classes
FS, SEG = _cfg().fs, _cfg().seg
SEC_PER_WIN = SEG / FS              # 4.0 s — the granularity of every budget


def n_windows(seconds):
    """Seconds -> window count, rounded UP (2 s cannot be had; 4 s is the floor)."""
    return max(1, int(np.ceil(seconds / SEC_PER_WIN)))


def pick_from_clip(w, n, mode):
    """``n`` windows out of one clip's window indices ``w`` (time-ordered).

    mode 'prefix' — the clip's first n windows: what a recording actually
                    starts with, and what S8/S9 used.
    mode 'spread' — n windows evenly spaced over the whole clip.  A calibration
                    session CAN be built this way (sample briefly at intervals),
                    and it removes the confound that a prefix sits entirely in
                    the clip's opening seconds, where the emotional response has
                    not developed yet.
    """
    if n >= len(w):
        return list(w)
    if mode == "prefix":
        return list(w[:n])
    if mode == "spread":
        return [w[i] for i in np.linspace(0, len(w) - 1, n).astype(int)]
    raise ValueError(f"mode must be prefix|spread, got {mode!r}")


def clips_by_domain_label(cidx, lab, subject):
    """{((subj, sess), label): [clip key, ...]} for one subject."""
    by = defaultdict(list)
    for k in sorted(cidx):
        if k[0] == subject:
            by[((k[0], k[1]), int(lab[cidx[k][0]]))].append(k)
    return by


def draw_holdout(by, te_doms, rng, n_cls=N_CLS):
    """One held-out clip per emotion per session; returns (held, eval_keys_set).

    Consumes exactly ``len(te_doms) * n_cls`` integers from ``rng`` in a fixed
    order, so a run with the same seed draws the same clips as S8/S9 did.
    """
    held = {d: [by[(d, c)][rng.integers(len(by[(d, c)]))] for c in range(n_cls)]
            for d in te_doms}
    return held, {k for v in held.values() for k in v}


def calib_indices(held, cidx, te_doms, seconds, mode):
    """{domain: [window index, ...]} for a budget of ``seconds`` PER CLIP.

    ``seconds`` may be the string 'full' for whole clips, where 'prefix' and
    'spread' coincide.
    """
    out = {}
    for d in te_doms:
        idx = []
        for k in held[d]:
            w = cidx[k]
            n = len(w) if seconds == "full" else n_windows(float(seconds))
            idx.extend(pick_from_clip(w, n, mode))
        out[d] = idx
    return out


def cond_key(seconds, mode):
    """Canonical name.  'full' ignores the mode because the two coincide."""
    if seconds == "full":
        return "Tfull"
    return f"T{float(seconds):g}" + ("" if mode == "prefix" else "s")


def parse_cond(key):
    """'T20s' -> (20.0, 'spread');  'Tfull' -> ('full', 'prefix')."""
    if key == "Tfull":
        return "full", "prefix"
    mode = "spread" if key.endswith("s") else "prefix"
    return float(key[1:-1] if mode == "spread" else key[1:]), mode
