"""개념 도식 (데이터가 아님): 새 사용자의 특징이 '밀려 있고 (오프셋) 돌아가 있는 (회전)' 상황과,
중심화와 자극 시점 정렬이 각각 무엇을 고치는지.  총정리 PDF 의 '누구나 이해' 용.

    python make_fig_concept.py  → figs/fig_concept.png / .pdf
"""
from __future__ import annotations

import numpy as np

import figstyle as S

S.use()
plt = S.plt
rng = np.random.default_rng(3)
COLS = ("#c1553b", "#2c5f8a", "#3f7d5a")          # 부정 · 중립 · 긍정
NAMES = ("negative", "neutral", "positive")
P = np.array([[np.cos(a), np.sin(a)] for a in np.deg2rad([90, 210, 330])]) * 1.0   # 학습 prototype


def cloud(center, n=40, sd=0.28):
    return center + rng.normal(0, sd, size=(n, 2))


def rot(theta):
    t = np.deg2rad(theta)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


R, SHIFT = rot(55), np.array([1.6, 1.1])
base = [cloud(p) for p in P]
raw = [c @ R.T + SHIFT for c in base]
mu = np.mean(np.concatenate(raw), axis=0)
cen = [c - mu for c in raw]
ali = [c @ R for c in cen]                         # 회전을 되돌림 (도식)

panels = (("(a) training subjects", base, "class prototypes $P_c$ (stars)"),
          ("(b) new user, raw features", raw, "shifted AND rotated"),
          ("(c) after centering", cen, "origin fixed; directions still rotated"),
          ("(d) after stimulus-locked alignment", ali, "direction fixed too"))
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.7))
for ax, (title, clouds, sub) in zip(axes, panels):
    for c, col in zip(clouds, COLS):
        ax.scatter(c[:, 0], c[:, 1], s=5, color=col, alpha=0.55, lw=0)
    ax.scatter(P[:, 0], P[:, 1], marker="*", s=110, color=[*COLS], edgecolor="white", lw=0.6, zorder=5)
    ax.axhline(0, color=S.FAINT, lw=0.6, zorder=0); ax.axvline(0, color=S.FAINT, lw=0.6, zorder=0)
    ax.set_xlim(-2.0, 3.4); ax.set_ylim(-1.9, 3.0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, fontsize=7.8, loc="left")
    ax.text(-1.95, -1.85, sub, fontsize=6.3, color=S.GREY, va="bottom")
axes[0].legend([plt.Line2D([], [], ls="", marker="o", color=c, ms=4) for c in COLS], NAMES, fontsize=6.0,
               loc="upper right", handletextpad=0.2, borderaxespad=0.1)
fig.text(0.5, -0.03, "schematic, not data:  centering removes the shift (offset);  stimulus-locked alignment removes "
         "the remaining rotation", ha="center", fontsize=6.6, color=S.GREY)
fig.tight_layout(w_pad=0.6)
S.save(fig, "fig_concept")
