"""논문 figure 공통 스타일.

규칙 (좋은 AI 논문 figure 의 관례)
  - 축은 아래·왼쪽만, 눈금은 바깥쪽 짧게.  격자는 쓰더라도 아주 흐리게.
  - 범례보다 **직접 라벨**을 쓴다.  색은 의미를 가질 때만 쓴다.
  - 발견은 그림 안에 한 문장으로 적는다 (캡션을 안 읽어도 읽히게).
  - 모든 figure 가 같은 팔레트를 쓴다 — 패널을 섞어 봐도 일관되게.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK   = "#1a1d21"
GREY  = "#8a9097"
FAINT = "#d9dde1"
ACC   = "#d4703f"      # 제안 (중심화)
BLUE  = "#2c5f8a"      # SEED
RED   = "#c1553b"      # SEED-V
GREEN = "#3f7d5a"

PARAMS = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 8.0,
    "axes.linewidth": 0.8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
}


def use():
    plt.rcParams.update(PARAMS)


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def boot_ci(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(np.asarray(d, float), size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"figs/{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[saved] figs/{stem}.png / .pdf")
