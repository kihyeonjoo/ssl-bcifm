"""CAFT 구조 그림 — 단순화판 (2026-10-05 사용자: "글이 너무 많아 직관적으로 받아들여지지 않는다").

핵심 메시지 하나: 학습 (위) 과 배포 (아래) 에서 ① 평균 빼기는 같은 연산이다 — 두 ① 상자를 같은 세로줄에 두고 점선으로 잇는다.
글 대신 모양: 배치 = 색칸 격자 (행 = 사람, 열 = 같은 순간, 색 = 감정), ② = 흩어진 화살표가 한 방향으로 모이는 작은 그림.
새로운 부분만 색 (① 주황, ② 초록), 나머지는 무채색.  라벨은 몇 단어만, 설명은 캡션으로.  A4 본문 폭 (7.2인치) 으로 그린다.
이전 상세판은 make_fig_caft_detailed.py → figs/fig_caft_detailed.png.

    python make_fig_caft.py  → figs/fig_caft.png / .pdf
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

INK, GREY, FAINT = "#1a1d21", "#7d838a", "#d4d9dd"
ORG, ORGBG = "#d4703f", "#fbe8dc"
GRN, GRNBG = "#3f7d5a", "#e3efe7"
ENC = "#eef0f2"
EMO = ("#8c6bb1", "#2c7fb8", "#7fbf7b", "#fdae61", "#d7301f")       # 감정 5개 (SEED-V 예)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(7.2, 3.13))
ax.set_xlim(0, 100); ax.set_ylim(2.4, 46); ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)     # 축이 그림 폭 전체를 쓰게 (기본 여백이면 78%)


def box(x0, x1, y0, y1, text="", fc="white", ec=INK, lw=0.9, fs=7.5, tc=INK, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.9",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4,
                fontweight="bold" if bold else "normal")


def ar(p0, p1, color=INK, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_width=2.2,head_length=3.4", mutation_scale=1,
                                 lw=lw, color=color, ls=ls, zorder=2, shrinkA=0, shrinkB=0))


def encoder(y0, y1):
    d = 0.18 * (y1 - y0)
    ax.add_patch(Polygon([(27, y0), (35, y0 + d), (35, y1 - d), (27, y1)], closed=True, fc=ENC, ec=INK, lw=0.9,
                         zorder=3))
    ax.text(31, (y0 + y1) / 2, "LaBraM", ha="center", va="center", fontsize=7.8, zorder=4)


def cells(y, h, rows, hl=None):
    """감정 색칸: 15열 (감정 5 × 3).  rows 줄.  hl = 강조할 열 (②)."""
    xs = []
    for c in range(15):
        x = 2.5 + c * 1.32 + (c // 3) * 0.35
        xs.append(x)
        for r in range(rows):
            ax.add_patch(Rectangle((x, y + (rows - 1 - r) * 2.75), 1.2, h, fc=EMO[c // 3], ec="none", alpha=0.9,
                                   zorder=3))
    if hl is not None:
        ax.add_patch(Rectangle((xs[hl] - 0.3, y - 0.35), 1.8, (rows - 1) * 2.75 + h + 0.7, fc="none", ec=GRN,
                               lw=1.5, zorder=4))
    return xs


# 두 줄의 띠와 제목
for y0, y1 in ((21.0, 45.8), (2.8, 19.4)):
    ax.add_patch(FancyBboxPatch((0.4, y0), 99.2, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2", fc="none",
                                ec=FAINT, lw=0.8, zorder=0))
ax.text(1.6, 44.4, "Fine-tuning (CAFT)", fontsize=8.6, fontweight="bold", va="top")
ax.text(22.5, 44.3, r"loss = CE + $\lambda\cdot L_{\rm align}$", fontsize=7.3, color=GREY, va="top")
ax.text(1.6, 18.0, "Deployment on a new user (unchanged)", fontsize=8.6, fontweight="bold", va="top")

# ── 위: 파인튜닝 ─────────────────────────────────────────────────────────
xs = cells(25.6, 2.4, 4, hl=7)
ax.text(xs[7] + 0.6, 37.25, "②", ha="center", va="bottom", fontsize=7.5, color=GRN, fontweight="bold")
ax.text(13.1, 24.4, "4 people × 15 shared moments\n(colour = emotion)", ha="center", va="top", fontsize=6.8,
        color=GREY, linespacing=1.15)
ar((24.2, 31.1), (27, 31.1))
encoder(25.4, 36.8)
ar((35, 31.1), (40, 31.1)); ax.text(37.5, 31.7, "$z$", ha="center", va="bottom", fontsize=8)
box(40, 50, 28.1, 34.1, r"$z-\mu$", fc=ORGBG, ec=ORG, lw=1.4, fs=10)
ax.text(45, 34.9, "① person's batch mean", ha="center", va="bottom", fontsize=6.9, color=ORG, fontweight="bold")
# u 가 head 와 ② 로 갈라진다
ax.plot([50, 52.6], [31.1, 31.1], color=INK, lw=1.0, zorder=2)
ax.text(51.3, 31.6, "$u$", ha="center", va="bottom", fontsize=8)
ar((52.6, 31.1), (56, 35.5)); ar((52.6, 31.1), (56, 26.6))
box(56, 64, 33.4, 37.6, "head")
ar((64, 35.5), (67, 35.5)); ax.text(67.6, 35.5, "CE", ha="left", va="center", fontsize=7.8)
box(56, 84, 23.0, 30.2, fc=GRNBG, ec=GRN, lw=1.4)
# ② 그림: 흩어진 화살표 → 한 방향
for k, dth in enumerate((-38, -12, 14, 40)):
    t = np.deg2rad(62 + dth)
    ax.annotate("", xy=(58.6 + 3.0 * np.cos(t), 24.3 + 3.0 * np.sin(t) * 1.25), xytext=(58.6, 24.3), zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.32", color=GRN, lw=1.0, shrinkA=0,
                                shrinkB=0))
ax.annotate("", xy=(65.2, 26.6), xytext=(62.9, 26.6), zorder=5,
            arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.35", color=INK, lw=0.8))
for k, dth in enumerate((-3, -1, 1, 3)):
    t = np.deg2rad(62 + dth)
    ax.annotate("", xy=(66.4 + 3.0 * np.cos(t), 24.3 + 3.0 * np.sin(t) * 1.25), xytext=(66.4, 24.3), zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.32", color=GRN, lw=1.0, shrinkA=0,
                                shrinkB=0))
ax.text(76.8, 26.6, "② same moment\n→ same direction", ha="center", va="center", fontsize=6.9, color=GRN,
        fontweight="bold", linespacing=1.2)
ar((84, 26.6), (87, 26.6)); ax.text(87.6, 26.6, r"$L_{\rm align}$", ha="left", va="center", fontsize=7.8)

# ── 아래: 배포 ───────────────────────────────────────────────────────────
cells(8.5, 2.4, 1)
ax.text(13.1, 7.6, "calibration block\n(1 clip per emotion)", ha="center", va="top", fontsize=6.8, color=GREY,
        linespacing=1.15)
ar((24.2, 9.7), (27, 9.7))
encoder(5.3, 14.1)
ar((35, 9.7), (40, 9.7)); ax.text(37.5, 10.3, "$z$", ha="center", va="bottom", fontsize=8)
box(40, 50, 6.7, 12.7, r"$z-\mu$", fc=ORGBG, ec=ORG, lw=1.4, fs=10)
ax.text(45, 5.9, "① calibration-block mean", ha="center", va="top", fontsize=6.9, color=ORG, fontweight="bold")
ar((50, 9.7), (56, 9.7))
box(56, 67, 7.4, 12.0, "prototypes")
ar((67, 9.7), (70, 9.7)); ax.text(70.6, 9.7, "emotion", ha="left", va="center", fontsize=7.8)

# 핵심: 두 ① 은 같은 연산
ax.plot([45, 45], [12.9, 27.9], color=ORG, lw=1.3, ls=(0, (3, 2)), zorder=1)
ax.text(46.3, 20.2, "same operation", ha="left", va="center", fontsize=7.6, color=ORG, fontweight="bold",
        bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)

for ext in ("png", "pdf"):
    fig.savefig(f"figs/fig_caft.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("[saved] figs/fig_caft.png / .pdf")
