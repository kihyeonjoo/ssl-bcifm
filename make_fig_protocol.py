"""Figure 2: 평가 프로토콜.

세 가지를 한 그림에서 답한다.
  (a) 누구로 학습하고 누구를 평가하나 — LOSO 바깥 고리
  (b) 창/클립 점수를 어떻게 모아 검정하나
  (c) 한 피험자의 클립 중 **무엇이 평가 집합인가** — 분석마다 다르고,
      이것이 "같은 조건인데 값이 다른" 이유다

배치: (a)(b) 를 위 행에, (c) 는 아래 행 전체.  (c) 는 오른쪽에 설명 꼬리가
붙으므로 가로 폭이 필요하다 — 셋을 한 줄에 두면 제목과 라벨이 부딪힌다.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK, GREY = "#1a1d21", "#8a9097"
ACC = "#d4703f"
TRAIN, VAL, TEST = "#eef1f4", "#b9c4cd", "#d4703f"
POOL, WIN = "#dce7f0", "#2c5f8a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6,
                     "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(10.0, 5.2))
ax.set_xlim(0, 200); ax.set_ylim(-1, 104); ax.axis("off")


def box(x, y, w, h, text="", fs=7.4, bold=False, fc="white", ec=INK, lw=0.9,
        tc=None, r=0.5, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc or INK, zorder=z + 1, linespacing=1.5,
                fontweight="bold" if bold else "normal")


def ar(p0, p1, color=INK, lw=0.9, hw=2.0, hl=3.2):
    ax.add_patch(FancyArrowPatch(p0, p1, zorder=2, lw=lw, color=color,
                 arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                 mutation_scale=1, shrinkA=0.5, shrinkB=0.5))


def swatch(x, y, fc, lab, s=3.4, fs=7.0):
    ax.add_patch(Rectangle((x, y), s, s, fc=fc, ec=INK, lw=0.6, zorder=3))
    ax.text(x + s + 1.3, y + s / 2, lab, fontsize=fs, va="center", ha="left")


# ════════ (a) LOSO ════════════════════════════════════════════════════
ax.text(1.0, 102.0, "(a)  Leave-one-subject-out", fontsize=9.0,
        fontweight="bold", va="top", ha="left")
N, CW, CG, X0 = 15, 3.4, 0.5, 13.0
ax.text(X0, 94.5, "subjects  $S_1 \\ldots S_{15}$", fontsize=7.4,
        ha="left", va="bottom")
for r, (ti, lab) in enumerate([(0, "fold 1"), (1, "fold 2"), (14, "fold 15")]):
    yy = 88.0 - r * 7.2
    if r == 2:
        ax.text(X0 + N * (CW + CG) / 2, yy + 6.0, "⋮", fontsize=13,
                color=GREY, ha="center", va="center")
    ax.text(X0 - 1.5, yy + 2.1, lab, fontsize=7.0, color=GREY,
            ha="right", va="center")
    for i in range(N):
        role = ("test" if i == ti else
                "val" if i in ((ti + 1) % N, (ti + 2) % N) else "train")
        ax.add_patch(Rectangle((X0 + i * (CW + CG), yy), CW, 4.2,
                     fc={"test": TEST, "val": VAL, "train": TRAIN}[role],
                     ec=INK, lw=0.55, zorder=3))
swatch(X0, 64.0, TEST, "test")
swatch(X0 + 16, 64.0, VAL, "val  (next 2, cyclic)")
swatch(X0 + 54, 64.0, TRAIN, "train")
ax.text(X0, 59.5,
        "every subject is test exactly once;  validation rotates — no RNG\n"
        "each fold is run at 3 seeds   →   $15\\times3 = 45$ runs\n"
        "the epoch is chosen on validation;  test selects nothing",
        fontsize=7.0, color=GREY, va="top", ha="left", linespacing=1.75,
        style="italic")

# ════════ (b) 집계 ════════════════════════════════════════════════════
ax.text(104.0, 102.0, "(b)  Aggregation and testing", fontsize=9.0,
        fontweight="bold", va="top", ha="left")
box(105.0, 88.0, 60.0, 8.5,
    "per run:   window accuracy  AND  clip accuracy", fs=7.2)
ar((135.0, 88.0), (135.0, 83.0))
box(105.0, 73.5, 60.0, 9.0,
    "average the 3 seeds  →  one value per subject", fs=7.2, bold=True)
ax.text(167.0, 78.0, "spread across seeds\n= noise floor", fontsize=6.8,
        color=GREY, ha="left", va="center", style="italic", linespacing=1.4)
ar((135.0, 73.5), (135.0, 68.5))
box(105.0, 57.0, 60.0, 11.0,
    "paired across subjects\nWilcoxon  +  bootstrap CI  +  wins\n"
    "$n=15$ (SEED),  $n=16$ (SEED-V),  $n=15$ (SEED-IV)",
    fs=7.1, ec=ACC, lw=1.5, tc=ACC)
ax.text(105.0, 54.0,
        "same checkpoint, aggregation changed  →  paired test alone\n"
        "different training runs  →  also a detectable-effect floor",
        fontsize=7.0, color=GREY, va="top", ha="left", linespacing=1.75,
        style="italic")

# ════════ (c) 평가 집합이 분석마다 다르다 ═════════════════════════════
ax.text(1.0, 43.0,
        "(c)  Which clips are scored — and that is why the same condition "
        "has different values",
        fontsize=9.0, fontweight="bold", va="top", ha="left")

BX, BW, BG = 44.0, 4.6, 0.55
VARIANTS = [
    ("all 15 clips scored", "S2, role analysis", [], "45", INK),
    ("1 clip per emotion held out as calibration", "ladder, S11, S12",
     [0, 1, 2], "36", ACC),
    ("1 per emotion scored, the rest is the label pool",
     "few-shot (S14, S15)", list(range(3, 15)), "9", WIN),
]
for r, (title, where, special, nev, col) in enumerate(VARIANTS):
    yy = 30.0 - r * 10.0
    ax.text(BX - 2.0, yy + 2.3, title, fontsize=7.2, ha="right", va="center")
    for i in range(15):
        fc = "white" if i not in special else (ACC if r == 1 else POOL)
        ax.add_patch(Rectangle((BX + i * (BW + BG), yy), BW, 4.6,
                     fc=fc, ec=INK, lw=0.6, zorder=3))
    ax.text(BX + 15 * (BW + BG) + 2.0, yy + 2.3,
            f"{nev} eval clips / subject", fontsize=7.2, color=col,
            ha="left", va="center", fontweight="bold")
    ax.text(BX + 15 * (BW + BG) + 36.0, yy + 2.3, where, fontsize=6.8,
            color=GREY, ha="left", va="center", style="italic")

ax.text(BX, 36.2, "one session shown;  the protocol repeats in all 3 sessions",
        fontsize=6.8, color=GREY, ha="left", va="bottom", style="italic")
# 범례는 왼쪽 설명 글(x<62)을 피해 오른쪽에 둔다
swatch(72.0, 6.0, "white", "scored")
swatch(100.0, 6.0, ACC, "calibration (builds $\\mu$)")
swatch(146.0, 6.0, POOL, "label pool (few-shot only)")

ax.text(1.0, 8.0,
        "A clip that builds $\\mu$ cannot also be scored.\n"
        "Primary metric follows this panel:  clip when\n"
        "$\\geq$36 clips are scored,  window when only 9\n"
        "— one clip would then move it by 0.111.",
        fontsize=7.0, color=INK, va="top", ha="left", linespacing=1.75)

fig.tight_layout(pad=0.2)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/fig_protocol.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
print("[saved] figs/fig_protocol.png / .pdf")
