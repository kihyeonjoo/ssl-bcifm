"""CAFT 구조 그림 — 캘리브레이션 인지 파인튜닝 (2026-10-05).

위 = 학습 (CAFT): 한 배치 = 같은 세션의 학습 피험자 K 명 × 같은 (클립, 시점) M 곳 (감정별로 고르게).
     ① 배치 안 즉석 중심화 → head → 교차 엔트로피,  ② 같은 위치의 사람 간 일관성 (조건 ⓒ 만).
아래 = 배포 (바뀌지 않음): 새 사용자의 캘리브레이션 블록 평균을 빼고 prototype 분류 (+ 선택: 자극 시점 정렬).
색은 파이프라인 그림과 같다 (주황 = 캘리브레이션 통계, 초록 = 영상·시점 정보).

    python make_fig_caft_detailed.py  → figs/fig_caft_detailed.png / .pdf (2026-10-05 단순화 전 상세판, 문서에는 안 씀)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK, GREY, FAINT = "#1a1d21", "#7d838a", "#c9ced3"
ORG, ORGBG = "#d4703f", "#fbe8dc"
GRN, GRNBG = "#3f7d5a", "#e3efe7"
BLU, BLUBG = "#2c5f8a", "#e2ebf3"
TRN = "#f1f2f4"
EMO = ("#8c6bb1", "#2c7fb8", "#7fbf7b", "#fdae61", "#d7301f")       # 감정 5개 (SEED-V 예)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4, "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(12.0, 6.6))
ax.set_xlim(0, 240); ax.set_ylim(0, 132); ax.axis("off")


def box(x0, x1, y, h, text, fs=7.4, bold=False, fc="white", ec=INK, lw=0.9, ls="-", tc=None, z=3):
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    ax.text((x0 + x1) / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or INK,
            zorder=z + 1, linespacing=1.4, fontweight="bold" if bold else "normal")


def ar(p0, p1, color=INK, ls="-", lw=1.0, z=2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_width=2.0,head_length=3.2", mutation_scale=1,
                                 lw=lw, color=color, ls=ls, zorder=z, shrinkA=0.5, shrinkB=0.5,
                                 connectionstyle=f"arc3,rad={rad}"))


def band(y, h, title, sub):
    ax.add_patch(FancyBboxPatch((0.5, y), 239, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="none", ec=FAINT, lw=0.8, zorder=0))
    ax.text(2.5, y + h - 1.4, title, ha="left", va="top", fontsize=8.6, fontweight="bold", zorder=6,
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.text(2.5, y + h - 6.0, sub, ha="left", va="top", fontsize=6.6, color=GREY, style="italic", zorder=6,
            bbox=dict(fc="white", ec="none", pad=0.6))


# ════ 위: 학습 (CAFT) ══════════════════════════════════════════════════════
band(40, 90, "Training: calibration-aware fine-tuning (CAFT)",
     "every batch imitates the deployment calibration block;  all LaBraM layers + head are fine-tuned "
     "(same optimiser, schedule and epochs as the baseline)")

# 배치 격자: 피험자 4 × 위치 15  (라벨이 띠 안에 들어오게 오른쪽으로 둔다)
GX0, GY0, CW, CH = 20.0, 74.0, 3.6, 5.2
GXL = GX0 + 4.0                               # 격자 왼쪽 끝
subs = ("subject $a$", "subject $b$", "subject $c$", "subject $d$")
for r, sname in enumerate(subs):
    y = GY0 + (3 - r) * (CH + 1.0)
    ax.text(GXL - 1.5, y + CH / 2, sname, ha="right", va="center", fontsize=6.4, color=INK)
    for c in range(15):
        ax.add_patch(Rectangle((GXL + c * CW, y), CW - 0.5, CH, fc=EMO[c // 3], ec="white", lw=0.4,
                               alpha=0.85, zorder=3))
GXC = GXL + 7.5 * CW                          # 격자 가운데
hx = GXL + 7 * CW                             # 같은 위치 한 열 강조 (②)
ax.add_patch(Rectangle((hx - 0.35, GY0 - 0.6), CW + 0.2, 4 * (CH + 1.0) + 0.2, fc="none", ec=GRN, lw=1.6, zorder=4))
ax.text(GXC, GY0 - 2.0, "same (clip, second) across subjects → ②", ha="center", va="top", fontsize=6.2, color=GRN)
for e in range(5):
    ax.text(GXL + (3 * e + 1.5) * CW - 0.25, GY0 + 4 * (CH + 1.0) + 0.6, f"emotion {e + 1}", ha="center",
            va="bottom", fontsize=5.6, color=EMO[e])
ax.text(6.0, 112.5,
        "one batch = one session:  K = 4 training subjects × M = 15 (clip, second) positions,\n"
        "the same positions for every subject, 3 per emotion — i.e. a calibration block",
        ha="left", va="center", fontsize=6.5, color=INK, linespacing=1.35)
ax.text(GXC, 64.5, "60 windows of 4 s  (62 ch, 200 Hz)", ha="center", va="center", fontsize=6.3,
        color=GREY, style="italic")

# 인코더 → ① → head / ②
GXR = GXL + 15 * CW                           # 격자 오른쪽 끝
box(GXR + 6, GXR + 30, 78, 16, "LaBraM\nencoder $f$\n→ class token $z$", fc=TRN)
ar((GXR + 0.8, 86), (GXR + 6, 86))
box(GXR + 36, GXR + 80, 74, 24,
    "①  in-batch centering\n$\\tilde z_{s,p} = z_{s,p} - \\bar z_s$\n"
    "$\\bar z_s$: mean of subject $s$'s\n15 windows in this batch\n(current model → never stale)",
    fs=6.5, fc=ORGBG, ec=ORG, lw=1.4)
ar((GXR + 30, 86), (GXR + 36, 86))
box(GXR + 94, 236, 96, 14, "head $g$ → cross-entropy\n(emotion of each window)", fs=6.8)
ar((GXR + 80, 90), (GXR + 94, 101))
box(GXR + 94, 236, 56, 26,
    "②  stimulus-locked consistency  (variant c)\n"
    "$L_{\\mathrm{align}}$ = mean over positions $p$ of\n(1 − mean pairwise cosine of $\\tilde z_{a,p}, \\tilde z_{b,p}, \\ldots$)\n"
    "same movie moment, different subjects\n→ the same direction  (weight λ = 0.5)", fs=6.4, fc=GRNBG, ec=GRN, lw=1.4)
ar((GXR + 80, 82), (GXR + 94, 72))
ax.text((GXR + 94 + 236) / 2, 47.5, "loss = CE  (+ λ · L_align in variant c)\nvariant b = ① only,   variant c = ① + ②",
        ha="center", va="center", fontsize=6.6, color=INK, fontweight="bold", linespacing=1.35)

# ════ 아래: 배포 (바뀌지 않음) ═══════════════════════════════════════════
band(2, 34, "Deployment on a new user (unchanged)",
     "the same operation the network was trained with;  no gradient step at test time")
YB, HB = 6, 14
items = (("calibration block\n1 clip per emotion\n20 s / 40 s / full", "white", INK),
         ("$\\mu_d$ = mean class token\nof the block", ORGBG, ORG),
         ("centre every window\n$z - \\mu_d$", ORGBG, ORG),
         ("(optional) stimulus-locked\nalignment $W$", GRNBG, GRN),
         ("prototype classifier\n$\\arg\\max_c \\cos(\\cdot, P_c)$", "white", INK),
         ("window / clip\ndecision", "white", INK))
x = 6
w = 34
for i, (t, fc, ec) in enumerate(items):
    box(x, x + w, YB, HB, t, fs=6.5, fc=fc, ec=ec, lw=1.1, ls="--" if "optional" in t else "-")
    if i:
        ar((x - 5, YB + HB / 2), (x, YB + HB / 2))
    x += w + 5

for ext in ("png", "pdf"):
    fig.savefig(f"figs/fig_caft_detailed.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("[saved] figs/fig_caft_detailed.png / .pdf")
