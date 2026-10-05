"""Figure 3: 클립 내 감정 신호의 시간 경과 — 두 데이터셋.

읽는 사람이 바로 알아야 할 것: 정확도가 **영상이 진행될수록 오른다**.  그래서
캘리브레이션 영상을 짧게 자르면 손해다.

클래스 여백은 두 번째 축으로 그리면 라벨이 충돌하고 축이 둘이라 읽기 어렵다 —
배수만 글로 적는다.  우연 수준도 축을 자르면 보이지 않으므로 글로 적는다.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import figstyle as S

S.use()
BINS = ["0-20초", "20-40초", "40-80초", "80-160초", "160-끝초"]
XLAB = ["0\u201320", "20\u201340", "40\u201380", "80\u2013160", "160+"]

SETS = [("SEED", "results/timecourse.npz", 1 / 3, S.BLUE, 15, (0.47, 0.745)),
        ("SEED-V", "results/seedv_timecourse.npz", 1 / 5, S.RED, 16,
         (0.248, 0.385))]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7),
                         gridspec_kw=dict(wspace=0.26))

for i, (ax, (name, path, chance, col, n, ylim)) in enumerate(zip(axes, SETS)):
    z = np.load(path)
    acc = np.array([z[f"acc__{b}"] for b in BINS])
    mar = np.array([z[f"marg__{b}"] for b in BINS])
    x = np.arange(len(BINS))

    S.despine(ax)
    lo = np.array([S.boot_ci(a)[0] for a in acc])
    hi = np.array([S.boot_ci(a)[1] for a in acc])
    ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0, zorder=1)
    # 개별 피험자를 아주 얇게 — 평균선 뒤의 분포가 보인다
    for j in range(acc.shape[1]):
        ax.plot(x, acc[:, j], color=col, lw=0.4, alpha=0.22, zorder=2)
    ax.plot(x, acc.mean(1), color=col, lw=2.2, marker="o", ms=5.2,
            mfc="white", mew=1.8, zorder=4, clip_on=False)

    d = acc[-1] - acc[0]
    pv = stats.wilcoxon(acc[-1], acc[0]).pvalue
    ax.set_xticks(x); ax.set_xticklabels(XLAB)
    ax.set_xlim(-0.38, len(BINS) - 0.62)
    ax.set_ylim(*ylim)
    ax.set_ylabel("window accuracy")
    ax.set_xlabel("time within clip  (s)")
    ax.set_title(f"({'ab'[i]})  {name}   ($n={n}$)", fontsize=8.8,
                 loc="left", pad=6, fontweight="bold")

    # 발견을 패널 안에 한 덩어리로 (빈 왼쪽 위)
    ax.text(0.03, 0.965,
            f"last vs. first bin:  $+{d.mean():.3f}$\n"
            f"{int((d>0).sum())}/{n} subjects,  $p={pv:.4f}$",
            transform=ax.transAxes, fontsize=7.3, color=col,
            va="top", ha="left", fontweight="bold", linespacing=1.6)
    ax.text(0.03, 0.035,
            f"class margin  {mar.mean(1)[0]:.3f} $\\rightarrow$ "
            f"{mar.mean(1)[-1]:.3f}   (${mar.mean(1)[-1]/mar.mean(1)[0]:.1f}"
            f"\\times$)\nchance = {chance:.3f}   (axis truncated)",
            transform=ax.transAxes, fontsize=6.8, color=S.GREY,
            va="bottom", ha="left", linespacing=1.6)

fig.text(0.012, -0.055,
         "Thick line: mean over subjects.  Thin lines: individual subjects.  "
         "Shaded: bootstrap 95% CI of the mean.",
         fontsize=6.8, color=S.GREY, ha="left", va="top")
S.save(fig, "fig_timecourse")
