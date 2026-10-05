"""Figure 2: 주 결과 — 2026-10-05 개정 (세 데이터셋, 주 파이프라인 = 유클리드 정렬 없는 모델).

(a) 시청 시간 대 중심화 이득 (중심화 − 적응 없음, clip) — SEED · SEED-V · SEED-IV.
    x 를 **감정당 시청 시간** 으로 둔 이유: 비용은 창 수가 아니라 사용자가 화면 앞에 앉아 있는 시간이다.
(b) 피험자별 짝지은 변화 (영상 끝까지) — 평균이 아니라 **몇 명에게서 오르는가**.

이전 판 (10-03) 은 SEED 하나와 유클리드 정렬을 켠 모델을 주로 그렸다.  유클리드 정렬은 현실적 조건에서 이득이 유의하지
않아 주 파이프라인에서 뺐으므로, 여기서는 끈 모델 (results/{,seedv_,seediv_}calib_protocol_noea_r10.npz) 만 쓴다.
검정은 피험자 단위 Wilcoxon (보정 전, 분야 관례).
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import figstyle as S

S.use()
RED = "#c1553b"
SETS = (("SEED", "", 3, S.BLUE), ("SEED-V", "seedv_", 5, RED), ("SEED-IV", "seediv_", 4, S.GREEN))
TAG = ("T20", "T40", "Tfull")
Z = {n: np.load(f"results/{p}calib_protocol_noea_r10.npz") for n, p, _, _ in SETS}

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.0, 1.25], wspace=0.30))

# ══ (a) 예산별 중심화 이득 ═════════════════════════════════════════════
ax = axes[0]
S.despine(ax)
x = np.arange(3)
LBL = []
for j, (name, _, k, col) in enumerate(SETS):
    z = Z[name]
    d = np.array([z[f"{t}__proto_clip"] - z["none__proto_clip"] for t in TAG])      # (3, n)
    m = d.mean(1)
    ci = np.array([S.boot_ci(v) for v in d])
    off = (j - 1) * 0.08
    ax.fill_between(x + off, ci[:, 0], ci[:, 1], color=col, alpha=0.10, lw=0, zorder=1)
    ax.plot(x + off, m, color=col, lw=1.9, marker="o", ms=4.6, mfc="white", mew=1.6, zorder=3)
    share = m[0] / m[2]
    LBL.append([m[2], name, col, "20 s ≈ full clip" if share >= 0.95 else f"20 s already {share:.0%}"])
# 오른쪽 라벨: 값 순서대로, 겹치지 않게 최소 간격 0.019
LBL.sort(key=lambda e: e[0])
for i in range(1, len(LBL)):
    LBL[i][0] = max(LBL[i][0], LBL[i - 1][0] + 0.019)
for y, name, col, sub in LBL:
    v = Z[name]["Tfull__proto_clip"] - Z[name]["none__proto_clip"]
    ax.text(2.16, y + 0.003, f"{name}  {v.mean():+.3f}", color=col, fontsize=7.0, va="center", ha="left", fontweight="bold")
    ax.text(2.16, y - 0.0055, sub, color=col, fontsize=6.0, va="center", ha="left")
ax.axhline(0, color=S.FAINT, lw=0.8, zorder=0)
ax.set_xticks(x); ax.set_xticklabels(["20 s", "40 s", "full clip"])
ax.set_xlim(-0.25, 3.2)
ax.set_ylim(-0.005, 0.135)
ax.set_xlabel("calibration viewing time per emotion")
ax.set_ylabel("Δ clip accuracy  (centring − none)")
ax.set_title("(a)  Centring gain vs. viewing time", fontsize=8.6, loc="left", pad=6, fontweight="bold")

# ══ (b) 피험자별 짝지은 변화 (영상 끝까지) ═══════════════════════════════
ax = axes[1]
S.despine(ax)
for j, (name, _, k, col) in enumerate(SETS):
    z = Z[name]
    a, b = z["none__proto_clip"], z["Tfull__proto_clip"]
    x0, x1 = 2.2 * j, 2.2 * j + 1.0
    for p_, q_ in zip(a, b):
        c_ = col if q_ > p_ else S.GREY
        ax.plot([x0, x1], [p_, q_], color=c_, lw=0.8, alpha=0.65, zorder=2)
        ax.plot([x0], [p_], "o", ms=2.6, color="white", mec=S.GREY, mew=0.8, zorder=3)
        ax.plot([x1], [q_], "o", ms=2.6, color=c_, mec=c_, mew=0.8, zorder=3)
    ax.plot([x0, x1], [a.mean(), b.mean()], color=S.INK, lw=2.0, zorder=4)
    ax.plot([x0, x1], [a.mean(), b.mean()], "o", ms=4.5, color=S.INK, zorder=5)
    ax.hlines(1 / k, x0 - 0.25, x1 + 0.25, color=S.INK, lw=0.6, ls=(0, (2, 2)), zorder=1)
    d = b - a
    p = stats.wilcoxon(b, a).pvalue
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text((x0 + x1) / 2, 1.16, f"{name}\n{d.mean():+.3f}\n{int((d > 0).sum())}/{len(d)} subjects\n{ptxt}",
            ha="center", va="top", fontsize=6.6, color=col, fontweight="bold", linespacing=1.35)
ax.set_xticks([v for j in range(3) for v in (2.2 * j, 2.2 * j + 1.0)])
ax.set_xticklabels(["none", "centred"] * 3, fontsize=6.6)
ax.set_xlim(-0.45, 2.2 * 2 + 1.45)
ax.set_ylim(0.08, 1.17)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_ylabel("clip accuracy")
ax.set_title("(b)  Almost every subject improves  (full clip)", fontsize=8.6, loc="left", pad=6, fontweight="bold")

fig.text(0.012, -0.05,
         "Model fine-tuned without Euclidean alignment (the main pipeline).  (a) shaded = bootstrap 95% CI of the paired gain;\n"
         "label = gain with the full clip and the share already reached with 20 s.  (b) dashed = chance; Wilcoxon over subjects.",
         fontsize=6.4, color=S.GREY, ha="left", va="top")

S.save(fig, "fig_results")
