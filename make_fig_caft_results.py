"""Figure: CAFT 결과 — SEED-V 16명, 시드 3개씩 (2026-10-06).  make_fig_results.py 와 같은 형식 · 팔레트.

(a) 캘리브레이션 길이별 CAFT 이득 (CAFT ① − 일반 파인튜닝; 피험자마다 시드 3개 평균 → 짝 차이).  'none' = 캘리브레이션
    없이 (적응 없음).  중심화 경로와 그 위 자극 시점 정렬 (SLA) 경로.  띠 = 부트스트랩 95% CI, 별 = 피험자 단위 Wilcoxon.
(b) 피험자별 짝지은 변화 (영상 끝까지): 캘리브레이션 없음 · 중심화 · 중심화 + SLA 에서 일반 → CAFT.
(c) 시드 수준: 16명 평균 정확도 (중심화, 끝까지) 대 어긋남 c — 일반 시드 3개 · CAFT ① 시드 3개 · CAFT ①+② 시드 0.
색: 회색 = 일반 파인튜닝, 주황 = CAFT ① (제안, fig_caft 의 ① 과 같은 색), 초록 = 자극 시점 계열 (SLA · ②).
입력은 summarize_caft_seeds.py 와 같은 결과 파일 (caft_analyze.sh 출력).  검정은 보정 전 (분야 관례).
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import figstyle as S

S.use()
A3 = ("seedv_noea_seed0", "seedv_noea_seed1", "seedv_noea_seed2")          # 일반 파인튜닝
B3 = ("caftb_seedv_noea", "caftb_seedv_noea_s1", "caftb_seedv_noea_s2")   # CAFT ①
C1 = "caftc_seedv_noea"                                                   # CAFT ①+② (시드 0)
FN = {"cp": "calib_protocol_r10", "sla": "sla", "mis": "misalignment"}
KEYS = {"none": ("cp", "none__proto_clip"), "T20": ("sla", "T20__center__clip"), "T40": ("sla", "T40__center__clip"),
        "Tfull": ("sla", "Tfull__center__clip"), "T20_sla": ("sla", "T20__sla__clip"),
        "T40_sla": ("sla", "T40__sla__clip"), "Tfull_sla": ("sla", "Tfull__sla__clip"), "c": ("mis", "c")}

L, subj = {}, None
for t in A3 + B3 + (C1,):
    f = {k: np.load(f"results/{t}_{v}.npz") for k, v in FN.items()}
    for k in ("sla", "mis"):
        subj = f[k]["subjects"] if subj is None else subj
        assert np.array_equal(f[k]["subjects"], subj), (t, k)
    L[t] = {k: np.asarray(f[s][key], float) for k, (s, key) in KEYS.items()}
n = len(subj)


def avg(tags, key):
    """피험자마다 시드 평균 (16,)."""
    return np.mean([L[t][key] for t in tags], axis=0)


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def ptxt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.0, 1.3, 0.92], wspace=0.46))

# ══ (a) 캘리브레이션 길이별 CAFT 이득 ════════════════════════════════════
ax = axes[0]
S.despine(ax)
END = []
for keys, col, name, off in ((("none", "T20", "T40", "Tfull"), S.ACC, "centring", -0.05),
                             ((None, "T20_sla", "T40_sla", "Tfull_sla"), S.GREEN, "+ SLA", 0.05)):
    xx, mm, lo, hi, pp = [], [], [], [], []
    for i, k in enumerate(keys):
        if k is None:
            continue
        a, b = avg(A3, k), avg(B3, k)
        ci = S.boot_ci(b - a)
        xx.append(i + off); mm.append((b - a).mean()); lo.append(ci[0]); hi.append(ci[1])
        pp.append(stats.wilcoxon(b, a).pvalue)
    ax.vlines(xx, lo, hi, color=col, lw=1.0, alpha=0.75, zorder=2)          # 두 경로의 띠가 겹치면 탁해져 막대로
    ax.plot(xx, mm, color=col, lw=1.9, marker="o", ms=4.6, mfc="white", mew=1.6, zorder=3)
    for x_, h_, p_ in zip(xx, hi, pp):
        ax.text(x_, h_ + 0.001, star(p_), ha="center", va="bottom", fontsize=7.0, color=col)
    END.append([mm[-1], mm[-1], name, col])                  # [라벨 y, 끝까지 이득, 이름, 색]
# 오른쪽 라벨: 값 순서대로, 겹치지 않게 최소 간격 0.012
END.sort(key=lambda e: e[0])
for i in range(1, len(END)):
    END[i][0] = max(END[i][0], END[i - 1][0] + 0.012)
for y, v, name, col in END:
    ax.text(3.22, y, f"{name}\n{v:+.3f}", color=col, fontsize=6.6, va="center", ha="left", fontweight="bold",
            linespacing=1.2)
ax.axhline(0, color=S.FAINT, lw=0.8, zorder=0)
ax.set_xticks(range(4)); ax.set_xticklabels(["none", "20 s", "40 s", "full\nclip"])
ax.set_xlim(-0.3, 4.15)
ax.set_ylabel("Δ clip accuracy  (CAFT − standard)")
ax.set_xlabel("calibration per emotion")
ax.set_title("(a)  Gain needs calibration", fontsize=8.3, loc="left", pad=6, fontweight="bold")

# ══ (b) 피험자별 짝지은 변화 (영상 끝까지) ═══════════════════════════════
ax = axes[1]
S.despine(ax)
GROUPS = (("none", "none"), ("Tfull", "centring"), ("Tfull_sla", "+ SLA"))
for j, (k, name) in enumerate(GROUPS):
    a, b = avg(A3, k), avg(B3, k)
    x0, x1 = 2.2 * j, 2.2 * j + 1.0
    for p_, q_ in zip(a, b):
        c_ = S.ACC if q_ > p_ else S.GREY
        ax.plot([x0, x1], [p_, q_], color=c_, lw=0.8, alpha=0.65, zorder=2)
        ax.plot([x0], [p_], "o", ms=2.6, color="white", mec=S.GREY, mew=0.8, zorder=3)
        ax.plot([x1], [q_], "o", ms=2.6, color=c_, mec=c_, mew=0.8, zorder=3)
    ax.plot([x0, x1], [a.mean(), b.mean()], color=S.INK, lw=2.0, zorder=4)
    ax.plot([x0, x1], [a.mean(), b.mean()], "o", ms=4.5, color=S.INK, zorder=5)
    ax.hlines(0.2, x0 - 0.25, x1 + 0.25, color=S.INK, lw=0.6, ls=(0, (2, 2)), zorder=1)
    d = b - a
    p = stats.wilcoxon(b, a).pvalue
    ax.text((x0 + x1) / 2, 0.985, f"{name}\n{d.mean():+.3f}\n{int((d > 0).sum())}/{n} up\n{ptxt(p)}",
            ha="center", va="top", fontsize=6.4, color=S.ACC if p < 0.05 else S.GREY, fontweight="bold", linespacing=1.35)
ax.set_xticks([v for j in range(3) for v in (2.2 * j, 2.2 * j + 1.0)])
ax.set_xticklabels(["std", "CAFT"] * 3, fontsize=6.6)
ax.set_xlim(-0.45, 2.2 * 2 + 1.45)
ax.set_ylim(0.08, 1.0)
ax.set_yticks([0.2, 0.4, 0.6])
ax.set_ylabel("clip accuracy")
ax.set_title("(b)  13 of 16 subjects improve", fontsize=8.3, loc="left", pad=6, fontweight="bold")

# ══ (c) 시드 수준: 정확도 대 어긋남 c ═══════════════════════════════════
ax = axes[2]
S.despine(ax)
P = {}
for tags, key in ((A3, "std"), (B3, "caft")):
    P[key] = np.array([[L[t]["c"].mean(), L[t]["Tfull"].mean()] for t in tags])
ax.plot(*P["std"].T, "o", ms=5.5, mfc="white", mec=S.GREY, mew=1.4, ls="none", zorder=3)
ax.plot(*P["caft"].T, "o", ms=5.5, color=S.ACC, ls="none", zorder=3)
pc = (L[C1]["c"].mean(), L[C1]["Tfull"].mean())
ax.plot(*pc, "^", ms=6.2, color=S.GREEN, ls="none", zorder=3)
ma, mb = P["std"].mean(0), P["caft"].mean(0)
ax.annotate("", xy=mb, xytext=ma, arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.35", color=S.ACC,
                                                   lw=1.1, shrinkA=8, shrinkB=12), zorder=2)
ax.text(P["std"][:, 0].max() + 0.012, P["std"][:, 1].min() - 0.004, "standard\n3 seeds", color=S.GREY, fontsize=6.6,
        ha="left", va="top", fontweight="bold")
ax.text(P["caft"][:, 0].min() - 0.012, P["caft"][:, 1].max() + 0.006, "CAFT ①\n3 seeds", color=S.ACC, fontsize=6.6,
        ha="right", va="bottom", fontweight="bold")
ax.text(pc[0], pc[1] - 0.008, "CAFT ①+②\nseed 0", color=S.GREEN, fontsize=6.6, ha="center", va="top", fontweight="bold")
ax.set_xlim(0.26, 0.62)
ax.set_xticks([0.3, 0.4, 0.5, 0.6])
ax.set_ylim(0.325, 0.445)
ax.set_xlabel("misalignment c  (1 = aligned)")
ax.set_ylabel("clip accuracy, centred (full clip)")
ax.set_title("(c)  Robust across seeds", fontsize=8.3, loc="left", pad=6, fontweight="bold")

fig.text(0.012, -0.07,      # 줄마다 그림 폭 (7.4 in) 안에 들도록 — 길면 bbox_inches="tight" 가 캔버스를 넓힌다
         "SEED-V, 16 subjects.  Standard fine-tuning (std) vs CAFT ① (in-batch centring), 3 seeds each; per-subject values "
         "are averaged over seeds.\n"
         "(a) bars = bootstrap 95% CI of the paired gain; stars = Wilcoxon over subjects (* p < 0.05, ** p < 0.01, "
         "*** p < 0.001).\n"
         "(b) full clip; x/16 up = subjects improved; dashed = chance.  (c) one point per seed (16-subject mean); "
         "CAFT ①+② has seed 0 only.\n"
         "SLA = stimulus-locked alignment at deployment.",
         fontsize=6.4, color=S.GREY, ha="left", va="top")

S.save(fig, "fig_caft_results")
