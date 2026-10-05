"""Figure 4: 자극 라벨을 쓰면 얼마나 오르나 — 그리고 그 이득은 세션을 모은 덕이다.

(a) 방법별 라벨 곡선 (세션 통합).  규제를 걸면 이득이 사라진다는 것이 보여야 한다.
(b) 세 시나리오 대조.  **실전은 "같은 날" 이고 거기서 k=1 은 무효다.**
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import figstyle as S

S.use()
F = np.load("results/fewshot.npz")          # SEED, 세션 통합
Z = np.load("results/fewshot_sessions.npz") # SEED, 시나리오별
KS = [1, 2, 3, 4]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8),
                         gridspec_kw=dict(wspace=0.30))

# ══ (a) 방법별 라벨 곡선 ═══════════════════════════════════════════════
ax = axes[0]
S.despine(ax)
base = np.array([F[f"k{k}__a_none__win"] for k in KS])

METH = [("c_lr",    "logistic (no reg.)",      S.ACC,   2.2, "-"),
        ("d_head",  "head adaptation",         S.BLUE,  1.3, "-"),
        ("b_mix",   "prototype mixing",        S.GREEN, 1.3, "-"),
        ("c_lrreg", "logistic + reg.",         S.GREY,  1.3, (0, (2.4, 1.8)))]

for key, lab, col, lw, ls in METH:
    arr = np.array([F[f"k{k}__{m}__win"] for k, m in zip(KS, [key] * 4)])
    d = arr - base
    ax.plot(KS, d.mean(1), color=col, lw=lw, ls=ls, marker="o", ms=4.4,
            mfc="white", mew=1.4, zorder=3 if key == "c_lr" else 2)
    ax.text(4.12, d.mean(1)[-1], lab, fontsize=7.2, color=col,
            va="center", ha="left",
            fontweight="bold" if key == "c_lr" else "normal")

ax.axhline(0, color=S.FAINT, lw=1.0, zorder=0)
ax.set_xticks(KS)
ax.set_xlim(0.85, 6.1)
ax.set_xlabel("labelled clips per emotion  ($k$)")
ax.set_ylabel("gain over label-free  (window)")
ax.set_title("(a)  Stimulus labels help — if unregularised",
             fontsize=8.6, loc="left", pad=6, fontweight="bold")
# 시청 비용을 위쪽 축에 (라벨의 진짜 가격)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(KS)
ax2.set_xticklabels([f"{33.7*k:.0f}" for k in KS], fontsize=7.0,
                    color=S.GREY)
ax2.set_xlabel("viewing time  (min)", fontsize=7.4, color=S.GREY, labelpad=3)
ax2.tick_params(colors=S.GREY, length=2.5)
for sp in ("right", "left", "bottom"):
    ax2.spines[sp].set_visible(False)
ax2.spines["top"].set_color(S.GREY)

# ══ (b) 세 시나리오 ════════════════════════════════════════════════════
ax = axes[1]
S.despine(ax)
SCEN = [("sessions pooled", S.ACC, "-",
         lambda k: (F[f"k{k}__c_lr__win"], F[f"k{k}__a_none__win"])),
        ("same day only", S.BLUE, "-",
         lambda k: (Z[f"i__k{k}__c_lr__win"], Z[f"i__k{k}__a_none__win"])),
        ("enrol, reuse later", S.GREY, (0, (2.4, 1.8)),
         lambda k: (Z[f"ii__k{k}__full__c_lr__win"],
                    Z[f"ii__k{k}__full__a_none__win"]))]

for lab, col, ls, get in SCEN:
    m, lo, hi = [], [], []
    for k in KS:
        a, b = get(k)
        d = np.asarray(a, float) - np.asarray(b, float)
        m.append(d.mean()); l, h = S.boot_ci(d); lo.append(l); hi.append(h)
    # CI 를 셋 다 칠하면 서로 겹쳐 탁해진다 — 핵심 주장(같은 날)에만 둔다
    if lab.startswith("same"):
        ax.fill_between(KS, lo, hi, color=col, alpha=0.15, lw=0, zorder=1)
    ax.plot(KS, m, color=col, lw=2.2 if lab.startswith("same") else 1.4,
            ls=ls, marker="o", ms=4.6, mfc="white", mew=1.5, zorder=3)
    ax.text(4.12, m[-1], lab, fontsize=7.2, color=col, va="center",
            ha="left", fontweight="bold" if lab.startswith("same") else "normal")

ax.axhline(0, color=S.FAINT, lw=1.0, zorder=0)
ax.set_xticks(KS); ax.set_xlim(0.85, 6.3)
ax.set_xlabel("labelled clips per emotion  ($k$)")
ax.set_ylabel("gain over label-free  (window)")
ax.set_title("(b)  The gain needs sessions pooled", fontsize=8.6,
             loc="left", pad=6, fontweight="bold")

a, b = Z["i__k1__c_lr__win"], Z["i__k1__a_none__win"]
pv = stats.wilcoxon(a, b).pvalue
# 화살표를 쓰면 다른 선을 가로지른다 — 두 선 사이 빈 자리에 직접 둔다
ax.text(1.12, -0.0095,
        f"same day, $k=1$:  $+{(a-b).mean():.4f}$,  $p={pv:.2f}$",
        fontsize=7.0, color=S.BLUE, ha="left", va="center")
ax.set_ylim(-0.027, 0.088)

fig.text(0.012, -0.055,
         "SEED, $n=15$.  Shaded: bootstrap 95% CI of the paired difference.  "
         "Baseline in both panels: centring without any label.",
         fontsize=6.8, color=S.GREY, ha="left", va="top")
S.save(fig, "fig_fewshot")
