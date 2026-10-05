"""CAFT 개념도 — 무엇을 해결하나 (2026-10-05).

(a) 원래 특징: 사람마다 감정 덩어리 전체가 밀려 있다 (평행 이동).
(b) ① 사람마다 자기 평균을 빼면 밀림은 사라지지만, 감정 방향이 사람마다 돌아가 있다 (회전 — 어긋남 c 가 낮다).
    회색 화살표: ② 가 같은 영상의 같은 순간을 본 두 사람의 반응을 서로 당긴다.
(c) ② 뒤: 회전이 줄어 감정 방향이 사람과 무관해진다 (c 가 1 에 가깝다).
점은 설명용 합성 데이터다 (실제 특징이 아님).  사람 1 = 채운 원, 사람 2 = 빈 삼각형.
처음부터 A4 본문 폭 (7.4인치) 으로 그려 문서에 넣어도 글자가 6.5 pt 이상이 되게 한다.

    python make_fig_caft_concept.py  → figs/fig_caft_concept.png / .pdf
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Arc

import figstyle as S

S.use()
plt = S.plt
ANG = {"happy": 75.0, "sad": 195.0, "neutral": 315.0}           # 기준 감정 방향 (도)
COL = {"happy": "#d7301f", "sad": "#2c7fb8", "neutral": "#8c6bb1"}
OFF = {1: np.array([-1.30, 0.55]), 2: np.array([1.35, -0.55])}   # 사람별 밀림
TURN_BEFORE, TURN_AFTER = 65.0, 6.0                               # 사람 2 의 회전 (도): ② 전 / 뒤
rng = np.random.default_rng(3)
NOISE = {(s, e): rng.normal(0, 0.13, (6, 2)) for s in (1, 2) for e in ANG}


def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def person(s, turn, centred):
    """{감정: (7, 2) 점}.  centred 면 그 사람 전체 평균을 뺀다."""
    pts = {e: OFF[s] + rot(turn if s == 2 else 0.0) @ np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
           + NOISE[(s, e)] for e, a in ANG.items()}
    if centred:
        mu = np.concatenate(list(pts.values())).mean(0)
        pts = {e: p - mu for e, p in pts.items()}
    return pts


def scatter(ax, pts, s):
    for e, p in pts.items():
        if s == 1:
            ax.scatter(p[:, 0], p[:, 1], s=13, color=COL[e], alpha=0.85, lw=0, zorder=3)
        else:
            ax.scatter(p[:, 0], p[:, 1], s=16, marker="^", facecolor="white", edgecolor=COL[e], lw=0.9, zorder=3)


def arrows(ax, pts, s):
    for e, p in pts.items():
        m = p.mean(0)
        ax.annotate("", xy=m * 1.08, xytext=(0, 0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45", color=COL[e], lw=1.5,
                                    linestyle="-" if s == 1 else (0, (3, 2)), shrinkA=0, shrinkB=0))


def frame(ax, lim):
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.62), gridspec_kw=dict(width_ratios=[1.3, 1, 1], wspace=0.06))
LIM = ((-1.75, 1.75), (-1.75, 1.85))

# ── (a) 원래 특징: 밀림 ────────────────────────────────────────────────
ax = axes[0]
P1, P2 = person(1, TURN_BEFORE, False), person(2, TURN_BEFORE, False)
scatter(ax, P1, 1); scatter(ax, P2, 2)
m1, m2 = (np.concatenate(list(P.values())).mean(0) for P in (P1, P2))
for m in (m1, m2):
    ax.plot(*m, marker="+", ms=8, mew=1.5, color=S.INK, zorder=5)
ax.text(m1[0] + 0.12, m1[1] + 0.10, r"$\mu_1$", fontsize=7.4, ha="left", va="bottom")
ax.text(m2[0] + 0.12, m2[1] - 0.12, r"$\mu_2$", fontsize=7.4, ha="left", va="top")
ax.annotate("", xy=m2, xytext=m1, zorder=2,
            arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.4", color=S.GREY, lw=1.0, ls=(0, (3, 2)),
                            shrinkA=4, shrinkB=4))
ax.text(0.15, 0.98, "shift between\npeople / days", fontsize=6.8, color=S.GREY, ha="center", va="center",
        rotation=-21, linespacing=1.1)
# 범례 (비어 있는 왼쪽 아래)
x0, y0, dy = -2.62, -1.05, 0.27
ax.text(x0, y0, "\u25cf person 1", fontsize=6.8, ha="left", va="center", color=S.INK)
ax.text(x0, y0 - dy, "\u25b3 person 2", fontsize=6.8, ha="left", va="center", color=S.INK)
for i, e in enumerate(ANG):
    ax.text(x0, y0 - (2.3 + i) * dy, e, fontsize=6.8, ha="left", va="center", color=COL[e], fontweight="bold")
ax.text(x0, y0 - 5.6 * dy, "(illustrative points)", fontsize=6.0, ha="left", va="center", color=S.GREY)
frame(ax, ((-2.7, 2.75), (-2.1, 2.1)))
ax.set_title("(a) Raw features\nthe whole cloud is shifted", fontsize=7.6, loc="left", fontweight="bold",
             linespacing=1.3)

# ── (b) ① 평균 빼기 뒤: 회전이 남는다 ─────────────────────────────────
ax = axes[1]
C1, C2 = person(1, TURN_BEFORE, True), person(2, TURN_BEFORE, True)
ax.axhline(0, color=S.FAINT, lw=0.6, zorder=0); ax.axvline(0, color=S.FAINT, lw=0.6, zorder=0)
scatter(ax, C1, 1); scatter(ax, C2, 2); arrows(ax, C1, 1); arrows(ax, C2, 2)
a1 = np.degrees(np.arctan2(*C1["happy"].mean(0)[::-1])); a2 = np.degrees(np.arctan2(*C2["happy"].mean(0)[::-1]))
ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=a1, theta2=a2, color=S.INK, lw=0.9, zorder=5))
mid = np.deg2rad((a1 + a2) / 2)
ax.annotate("rotation (c low)", xy=(0.45 * np.cos(mid), 0.45 * np.sin(mid)), xytext=(-1.7, 1.55), fontsize=6.6,
            ha="left", va="center", color=S.INK, zorder=6,
            arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.6, shrinkA=2, shrinkB=0))
# ② 가 당기는 같은 순간의 두 반응 (슬픔 덩어리에서 하나씩)
p, q = C1["sad"][1], C2["sad"][3]
mpq = 0.5 * (p + q)
ax.plot([p[0], q[0]], [p[1], q[1]], color=S.INK, lw=0.6, ls=(0, (1, 1.5)), zorder=5)
for a in (p, q):
    ax.annotate("", xy=a + 0.7 * (mpq - a), xytext=a, zorder=6,
                arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.35", color=S.INK, lw=0.9))
ax.text(-1.7, -1.72, "② same clip & second:\npulled together", fontsize=6.4, color=S.INK, ha="left", va="bottom",
        linespacing=1.1)
frame(ax, LIM)
ax.set_title("(b) ① Centre each person\nshift gone, rotation remains", fontsize=7.6, loc="left", fontweight="bold",
             linespacing=1.3)

# ── (c) ② 뒤: 방향이 일치 ─────────────────────────────────────────────
ax = axes[2]
D1, D2 = person(1, TURN_AFTER, True), person(2, TURN_AFTER, True)
ax.axhline(0, color=S.FAINT, lw=0.6, zorder=0); ax.axvline(0, color=S.FAINT, lw=0.6, zorder=0)
scatter(ax, D1, 1); scatter(ax, D2, 2); arrows(ax, D1, 1); arrows(ax, D2, 2)
ax.text(-1.7, 1.78, "axes no longer\ndepend on the\nperson (c \u2248 1)", fontsize=6.6, ha="left", va="top",
        color=S.INK, linespacing=1.15)
frame(ax, LIM)
ax.set_title("(c) ② Align same moments\nrotation reduced", fontsize=7.6, loc="left", fontweight="bold",
             linespacing=1.3)
S.save(fig, "fig_caft_concept")
