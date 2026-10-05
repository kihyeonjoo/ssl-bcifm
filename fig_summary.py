"""총정리 그림 (SEED, SEED-V, SEED-IV, no-EA 팔, clip accuracy) — 2026-10-05 SEED-IV 반영.

(a) 주 효과: 적응 없음 → 중심화 (클립 전체 캘리브레이션), 세 데이터셋
(b–d) 중심화 위의 추가 이득: CR · L3 · SLA − 중심화, 캘리브레이션 시간별, 95% 부트스트랩 CI.
      패널은 어긋남 c 가 작은 데이터셋부터 (SEED → SEED-IV → SEED-V) — 회전이 클수록 SLA 이득이 커지는 순서가 보이게.
      별표는 SLA − 중심화의 피험자 단위 Wilcoxon (보정 전, 분야 관례).

center·L3·SLA 는 results/{,seedv_,seediv_}sla_noea.npz, CR 은 results/{...}calib_rotation_noea.npz — 둘의 center 는
피험자별로 같다 (V5 · V6 회귀 검사).  '적응 없음' 은 calib_protocol r10 의 none.  c 는 results/{...}misalignment_noea.npz.

    python fig_summary.py  → figs/fig_summary.png / .pdf
"""
from __future__ import annotations

import numpy as np
from scipy import stats

import figstyle as S

S.use()
plt = S.plt
BUD = ("T20", "T40", "Tfull")
SETS = (("SEED", "", 3, ("1 min", "2 min", "~12 min")),
        ("SEED-V", "seedv_", 5, ("1.7 min", "3.3 min", "~14 min")),
        ("SEED-IV", "seediv_", 4, ("1.3 min", "2.7 min", "~9 min")))
METHODS = (("cr", "Label rotation", S.GREY, -0.12),
           ("l3", "Label-prototype mixing", S.BLUE, 0.0),
           ("sla", "Stimulus-locked alignment\n(no emotion labels)", S.GREEN, 0.12))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load(pre):
    sla = np.load(f"results/{pre}sla_noea.npz")
    cr = np.load(f"results/{pre}calib_rotation_noea.npz")
    none = np.load(f"results/{pre}calib_protocol_noea_r10.npz")["none__proto_clip"]
    c = float(np.load(f"results/{pre}misalignment_noea.npz")["c"].mean())
    return sla, cr, none, c


fig = plt.figure(figsize=(7.4, 5.3))
gs = fig.add_gridspec(2, 2, wspace=0.62, hspace=0.62)

# (a) 주 효과
ax = fig.add_subplot(gs[0, 0])
for i, (name, pre, k, _) in enumerate(SETS):
    sla, _, none, _ = load(pre)
    cen = sla["Tfull__center__clip"]
    for j, (v, col) in enumerate(((none.mean(), S.FAINT), (cen.mean(), S.ACC))):
        ax.bar(i + (j - 0.5) * 0.36, v, 0.34, color=col, edgecolor="none", zorder=2)
    ax.hlines(1 / k, i - 0.42, i + 0.42, color=S.INK, lw=0.7, ls=(0, (2, 2)), zorder=3)
    d = cen - none
    ax.text(i + 0.18, cen.mean() + 0.012, f"+{d.mean():.3f}\n{int((d > 0).sum())}/{len(d)}",
            ha="center", va="bottom", fontsize=6.3, color=S.ACC)
ax.set_xticks(range(len(SETS))); ax.set_xticklabels([s[0] for s in SETS])
ax.set_xlim(-0.6, len(SETS) - 0.4)
ax.set_ylim(0, 0.86); ax.set_ylabel("clip accuracy")
ax.text(0.98, 0.98, "no adaptation", color=S.GREY, fontsize=6.2, ha="right", va="top", transform=ax.transAxes)
ax.text(0.98, 0.91, "centering (full clip)", color=S.ACC, fontsize=6.2, ha="right", va="top", transform=ax.transAxes)
ax.text(0.98, 0.84, "dashed line = chance", color=S.INK, fontsize=6.0, ha="right", va="top", transform=ax.transAxes)
ax.set_title("(a) main effect: centering", fontsize=7.8, loc="left")
S.despine(ax)

# (b–d) 중심화 위의 추가 이득 — 어긋남 c 가 작은 데이터셋부터
order = sorted(SETS, key=lambda s: -load(s[1])[3])
slots = (gs[0, 1], gs[1, 0], gs[1, 1])
for p_i, ((name, pre, k, totals), slot) in enumerate(zip(order, slots)):
    ax = fig.add_subplot(slot)
    sla, cr, _, c = load(pre)
    ends = []
    for key, lab, col, off in METHODS:
        src = cr if key == "cr" else sla
        ds, lo, hi = [], [], []
        for t in BUD:
            d = src[f"{t}__{key}__clip"] - src[f"{t}__center__clip"]
            ci = S.boot_ci(d)
            ds.append(d.mean()); lo.append(d.mean() - ci[0]); hi.append(ci[1] - d.mean())
            if key == "sla":
                p = stats.wilcoxon(src[f"{t}__sla__clip"], src[f"{t}__center__clip"]).pvalue
                ax.text(BUD.index(t) + off, ci[1] + 0.003, stars(p), ha="center", fontsize=6.5, color=col)
        x = np.arange(3) + off
        lw = 1.8 if key == "sla" else 1.0
        ax.errorbar(x, ds, yerr=[lo, hi], color=col, lw=lw, marker="o", ms=3.0, capsize=0, elinewidth=0.8, zorder=3)
        ends.append([ds[-1], lab, col])
    # 오른쪽 끝 라벨 — 겹치지 않게 최소 간격을 둔다
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.013)
    for y, lab, col in ends:
        ax.text(2.32, y, lab, color=col, fontsize=6.0, va="center")
    ax.hlines(0, -0.35, 2.22, color=S.ACC, lw=1.0, zorder=1)   # 라벨 열(x ≥ 2.32)은 가로지르지 않게
    ax.text(2.32, min(-0.006, ends[0][0] - 0.013), "centering (= 0)", color=S.ACC, fontsize=6.0, va="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["20 s", "40 s", "full clip"], fontsize=6.6)
    ax.set_xlabel(f"calibration per emotion\n(total {' / '.join(totals)})", fontsize=6.4)
    ax.set_xlim(-0.35, 3.25); ax.set_ylim(-0.03, 0.095)
    ax.set_ylabel("Δ clip accuracy vs centering")
    ax.set_title(f"({'bcd'[p_i]}) {name} ({k} emotions), misalignment c = {c:.2f}", fontsize=7.6, loc="left")
    S.despine(ax)
fig.text(0.5, -0.01, "Panels (b)–(d) ordered by misalignment c (1 = test-session emotion directions match the training "
         "prototypes).  Stars: Wilcoxon over subjects, stimulus-locked alignment − centering.  Bars: bootstrap 95% CI.",
         ha="center", fontsize=6.0, color=S.GREY)
S.save(fig, "fig_summary")
