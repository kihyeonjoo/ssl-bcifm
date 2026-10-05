"""파이프라인 그림: 입력부터 추론까지 EA · 중심화 · SLA · L3 가 어디에 끼어드는가.

열마다 한 단계.  위 = 새 사용자의 추론 경로, 가운데 = 그 단계에 캘리브레이션이 공급하는 것,
아래 = 학습 때 만들어 두는 것.  색 = 쓰는 정보의 종류.  결과 요약은 상자 위에 둔다
(아래에서 올라오는 화살표와 겹치지 않게).

    python make_fig_pipeline.py  → figs/fig_pipeline.png / .pdf
"""
from __future__ import annotations

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, GREY, FAINT = "#1a1d21", "#7d838a", "#c9ced3"
ORG, ORGBG = "#d4703f", "#fbe8dc"      # 캘리브레이션 통계만 (라벨 없음)
GRN, GRNBG = "#3f7d5a", "#e3efe7"      # + 영상 정체성·시점
BLU, BLUBG = "#2c5f8a", "#e2ebf3"      # + 자극 라벨
TRN = "#f1f2f4"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4, "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(12.0, 7.4))
ax.set_xlim(0, 240); ax.set_ylim(0, 150); ax.axis("off")

# 열 (x0, x1)
COL = {"eeg": (3, 27), "ea": (31, 59), "enc": (63, 91), "cen": (95, 123), "sla": (127, 155),
       "cls": (159, 191), "dec": (195, 237)}
cx = {k: (a + b) / 2 for k, (a, b) in COL.items()}


def box(x0, x1, y, h, text, fs=7.4, bold=False, fc="white", ec=INK, lw=0.9, ls="-", tc=None, z=3):
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    ax.text((x0 + x1) / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or INK,
            zorder=z + 1, linespacing=1.4, fontweight="bold" if bold else "normal")


def ar(p0, p1, color=INK, ls="-", lw=1.0, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_width=2.0,head_length=3.2", mutation_scale=1,
                                 lw=lw, color=color, ls=ls, zorder=z, shrinkA=0.5, shrinkB=0.5))


def band(y, h, title, sub):
    ax.add_patch(FancyBboxPatch((0.5, y), 239, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="none", ec=FAINT, lw=0.8, zorder=0))
    for txt, dy, kw in ((title, 1.4, dict(fontsize=8.6, fontweight="bold")),
                        (sub, 6.0, dict(fontsize=6.6, color=GREY, style="italic"))):
        ax.text(2.5, y + h - dy, txt, ha="left", va="top", zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.6), **kw)


def note(k, y, text, color):
    ax.text(cx[k], y, text, ha="center", va="bottom", fontsize=6.4, color=color, linespacing=1.3, zorder=5)


# ════ 위: 추론 경로 ══════════════════════════════════════════════════════
band(84, 64, "Inference on a new user's session  (test time)",
     "each 4-s window goes left → right;  steps ①–④ use statistics computed ONCE from that user's calibration session;  "
     "②–④ change no network parameter")
Y, H = 100, 16
box(*COL["eeg"], Y, H, "EEG window\n62 ch × 4 s")
box(*COL["ea"], Y, H, "①  Euclidean alignment\n(input stage)\n$x' \\propto \\mathrm{diag}(\\bar R_d)^{-1/2}\\,x$", fs=6.8,
    fc=ORGBG, ec=ORG, ls="--", lw=1.1)
box(*COL["enc"], Y, H, "LaBraM\nencoder $f$\n→ class token $z$", fc=TRN)
box(*COL["cen"], Y, H, "②  Centering\n$z - \\mu_d$\nshift (origin)", fc=ORGBG, ec=ORG, lw=1.4, bold=True)
box(*COL["sla"], Y, H, "③  Stimulus-locked\nalignment  $(z - \\mu_d)\\,W$\nrotation (direction)", fs=6.5, fc=GRNBG, ec=GRN, lw=1.4,
    bold=True)
box(*COL["cls"], Y, H, "④  Prototype classifier\n$\\arg\\max_c\\,\\cos(\\cdot,\\,P_c)$\nlabel mixing: $P_c \\to P'_c$", fs=6.5)
box(*COL["dec"], Y + 8.6, 7.4, "window decision (per 4-s window)", fs=6.2, ec=BLU)
box(*COL["dec"], Y, 7.4, "clip decision (mean of all windows)", fs=6.2, ec=GRN)
for a, b in (("eeg", "ea"), ("ea", "enc"), ("enc", "cen"), ("cen", "sla"), ("sla", "cls"), ("cls", "dec")):
    ar((COL[a][1], Y + H / 2), (COL[b][0], Y + H / 2))
NY = Y + H + 2.0
note("ea", NY, "optional; changes the INPUT, so\nthe model must be trained with\nit too.  Realistic gain not\n"
     "significant → OFF in main pipeline", ORG)
note("enc", NY, "fine-tuned on the\nother subjects;\nfrozen at test time", INK)
# 수치는 결과 파일에서 (2026-10-05: SEED-IV 추가, prototype 혼합은 '끝까지만' 이 아니라 시청 시간에 따라 커진다)
_R = lambda f: np.load(f"results/{f}")
_d = lambda z, a, b: float((z[a] - z[b]).mean())
_SL = {d: _R(f"{p}sla_noea.npz") for d, p in (("SEED", ""), ("SEED-V", "seedv_"), ("SEED-IV", "seediv_"))}
_NONE = {d: _R(f"{p}calib_protocol_noea_r10.npz")["none__proto_clip"].mean()
         for d, p in (("SEED", ""), ("SEED-V", "seedv_"), ("SEED-IV", "seediv_"))}
_cen = {d: float(z["Tfull__center__clip"].mean() - _NONE[d]) for d, z in _SL.items()}
_sla = {d: _d(z, "Tfull__sla__clip", "Tfull__center__clip") for d, z in _SL.items()}
_l3 = {d: {t: _d(z, f"{t}__l3__clip", f"{t}__center__clip") for t in ("T40", "Tfull")} for d, z in _SL.items()}
note("cen", NY, "MAIN EFFECT\n(full-clip calibration)\n" + "\n".join(f"{d} {_cen[d]:+.3f}" for d in _SL), ORG)
note("sla", NY, "on top of centering,\ngrows with misalignment\n"
     f"SEED-V {_sla['SEED-V']:+.3f}\nSEED-IV {_sla['SEED-IV']:+.3f}\nSEED ≈ 0 ({_sla['SEED']:+.3f})", GRN)
note("cls", NY, "label-prototype mixing\n(stimulus labels), grows\nwith viewing time:\n"
     f"SEED-V {_l3['SEED-V']['T40']:+.3f} (40 s) → {_l3['SEED-V']['Tfull']:+.3f}\n"
     f"SEED-IV {_l3['SEED-IV']['Tfull']:+.3f} · SEED ≈ 0", BLU)
note("dec", NY, "both are reported;\nclip = one decision per film clip", INK)

# ════ 가운데: 캘리브레이션 ════════════════════════════════════════════════
band(42, 38, "Calibration session  (same new user, before use)",
     "the ONLY test-side data used:  1 film clip per emotion, first 20 s / 40 s / full;  excluded from evaluation")
CY, CH = 52, 14
box(*COL["eeg"], 47, 22, "calibration\nwindows\n(1 clip per emotion\n20 s / 40 s / full)", fs=6.6)
box(*COL["ea"], CY, CH, "$\\bar R_d$ : mean spatial\ncovariance", fs=6.6, fc=ORGBG, ec=ORG, ls="--")
box(*COL["cen"], CY, CH, "$\\mu_d$ : mean class token\nof calibration windows", fs=6.6, fc=ORGBG, ec=ORG)
box(*COL["sla"], CY, CH, "$W$ : Procrustes on pairs\n(window ↔ template at\nsame clip & second)", fs=6.3, fc=GRNBG,
    ec=GRN)
box(159, 179, CY, CH, "$M_c$ : mean\nper emotion", fs=6.5, fc=BLUBG, ec=BLU)
BUS = 48.5
ax.plot([27, 169], [BUS, BUS], color=GREY, lw=0.9, zorder=1)
for x in (cx["ea"], cx["cen"], 133.0, 169.0):
    ax.plot([x, x], [BUS, CY], color=GREY, lw=0.9, zorder=1)
for k, col, ls in (("ea", ORG, "--"), ("cen", ORG, "-"), ("sla", GRN, "-")):
    ar((cx[k], CY + CH), (cx[k], Y), color=col, ls=ls, lw=1.2)
ar((169, CY + CH), (169, Y), color=BLU, lw=1.2)
# 범례 (가운데 줄 오른쪽 빈칸)
for i, (col, bg, txt) in enumerate(((ORG, ORGBG, "calibration statistics only (no labels)"),
                                    (GRN, GRNBG, "+ which film & which second (no labels)"),
                                    (BLU, BLUBG, "+ stimulus labels of calibration clips"))):
    yy = 68 - i * 6.2
    ax.add_patch(FancyBboxPatch((193, yy), 4.2, 3.4, boxstyle="round,pad=0,rounding_size=0.4", fc=bg, ec=col,
                                lw=1.0))
    ax.text(199, yy + 1.7, txt, va="center", fontsize=6.2, color=col)
ax.text(193, 48.5, "dashed = optional, not in the main\n(fully realistic) pipeline", fontsize=6.2, color=GREY,
        va="center")

# ════ 아래: 학습 ══════════════════════════════════════════════════════════
band(2, 36, "Training  (once, on the other subjects — leave-one-subject-out)", "nothing here sees the test subject")
TY, TH = 8, 15
box(*COL["eeg"], TY, TH, "training\nsubjects (12–13,\n+2 for validation)", fs=6.8, fc=TRN)
box(31, 91, TY, TH, "fine-tune ALL LaBraM layers + head\n(layer-wise learning-rate decay 0.65:\nlow layers barely move)",
    fs=6.8, fc=TRN)
box(*COL["sla"], TY, TH, "templates $T_{a,k,t}$: mean\ntraining class token at\nclip $k$, second $t$", fs=6.3, fc=GRNBG,
    ec=GRN)
box(165, 237, TY, TH, "prototypes $P_c$: mean of centred\ntraining class tokens per emotion", fs=6.7, fc=TRN)
ar((COL["eeg"][1], TY + TH / 2), (31, TY + TH / 2))
ar((91, TY + TH / 2), (COL["sla"][0], TY + TH / 2))
# 템플릿과 prototype 은 둘 다 학습 피험자의 CLS 에서 따로 만든다 — 템플릿 아래로 돌아가는 화살표
ax.add_patch(FancyArrowPatch((80, TY), (200, TY), connectionstyle="arc3,rad=0.06",
                             arrowstyle="-|>,head_width=2.0,head_length=3.2", mutation_scale=1, lw=1.0,
                             color=INK, zorder=2, shrinkA=0.5, shrinkB=0.5))
ax.text(109, TY + TH / 2 + 1.2, "training class-token features", fontsize=6.2, color=GREY, va="bottom", ha="center")
ar((cx["enc"], TY + TH), (cx["enc"], Y), lw=1.1)                       # 인코더
ax.text(cx["enc"] + 1.2, 26.0, "encoder $f$", fontsize=6.4, color=INK, zorder=6,
        bbox=dict(fc="white", ec="none", pad=0.5))
ar((149, TY + TH), (149, CY), color=GRN, lw=1.1)                       # 템플릿 → W
ar((186, TY + TH), (186, Y), lw=1.1)                                   # prototype → 분류기
ax.text(187.2, 26.0, "$P_c$", fontsize=6.6, color=INK, zorder=6, bbox=dict(fc="white", ec="none", pad=0.5))

for ext in ("png", "pdf"):
    fig.savefig(f"figs/fig_pipeline.{ext}", dpi=300, bbox_inches="tight")
print("[saved] figs/fig_pipeline.png / .pdf")
