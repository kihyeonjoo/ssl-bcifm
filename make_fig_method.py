"""Figure 1: method overview (2행, figs/fig_architecture.png 스타일).

행 1  (a) 교차 피험자 fine-tuning   (b) 캘리브레이션: mu 를 만든다
행 2  (c) 추론과 채점 — **window 와 clip 이 어디서 갈리는지**

(c) 를 넣은 이유: 두 지표의 정의가 헷갈린다.  clip 은 "창을 몇 개까지 보느냐"
가 아니라 **영화 클립 한 편**이고, 그 클립의 창을 **전부** 평균한다.  숫자는
실제 로짓에서 가져왔다 (피험자1/세션1/클립11, 정답 neutral, 창 58개).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.path import Path

INK, GREY, FAINT = "#1a1d21", "#8a9097", "#c9ced3"
ACC, ACCBG = "#d4703f", "#fbe8dc"
WIN, CLIP = "#2c5f8a", "#3f7d5a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6,
                     "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(10.0, 6.0))
ax.set_xlim(0, 200); ax.set_ylim(-1, 122); ax.axis("off")


def box(x, y, w, h, text, fs=7.6, bold=False, fc="white", ec=INK, lw=0.9,
        tc=None, italic=False, r=0.6, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or INK, zorder=z + 1, linespacing=1.45,
            fontweight="bold" if bold else "normal",
            style="italic" if italic else "normal")


def group(x, y, w, h, title, ls="-", ec=INK, lw=0.9, tfs=7.8, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=0.8",
                 fc="none", ec=ec, lw=lw, ls=ls, zorder=1))
    ax.text(x + 1.6, y + h - 1.4, title, ha="left", va="top", fontsize=tfs,
            fontweight="bold", color=tc or INK, zorder=4)


def ar(p0=None, p1=None, ls="-", color=INK, lw=0.9, hw=2.0, hl=3.2,
       rad=None, path=None, z=2):
    kw = dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
              mutation_scale=1, lw=lw, color=color, ls=ls, zorder=z,
              shrinkA=0.5, shrinkB=0.5)
    if path is not None:
        ax.add_patch(FancyArrowPatch(path=path, **kw)); return
    if rad is not None:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))


def oper(x, y, sym, r=2.1, color=ACC, fs=10.5):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=1.4, zorder=5))
    ax.text(x, y + 0.05, sym, ha="center", va="center", fontsize=fs,
            color=color, zorder=6, fontweight="bold")


# ════════ 행 1 (a) 교차 피험자 fine-tuning ════════════════════════════
ax.text(1.0, 120.0, "(a)  Cross-subject fine-tuning", fontsize=9.0,
        fontweight="bold", va="top", ha="left")
box(1.0, 95.0, 17.0, 10.0, "Training\nsubjects\n(labelled)", fs=7.4)
ar((18.0, 100.0), (21.5, 100.0))
group(21.5, 91.0, 29.0, 18.0, "LaBraM encoder $f$")
box(23.0, 93.0, 11.5, 9.0, "patch\nembed", fs=7.2)
ar((34.5, 97.5), (37.0, 97.5))
box(37.0, 93.0, 11.5, 9.0, "12 ×\nTransf.", fs=7.2)
ar((50.5, 100.0), (54.5, 100.0))
box(54.5, 95.0, 13.0, 10.0, "linear\nhead", fs=7.4)
ar((67.5, 100.0), (71.5, 100.0))
box(71.5, 96.0, 12.0, 8.0, "CE loss", fs=7.4, italic=True)
ar((44.0, 91.0), (44.0, 85.0))
box(29.5, 75.0, 29.0, 9.5,
    "centre each training\ndomain by its own mean", fs=6.8)
ar((58.5, 79.8), (62.0, 79.8))
box(62.0, 75.0, 21.5, 9.5, "prototypes $P_c$\n(one per class)", fs=6.8, bold=True)
ax.text(44.0, 73.4, "epoch chosen on 2 held-out subjects",
        fontsize=6.6, color=GREY, ha="center", va="top", style="italic")

# ════════ 행 1 (b) 캘리브레이션 ═══════════════════════════════════════
ax.text(92.0, 120.0, "(b)  Calibration — estimate the domain mean",
        fontsize=9.0, fontweight="bold", va="top", ha="left")
group(92.0, 88.5, 72.0, 23.5, "once per test session", ls=(0, (3, 2)),
      ec=GREY, lw=1.0, tfs=7.2, tc=GREY)
box(94.0, 94.5, 28.0, 11.0,
    "ONE clip per emotion\nfirst $T$ s of each", fs=7.4)
ax.text(108.0, 93.9, "cost = viewing time   $n_{\\mathrm{cls}}\\!\\times\\!T$ s",
        fontsize=6.8, color=ACC, ha="center", va="top", fontweight="bold")
ar((122.0, 100.0), (126.0, 100.0))
box(126.0, 95.5, 11.0, 9.0, "$f$", fs=9.0)
ar((137.0, 100.0), (141.0, 100.0))
box(141.0, 94.5, 21.0, 11.0, "$\\mu=\\overline{z}_{\\mathrm{calib}}$", fs=9.0,
    bold=True, fc=ACCBG, ec=ACC, lw=1.5, tc=ACC)
ax.text(128.0, 86.6,
        "evaluation clips never enter $\\mu$   ·   no label from the subject is used",
        fontsize=6.7, color=GREY, ha="center", va="top", style="italic")

# ════════ 행 2 (c) 추론과 채점 ════════════════════════════════════════
ax.text(1.0, 64.0,
        "(c)  Inference and scoring — where window and clip differ",
        fontsize=9.0, fontweight="bold", va="top", ha="left")

# 평가 클립을 창 띠로
NW, W0, WW, WY, WH = 14, 2.0, 2.9, 46.0, 6.0
for i in range(NW):
    ax.add_patch(Rectangle((W0 + i * (WW + 0.5), WY), WW, WH,
                 fc="white", ec=INK, lw=0.8, zorder=3))
ax.text(W0, WY + WH + 1.6,
        "evaluation clip   (one film clip, $N$ non-overlapping 4 s windows)",
        fontsize=7.4, ha="left", va="bottom", fontweight="bold")
ax.annotate("", xy=(W0 + NW * (WW + 0.5) - 0.5, WY - 2.2), xytext=(W0, WY - 2.2),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
ax.text(W0 + NW * (WW + 0.5) / 2, WY - 3.4,
        "$N$ is set by the stimulus (SEED 46–66,  SEED-V 13–74,  SEED-IV 10–64) — not a choice",
        fontsize=6.7, color=GREY, ha="center", va="top", style="italic")

ar((W0 + NW * (WW + 0.5) + 0.5, WY + WH / 2), (55.0, WY + WH / 2))
box(55.0, WY + 0.5, 10.0, 5.0, "$f$", fs=8.6)
ar((65.0, WY + WH / 2), (69.5, WY + WH / 2))
oper(72.5, WY + WH / 2, "−", r=2.0, fs=10.0)
ar((75.5, WY + WH / 2), (80.0, WY + WH / 2))
box(80.0, WY + 0.5, 18.0, 5.0, "$\\cos(\\tilde z, P_c)$", fs=8.0)
# mu 가 (b) 에서 내려온다
ar((151.5, 94.5), (151.5, 68.0), ls=(0, (2.4, 1.8)), color=ACC, lw=1.3)
_pm = Path([(151.5, 68.0), (151.5, 66.0), (72.5, 66.0), (72.5, 51.6)],
           [Path.MOVETO] + [Path.LINETO] * 3)
ar(path=_pm, ls=(0, (2.4, 1.8)), color=ACC, lw=1.3)
ax.text(112.0, 66.9, "$\\mu$  from (b)", fontsize=7.4, color=ACC,
        ha="center", va="bottom", fontweight="bold")

# 창마다 확률 벡터
ar((98.0, WY + WH / 2), (103.0, WY + WH / 2))
PX, PY = 103.0, 40.0
ax.text(PX - 1.0, 56.2, "one probability vector\nper window", fontsize=7.2,
        ha="left", va="bottom", fontweight="bold", linespacing=1.4)
ROWS = [("win 1", 0.09, 0.85, 0.06), ("win 2", 0.03, 0.93, 0.04),
        ("win 3", 0.42, 0.49, 0.09), ("⋮", None, None, None),
        ("win $N$", 0.03, 0.94, 0.03)]
ax.text(PX + 9.5, 53.4, "neg", fontsize=6.8, color=GREY, ha="center")
ax.text(PX + 16.0, 53.4, "neu", fontsize=6.8, color=GREY, ha="center")
ax.text(PX + 22.5, 53.4, "pos", fontsize=6.8, color=GREY, ha="center")
for j, (lab, a_, b_, c_) in enumerate(ROWS):
    yy = 51.0 - j * 2.9
    ax.text(PX, yy, lab, fontsize=6.9, color=GREY, ha="left", va="center")
    if a_ is None:
        for xx in (PX + 9.5, PX + 16.0, PX + 22.5):
            ax.text(xx, yy, "⋮", fontsize=6.9, color=GREY,
                    ha="center", va="center")
        continue
    for xx, v in ((PX + 9.5, a_), (PX + 16.0, b_), (PX + 22.5, c_)):
        ax.text(xx, yy, f"{v:.2f}", fontsize=6.9, ha="center", va="center",
                color=INK)
ax.plot([PX + 6.0, PX + 25.5], [37.0, 37.0], color=INK, lw=0.8)

# 두 갈래 — 박스를 넓히고 글을 줄여 테두리를 넘지 않게 한다
ar((PX + 27.5, 46.5), (138.0, 54.5), rad=0.12, color=WIN, lw=1.3)
box(138.0, 49.0, 60.0, 11.0,
    "window accuracy\n"
    "each window is one decision — $N$ per clip\n"
    "long clips therefore weigh more",
    fs=7.2, ec=WIN, lw=1.4, tc=WIN)
ax.text(168.0, 60.6, "window", fontsize=8.2, color=WIN, ha="center",
        va="bottom", fontweight="bold")

ar((PX + 27.5, 42.0), (138.0, 36.5), rad=-0.12, color=CLIP, lw=1.3)
box(138.0, 28.0, 60.0, 15.0,
    "clip accuracy\n"
    "mean of the $N$ vectors, then $\\arg\\max$\n"
    "$[0.149,\\ \\mathbf{0.815},\\ 0.036]\\rightarrow$ neutral\n"
    "ONE decision per clip",
    fs=7.2, ec=CLIP, lw=1.4, tc=CLIP)
ax.text(168.0, 43.6, "clip", fontsize=8.2, color=CLIP, ha="center",
        va="bottom", fontweight="bold")

ax.text(1.0, 30.0,
        "Averaging the probabilities is not a majority vote: window 3 "
        "($0.42$ vs $0.49$) is nearly undecided and\n"
        "contributes only $0.49$, while a confident window contributes $0.94$.  "
        "In this clip $51/58$ windows vote\n"
        "neutral (window accuracy $0.879$) and the averaged vector also gives "
        "neutral — but the two can disagree.",
        fontsize=7.0, color=INK, ha="left", va="top", linespacing=1.7)

ax.text(1.0, 17.5,
        "Both are reported throughout.  Which one is primary depends on how many "
        "evaluation clips a given\nanalysis leaves: clip when there are "
        "$\\geq$36, window when only 9 (then one clip is worth 0.111).",
        fontsize=7.0, color=GREY, ha="left", va="top", linespacing=1.7,
        style="italic")

# ════════ 범례 ════════════════════════════════════════════════════════
LY = 1.5


def leg(x, kind, label):
    if kind == "solid":
        ax.plot([x, x + 6], [LY, LY], color=INK, lw=1.1)
    elif kind == "dash":
        ax.plot([x, x + 6], [LY, LY], color=ACC, lw=1.3, ls=(0, (2.4, 1.8)))
    elif kind == "op":
        oper(x + 3, LY, "−", r=1.7, fs=8.5)
    ax.text(x + 7.5, LY, label, fontsize=7.0, va="center", ha="left")


leg(1.0, "solid", "forward pass")
leg(44.0, "dash", "estimated once per session")
leg(104.0, "op", "proposed:  centring  $\\tilde z = z - \\mu$")

fig.tight_layout(pad=0.2)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/fig_method.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
print("[saved] figs/fig_method.png / .pdf")
