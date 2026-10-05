"""Figure: 재현 — SEED 결론의 SEED-V · SEED-IV 재현, 효과 크기와 95% CI.

2026-10-05: SEED-IV 추가.  SEED-IV 는 유클리드 정렬 없는 모델만 학습했으므로 같은 지표를 계산할 수 있는 항목 (1, 2, 4) 만
넣는다 (results/seediv_calib_protocol_noea_r10.npz: 1 = 평가 데이터 전체로 중심화한 prototype − 중심화 없는 head,
2 = 중심화 없는 prototype − 중심화 없는 head, 4 = 한 감정만 − 모든 감정 끝까지).  SEED · SEED-V 는 기존대로 유클리드 정렬 모델.

논문 영어 figure 로 쓸 수 있게 라벨은 영어, 한글 폰트는 쓰지 않는다.
CI 는 각 npz 의 피험자 단위 짝지은 차이에서 부트스트랩으로 다시 계산한다 —
보고서에 적힌 값을 옮겨 적으면 둘이 어긋날 수 있다.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def boot(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def eff(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = boot(d)
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    return d.mean(), lo, hi, p, int((d > 0).sum()), len(d)


def L(f):
    return np.load(f"results/{f}")


# ── 항목 정의: (라벨, SEED 효과, SEED-V 효과) ────────────────────────────
S_cen, V_cen = L("centering_analysis.npz"), L("seedv_centering_analysis.npz")
S_tc, V_tc = L("timecourse.npz"), L("seedv_timecourse.npz")
S_cp, V_cp = L("calib_protocol.npz"), L("seedv_calib_protocol.npz")
S_fl = L("final_ladder.npz")
V_ev = L("seedv_evidence.npz")
S_fs, V_fs = L("fewshot.npz"), L("seedv_fewshot.npz")
S_ss, V_ss = L("fewshot_sessions.npz"), L("seedv_fewshot_sessions.npz")
I_cp = L("seediv_calib_protocol_noea_r10.npz")          # SEED-IV (유클리드 정렬 없는 모델)

ITEMS = []


def add(label, s, v, iv=None, note=""):
    ITEMS.append((label, s, v, iv, note))


# 1 centering (transductive 조건4 − 조건1, clip) — 두 데이터셋 같은 계산
add("1  Domain centering\n    (cond4 − cond1, clip)",
    eff(S_cen["c4_clip"], S_cen["c1_clip"]),
    eff(V_cen["c4_clip"], V_cen["c1_clip"]),
    eff(I_cp["transductive__proto_clip"], I_cp["none__head_clip"]))

# 2 decision rule only (조건3 − 조건1)
add("2  Decision rule only\n    (cond3 − cond1, clip)",
    eff(S_cen["c3_clip"], S_cen["c1_clip"]),
    eff(V_cen["c3_clip"], V_cen["c1_clip"]),
    eff(I_cp["none__proto_clip"], I_cp["none__head_clip"]))

# 3 time weighting — SEED 는 사다리(fold별 τ), SEED-V 는 evidence
add("3  Time weighting\n    (clip)",
    eff(S_fl["eaRealTW_Tfull__proto_clip"], S_fl["eaReal_Tfull__proto_clip"]),
    eff(V_ev["Tfull__proto__time"], V_ev["Tfull__proto__uniform"]))

# 4 single-emotion calibration (클래스 평균 − 전체 클립)
s_single = np.mean([S_cp[f"single{c}__proto_clip"] for c in range(3)], axis=0)
v_single = np.mean([V_cp[f"single{c}__proto_clip"] for c in range(5)], axis=0)
add("4  Single-emotion calib.\n    (vs all held-out clips)",
    eff(s_single, S_cp["Tfull__proto_clip"]),
    eff(v_single, V_cp["Tfull__proto_clip"]),
    eff(np.mean([I_cp[f"single{c}__proto_clip"] for c in range(4)], axis=0), I_cp["Tfull__proto_clip"]))

# 5 within-clip time course (마지막 구간 − 첫 구간, window)
s_bins = [k[5:] for k in S_tc.files if k.startswith("acc__")]
add("5  Within-clip time course\n    (last − first bin, window)",
    eff(S_tc["acc__160-끝초"], S_tc["acc__0-20초"]),
    eff(V_tc["acc__160-끝초"], V_tc["acc__0-20초"]))

# 7a few-shot pooled (k=1, logistic − label-free, window)
add("7a Few-shot, sessions pooled\n    (k=1, window)",
    eff(S_fs["k1__c_lr__win"], S_fs["k1__a_none__win"]),
    eff(V_fs["k1__c_lr__win"], V_fs["k1__a_none__win"]))

# 7b few-shot same day (k=1)
add("7b Few-shot, same day\n    (k=1, window)",
    eff(S_ss["i__k1__c_lr__win"], S_ss["i__k1__a_none__win"]),
    eff(V_ss["i__k1__c_lr__win"], V_ss["i__k1__a_none__win"]))

# ── 그리기 ──────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.2, 6.2))
n = len(ITEMS)
OFF = 0.26
C = {"SEED": "#2c5f8a", "SEED-V": "#c1553b", "SEED-IV": "#3f7d5a"}

for i, (label, s, v, iv, note) in enumerate(ITEMS):
    y = n - 1 - i
    rows = [(s, +OFF, "SEED"), (v, 0.0, "SEED-V")] + ([(iv, -OFF, "SEED-IV")] if iv is not None else [])
    for (dm, lo, hi, p, w, nn), dy, name in rows:
        sig = p < 0.05
        ax.plot([lo, hi], [y + dy] * 2, color=C[name], lw=1.6,
                solid_capstyle="butt", zorder=2)
        ax.plot([dm], [y + dy], marker="o" if sig else "o",
                ms=5.5, color=C[name] if sig else "white",
                mec=C[name], mew=1.6, zorder=3)
        ax.text(hi + 0.004, y + dy, f"{dm:+.3f}  {w}/{nn}",
                va="center", ha="left", fontsize=7,
                color=C[name], zorder=3)

ax.axvline(0, color="#444", lw=0.9, ls="-", zorder=1)
ax.set_yticks(range(n))
ax.set_yticklabels([lab for lab, *_ in ITEMS][::-1], fontsize=8)
ax.set_xlabel("Paired effect on accuracy  (bootstrap 95% CI)")
ax.set_ylim(-0.6, n - 0.4)
ax.set_xlim(-0.23, 0.26)
ax.tick_params(axis="y", length=0)        # 눈금선이 라벨 뒤에 _ 처럼 보인다
ax.grid(axis="x", color="#ddd", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)

NLAB = {"SEED": "SEED  (n=15)", "SEED-V": "SEED-V  (n=16)", "SEED-IV": "SEED-IV  (n=15, model without EA;\n      items 3, 5, 7 not run)"}
h = [plt.Line2D([], [], marker="o", ls="-", color=C[k], ms=5.5, lw=1.6, label=NLAB[k]) for k in C]
h.append(plt.Line2D([], [], marker="o", ls="none", mfc="white",
                    mec="#555", mew=1.6, ms=5.5, label="p ≥ 0.05"))
ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=7.5)   # 왼쪽 아래는 비어 있다
ax.set_title("Replication on external datasets: SEED → SEED-V, SEED-IV",
             fontsize=10.5, loc="left", pad=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"figs/fig_replication.{ext}", dpi=300,
                bbox_inches="tight", facecolor="white")
print("[saved] figs/fig_replication.png / .pdf")

print(f"\n{'item':<40}{'SEED':>26}{'SEED-V':>26}")
for label, s, v, iv, _ in ITEMS:
    one = label.split("\n")[0].strip()
    print(f"  {one:<38}{s[0]:+.4f} [{s[1]:+.3f},{s[2]:+.3f}] p{s[3]:.4f}"
          f"   {v[0]:+.4f} [{v[1]:+.3f},{v[2]:+.3f}] p{v[3]:.4f}")
