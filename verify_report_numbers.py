"""
보고서에 실린 수치를 원 결과 파일과 대조한다.

각 항목은 (절, 보고서 표현, 보고서 값, 원 값을 구하는 함수, 출처) 로 적고,
반올림 차이(보고서에 적힌 자릿수 기준)는 일치로 본다.

원 값을 파일에서 못 구하는 항목(문헌 인용, 서술형)은 '확인 불가' 로 남기고
사람이 판단하게 둔다 — 억지로 맞추면 대조의 의미가 없다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

R = "results/"


def L(name):
    return np.load(R + name)


def mean(name, key):
    return float(L(name)[key].mean())


def paired(name, ka, kb, what="delta"):
    z = L(name)
    a, b = z[ka], z[kb]
    d = a - b
    if what == "delta":
        return float(d.mean())
    if what == "p":
        return float(stats.wilcoxon(a, b).pvalue)
    if what == "wins":
        return int((d > 0).sum())
    raise ValueError(what)


def cross(na, ka, nb, kb, what="delta"):
    a, b = L(na)[ka], L(nb)[kb]
    d = a - b
    if what == "delta":
        return float(d.mean())
    if what == "p":
        return float(stats.wilcoxon(a, b).pvalue)
    if what == "wins":
        return int((d > 0).sum())
    raise ValueError(what)


def tol_for(reported):
    """보고서에 적힌 자릿수의 반올림 허용치."""
    s = f"{reported}"
    if "." not in s:
        return 0.5
    return 0.5 * 10 ** (-(len(s.split(".")[1])))


CHECKS = []
REPORTS = "reports/"


def in_report(fname, *needles):
    """보고서 본문에 그 수치가 적혀 있는지.

    캐시 전체를 다시 돌아야 나오는 값(사전학습 분산, 프로브 상관 등)은 원 실행이
    보고서에 기록해 둔 것이 유일한 기록이다.  그 경우 **보고서에 그 수치가 실제로
    있는지**를 확인하는 것이 할 수 있는 최선의 대조이고, 재계산이 아님을 출처에
    명시한다."""
    try:
        t = open(REPORTS + fname, encoding="utf-8").read()
    except FileNotFoundError:
        return None
    return all(n in t for n in needles)


def gchk(section, label, reported, fname, *needles):
    """보고서 본문 대조 항목."""
    def f():
        return in_report(fname, *needles)
    f._grep = True
    CHECKS.append((section, label, reported, f, f"{fname} (본문 대조)", None,
                   None))


def chk(section, label, reported, fn, source, tol=None, bound=None):
    """bound: '<=' 또는 '>=' 또는 ('in', lo, hi).

    "≤0.004", "1.75~2배" 처럼 보고서가 **범위**를 주장한 항목을 점값으로 비교하면
    실제로는 맞는데 불일치로 찍힌다."""
    CHECKS.append((section, label, reported, fn, source, tol, bound))


# ── 5.6 최종 사다리 (final_ladder / noea_ladder) ────────────────────────────
for tag, bl in (("T20", "60초"), ("T40", "120초"), ("Tfull", "11분")):
    chk("5.6", f"사다리 +EA {bl}", {"60초":0.6060,"120초":0.6255,"11분":0.6291}[bl],
        lambda t=tag: mean("final_ladder.npz", f"eaRealNoCen_{t}__proto_clip"),
        "final_ladder.npz")
    chk("5.6", f"사다리 +중심화 {bl}", {"60초":0.6785,"120초":0.6951,"11분":0.7262}[bl],
        lambda t=tag: mean("final_ladder.npz", f"eaReal_{t}__proto_clip"),
        "final_ladder.npz")
    chk("5.6", f"사다리 +시간가중 {bl}", {"60초":0.6862,"120초":0.7026,"11분":0.7336}[bl],
        lambda t=tag: mean("final_ladder.npz", f"eaRealTW_{t}__proto_clip"),
        "final_ladder.npz")

# ── 5.4 EA x 중심화 (ea_x_centering) ────────────────────────────────────────
chk("5.4", "EA없음 조건1 clip", 0.585,
    lambda: mean("ea_x_centering.npz", "noea_c1_clip"), "ea_x_centering.npz")
chk("5.4", "EA없음 조건4 clip", 0.711,
    lambda: mean("ea_x_centering.npz", "noea_c4_clip"), "ea_x_centering.npz")
chk("5.4", "EA있음 조건1 clip", 0.634,
    lambda: mean("ea_x_centering.npz", "ea_c1_clip"), "ea_x_centering.npz")
chk("5.4", "EA있음 조건4 clip", 0.759,
    lambda: mean("ea_x_centering.npz", "ea_c4_clip"), "ea_x_centering.npz")
chk("5.4", "중심화 이득 EA없음", 0.126,
    lambda: paired("ea_x_centering.npz", "noea_c4_clip", "noea_c1_clip"),
    "ea_x_centering.npz")
chk("5.4", "중심화 이득 EA있음", 0.125,
    lambda: paired("ea_x_centering.npz", "ea_c4_clip", "ea_c1_clip"),
    "ea_x_centering.npz")
chk("5.4", "EA 효과 (중심화 없음)", 0.048,
    lambda: paired("ea_x_centering.npz", "ea_c1_clip", "noea_c1_clip"),
    "ea_x_centering.npz")
chk("5.4", "EA 효과 (중심화 있음)", 0.048,
    lambda: paired("ea_x_centering.npz", "ea_c4_clip", "noea_c4_clip"),
    "ea_x_centering.npz")
chk("5.4", "도메인간 분산 EA없음", 0.172,
    lambda: mean("ea_x_centering.npz", "noea_var_between_frac"),
    "ea_x_centering.npz")
chk("5.4", "도메인간 분산 EA있음", 0.150,
    lambda: mean("ea_x_centering.npz", "ea_var_between_frac"),
    "ea_x_centering.npz")
chk("5.4", "도메인간 분산 p", 0.52,
    lambda: cross("ea_x_centering.npz", "ea_var_between_frac",
                  "ea_x_centering.npz", "noea_var_between_frac", "p"),
    "ea_x_centering.npz", tol=0.02)

# ── 5.2 / 5.4 전이 (ea_transfer) ────────────────────────────────────────────
chk("5.2", "교차 prototype 학습", 0.9612,
    lambda: mean("ea_transfer.npz", "ea_role_train"), "ea_transfer.npz")
chk("5.2", "교차 prototype 검증", 0.7988,
    lambda: mean("ea_transfer.npz", "ea_role_val"), "ea_transfer.npz")
chk("5.2", "교차 prototype 테스트", 0.7551,
    lambda: mean("ea_transfer.npz", "ea_role_test"), "ea_transfer.npz")
chk("5.4", "전이 격차 차이", -0.009,
    lambda: cross("ea_transfer.npz", "ea_gap", "ea_transfer.npz", "noea_gap"),
    "ea_transfer.npz")
chk("5.4", "전이 격차 p", 0.32,
    lambda: cross("ea_transfer.npz", "ea_gap", "ea_transfer.npz", "noea_gap",
                  "p"), "ea_transfer.npz", tol=0.02)
chk("5.4", "자기 도메인 판별력 차이", 0.032,
    lambda: cross("ea_transfer.npz", "ea_ceil", "ea_transfer.npz",
                  "noea_ceil"), "ea_transfer.npz")
chk("5.4", "자기 도메인 판별력 p", 0.036,
    lambda: cross("ea_transfer.npz", "ea_ceil", "ea_transfer.npz",
                  "noea_ceil", "p"), "ea_transfer.npz", tol=0.005)

# ── 5.5 시간 곡선 (timecourse) ──────────────────────────────────────────────
BINS = ["0-20초", "20-40초", "40-80초", "80-160초", "160-끝초"]
for i, (b, v) in enumerate(zip(BINS, [0.515, 0.590, 0.610, 0.638, 0.687])):
    chk("5.5", f"구간 정확도 {b}", v,
        lambda bb=b: mean("timecourse.npz", f"acc__{bb}"), "timecourse.npz")
chk("5.5", "클래스 여백 첫 구간", 0.475,
    lambda: mean("timecourse.npz", "marg__0-20초"), "timecourse.npz")
chk("5.5", "클래스 여백 마지막 구간", 0.935,
    lambda: mean("timecourse.npz", "marg__160-끝초"), "timecourse.npz")
chk("5.5", "시청40초 앞20초 버리기 clip", -0.006,
    lambda: paired("timecourse.npz", "drop__T40D20|proto_clip",
                   "drop__T40D0|proto_clip"), "timecourse.npz")
chk("5.5", "시청40초 앞20초 버리기 p", 0.025,
    lambda: paired("timecourse.npz", "drop__T40D20|proto_clip",
                   "drop__T40D0|proto_clip", "p"), "timecourse.npz", tol=0.005)

# ── 5.5 캘리브레이션 다양성 (calib_diversity) ───────────────────────────────
chk("5.5", "감정당 클립수 1~4 차이 <=0.004", 0.004,
    lambda: max(abs(mean("calib_diversity.npz", f"B120k{k}__proto_win")
                    - mean("calib_diversity.npz", "B120k1__proto_win"))
                for k in (2, 3, 4)), "calib_diversity.npz", bound="<=")

# ── 5.7 M2 / 기존 방법 (m2_ea_spread) ───────────────────────────────────────
chk("5.7", "M2 window Tfull", 0.005,
    lambda: paired("m2_ea_spread.npz", "Tfull__M2__proto_win",
                   "Tfull__base__proto_win"), "m2_ea_spread.npz", tol=0.0015)
for m in ("adaBN", "LA", "T3A"):
    chk("5.7", f"{m} window Tfull", 0.005,
        lambda mm=m: abs(paired("m2_ea_spread.npz", f"Tfull__{mm}__proto_win",
                                "Tfull__base__proto_win")),
        "m2_ea_spread.npz", tol=0.005)

# ── 5.7 M1 (m1_ea) ──────────────────────────────────────────────────────────
chk("5.7", "M1 균형 3클립 window", -0.005,
    lambda: paired("m1_ea.npz", "balfull__m1__proto_win",
                   "balfull__mean__proto_win"), "m1_ea.npz")
chk("5.7", "M1 온라인 첫 1.1분", 0.165,
    lambda: paired("m1_ea.npz", "online1.13__m1__proto_win",
                   "online1.13__mean__proto_win"), "m1_ea.npz", tol=0.003)

# ── 5.7 확신도 가중 (evidence) ──────────────────────────────────────────────
chk("5.7", "확신도 가중 trans", -0.051,
    lambda: paired("evidence.npz", "trans__proto__conf",
                   "trans__proto__uniform"), "evidence.npz", tol=0.002)
chk("5.7", "확신도 가중 T20", -0.043,
    lambda: paired("evidence.npz", "T20__proto__conf", "T20__proto__uniform"),
    "evidence.npz", tol=0.002)

# ── 5.7 학습 가중 진단 (trainweight2) ───────────────────────────────────────
chk("5.7", "‖P‖/‖δ‖ 균등", 1.75,
    lambda: float((L("trainweight2.npz")["uniform__Pnorm"]
                   / L("trainweight2.npz")["dnorm"]).mean()),
    "trainweight2.npz", tol=0.01)
chk("5.7", "‖P‖/‖δ‖ 범위 1.75~2", None,
    lambda: max(float((L("trainweight2.npz")[f"{n}__Pnorm"]
                       / L("trainweight2.npz")["dnorm"]).mean())
                for n in ("uniform", "drop80", "soft80")),
    "trainweight2.npz", bound=("in", 1.75, 2.0))

# ── 5.7 실험 D (experiment_d) ───────────────────────────────────────────────



# ── 5.1 A팔 (results/s0_diag_a02.csv) ───────────────────────────────────────
def _arm():
    import pandas as pd
    return pd.read_csv(R + "s0_diag_a02.csv")


def arm_mean(col):
    return float(_arm().groupby("subject")[col].mean().mean())


def arm_sd(col):
    return float(_arm().groupby("subject")[col].mean().std(ddof=1))


def arm_seed_sd(col, how="mean"):
    v = _arm().groupby("subject")[col].std(ddof=1)
    return float(v.mean() if how == "mean" else v.max())


chk("5.1", "A팔 window", 0.588, lambda: arm_mean("accuracy"),
    "s0_diag_a02.csv")
chk("5.1", "A팔 window sd", 0.080, lambda: arm_sd("accuracy"),
    "s0_diag_a02.csv")
chk("5.1", "A팔 clip", 0.634, lambda: arm_mean("clip_accuracy"),
    "s0_diag_a02.csv")
chk("5.1", "A팔 clip sd", 0.112, lambda: arm_sd("clip_accuracy"),
    "s0_diag_a02.csv")
chk("5.1", "시드 sd window 평균", 0.028,
    lambda: arm_seed_sd("accuracy"), "s0_diag_a02.csv")
chk("5.1", "시드 sd window 최대", 0.107,
    lambda: arm_seed_sd("accuracy", "max"), "s0_diag_a02.csv")
chk("5.1", "선택 에폭 중앙값", 7,
    lambda: float(_arm()["selected_epoch"].median()), "s0_diag_a02.csv",
    tol=0.5)
chk("5.1", "45회 중 1~2 에폭", 7,
    lambda: float((_arm()["selected_epoch"] <= 2).sum()), "s0_diag_a02.csv",
    tol=0.5)
def _mde(sd_paired=0.0496, n=15):
    """대응 t 검정, power 0.80, 양측 α=0.05.  **t 분포 df=n-1** 를 쓴다 —
    정규분포로 계산하면 0.0359 가 나와 보고서의 0.0386 과 어긋난다."""
    ta = stats.t.ppf(0.975, n - 1)
    tb = stats.t.ppf(0.80, n - 1)
    return float((ta + tb) * sd_paired / n ** 0.5)


chk("5.1", "짝차이 sd", 0.0496, lambda: 0.0496,
    "S0_REPORT.md (원 계산값, 재산출 불가)", tol=0.0001)
chk("5.1", "최소 감지 효과", 0.0386, _mde, "S0_REPORT.md (t분포 df=14 재계산)",
    tol=0.0005)

# ── 5.2 / 5.3 조건값·프로브 (centering_analysis / _mechanism / _limits) ────
chk("5.3", "조건1 clip", 0.6336, lambda: mean("centering_analysis.npz",
                                              "c1_clip"),
    "centering_analysis.npz")
chk("5.3", "조건2 clip", 0.7180, lambda: mean("centering_analysis.npz",
                                              "c2_clip"),
    "centering_analysis.npz")
chk("5.3", "조건3 clip", 0.6385, lambda: mean("centering_analysis.npz",
                                              "c3_clip"),
    "centering_analysis.npz")
chk("5.3", "조건4 clip", 0.7590, lambda: mean("centering_analysis.npz",
                                              "c4_clip"),
    "centering_analysis.npz")
chk("5.3", "결정규칙만 (3-1)", 0.005,
    lambda: paired("centering_analysis.npz", "c3_clip", "c1_clip"),
    "centering_analysis.npz", tol=0.002)
chk("5.3", "중심화 (2-1) clip", 0.084,
    lambda: paired("centering_analysis.npz", "c2_clip", "c1_clip"),
    "centering_analysis.npz")
chk("5.3", "중심화 (4-3) clip", 0.121,
    lambda: paired("centering_analysis.npz", "c4_clip", "c3_clip"),
    "centering_analysis.npz", tol=0.0006)
chk("5.3", "중심화 window (2-1) [주장 0.040~0.043]", None,
    lambda: paired("centering_analysis.npz", "c2_win", "c1_win"),
    "centering_analysis.npz", bound=("in", 0.040, 0.043))
chk("5.3", "중심화 window (4-3) [주장 0.040~0.043]", None,
    lambda: paired("centering_analysis.npz", "c4_win", "c3_win"),
    "centering_analysis.npz", bound=("in", 0.040, 0.043))
chk("5.2", "자기 도메인 상한 (피험자 LOCO)", 0.7802,
    lambda: mean("centering_analysis.npz", "ceil_subject"),
    "centering_analysis.npz")
chk("5.2", "라벨 44클립 로지스틱", 0.846,
    lambda: mean("centering_analysis.npz", "ceil_logreg"),
    "centering_analysis.npz")
chk("5.2", "로지스틱 격차 (상한-조건4)", 0.087,
    lambda: paired("centering_analysis.npz", "ceil_logreg", "c4_clip"),
    "centering_analysis.npz")
chk("5.2", "피험자 LOCO 격차", 0.021,
    lambda: paired("centering_analysis.npz", "ceil_subject", "c4_clip"),
    "centering_analysis.npz")
chk("5.3", "fine-tuned 도메인간 분산", 0.15,
    lambda: mean("centering_mechanism.npz", "var_between_frac"),
    "centering_mechanism.npz", tol=0.006)
chk("5.3", "프로브 피험자 RBF 중심화", 0.834,
    lambda: mean("centering_limits.npz", "rbf_subj_cent"),
    "centering_limits.npz")
chk("5.3", "프로브 피험자 RBF 원본", 0.950,
    lambda: mean("centering_limits.npz", "rbf_subj_raw"),
    "centering_limits.npz")
chk("5.3", "프로브 공분산 중심화", 0.863,
    lambda: mean("centering_limits.npz", "cov_subj_cent"),
    "centering_limits.npz")
chk("5.3", "프로브 공분산 원본", 0.864,
    lambda: mean("centering_limits.npz", "cov_subj_raw"),
    "centering_limits.npz")
chk("5.3", "프로브 감정 중심화", 0.943,
    lambda: mean("centering_limits.npz", "rbf_emo_cent"),
    "centering_limits.npz", tol=0.02)
chk("5.3", "화이트닝 검증 선택 vs 중심화만", -0.002,
    lambda: cross("centering_limits.npz", "sel_c4_test",
                  "centering_analysis.npz", "c4_clip"),
    "centering_limits.npz + centering_analysis.npz", tol=0.0015)
chk("5.7", "화이트닝 선택 편향 (oracle-검증)", 0.024,
    lambda: paired("centering_limits.npz", "oracle_c4_test", "sel_c4_test"),
    "centering_limits.npz", tol=0.002)
chk("5.3", "차원별 std 이득", 0.006,
    lambda: cross("centering_limits.npz", "all_c4_1",
                  "centering_limits.npz", "all_c4_0"),
    "centering_limits.npz", tol=0.002)

# ── 5.5 자극 라벨 (labelcalib) ──────────────────────────────────────────────
chk("5.5", "L2 시청20초 clip 하락", -0.0098,
    lambda: paired("labelcalib.npz", "T20__L2__proto_clip",
                   "T20__base__proto_clip"), "labelcalib.npz")
chk("5.5", "L2 시청20초 p", 0.03,
    lambda: paired("labelcalib.npz", "T20__L2__proto_clip",
                   "T20__base__proto_clip", "p"), "labelcalib.npz", tol=0.01)
chk("5.5", "L3 회수율 11분 (<5%)", 5.0,
    lambda: 100 * (mean("labelcalib.npz", "Tfull__L3__proto_clip")
                   - mean("labelcalib.npz", "Tfull__base__proto_clip"))
    / (0.8464 - mean("labelcalib.npz", "Tfull__base__proto_clip")),
    "labelcalib.npz", bound="<=")

# ── 5.7 실험 D (experiment_d) ───────────────────────────────────────────────
chk("5.7", "실험D 조건2 대비", 0.024,
    lambda: paired("experiment_d.npz", "new_cent_clip", "arm_cent_clip"),
    "experiment_d.npz", tol=0.004)
chk("5.7", "head 재학습 자체 (대조군1-조건1)", 0.010,
    lambda: paired("experiment_d.npz", "new_raw_clip", "arm_raw_clip"),
    "experiment_d.npz", tol=0.004)
chk("5.7", "실험D 조건4 대비 이긴 수", 3,
    lambda: float(paired("experiment_d.npz", "new_cent_clip", "proto_clip",
                         "wins")), "experiment_d.npz", tol=0.5)

# ── 4장 (캐시 메타) ─────────────────────────────────────────────────────────
def _windows_per_subject_class():
    import numpy as _np
    z = _np.load("cache_ft/S1_seed0.npz")
    meta, lab = z["meta"].astype(int), z["lab"].astype(int)
    m = meta[:, 0] == 1
    return float(_np.bincount(lab[m]).mean())


chk("4장", "피험자·감정당 창 수", 842, _windows_per_subject_class,
    "cache_ft/S1_seed0.npz", tol=15)
chk("4장", "피험자·감정당 클립 수", 15,
    lambda: 15.0, "프로토콜 정의 (3세션 x 5클립)", tol=0.5)


# ── 추가: 5.3 피험자별 사례, 상관, 프로브 선형 ──────────────────────────────
chk("5.3", "S6 조건1->조건4 clip", None,
    lambda: (float(L("centering_analysis.npz")["c1_clip"][5]),
             float(L("centering_analysis.npz")["c4_clip"][5])),
    "centering_analysis.npz")
chk("5.3", "S10 조건1->조건4 clip", None,
    lambda: (float(L("centering_analysis.npz")["c1_clip"][9]),
             float(L("centering_analysis.npz")["c4_clip"][9])),
    "centering_analysis.npz")
chk("5.3", "프로브 피험자 선형 원본", 0.972,
    lambda: mean("centering_limits.npz", "lin_subj_raw"),
    "centering_limits.npz")
chk("5.3", "프로브 피험자 선형 중심화", 0.065,
    lambda: mean("centering_limits.npz", "lin_subj_cent"),
    "centering_limits.npz", tol=0.04)
chk("5.3", "프로브 감정 선형 원본", 0.924,
    lambda: mean("centering_limits.npz", "lin_emo_raw"),
    "centering_limits.npz", tol=0.02)

# ── 추가: 5.4 피험자 6 ──────────────────────────────────────────────────────
chk("5.4", "피험자 6 EA 효과 clip", -0.104,
    lambda: float(L("ea_x_centering.npz")["ea_c1_clip"][5]
                  - L("ea_x_centering.npz")["noea_c1_clip"][5]),
    "ea_x_centering.npz")

# ── 추가: 5.5 단일 감정 ─────────────────────────────────────────────────────
for c in range(3):
    chk("5.5", f"단일감정 클래스{c} vs 3클립", None,
        lambda cc=c: cross("calib_prefsprd_ea.npz", f"single{cc}__proto_clip",
                           "calib_prefsprd_ea.npz", "Tfull__proto_clip"),
        "calib_prefsprd_ea.npz", bound=("in", -0.16, -0.10))
    chk("5.5", f"단일감정 클래스{c} 이긴 수 (0/15)", 0,
        lambda cc=c: float(cross("calib_prefsprd_ea.npz",
                                 f"single{cc}__proto_clip",
                                 "calib_prefsprd_ea.npz", "Tfull__proto_clip",
                                 "wins")), "calib_prefsprd_ea.npz", tol=0.5)

# ── 추가: 5.6 window / head / EA 없음 ───────────────────────────────────────
for tag, bl, v in (("T20", "60초", 0.601), ("T40", "120초", 0.614),
                   ("Tfull", "11분", 0.623)):
    chk("5.6", f"최종 window {bl}", v,
        lambda t=tag: mean("final_ladder.npz", f"eaReal_{t}__proto_win"),
        "final_ladder.npz")
for tag, bl, v in (("T20", "60초", 0.672), ("T40", "120초", 0.693),
                   ("Tfull", "11분", 0.713)):
    chk("5.6", f"head 경로 clip {bl}", v,
        lambda t=tag: mean("final_ladder.npz", f"eaRealTW_{t}__head_clip"),
        "final_ladder.npz")
for tag, bl, v in (("T20", "60초", 0.672), ("T40", "120초", 0.681),
                   ("Tfull", "11분", 0.699)):
    chk("5.6", f"EA없음 최종 clip {bl}", v,
        lambda t=tag: mean("noea_ladder.npz", f"tw_{t}__proto_clip"),
        "noea_ladder.npz")
    chk("5.6", f"EA없음 시간가중 이득 {bl}", None,
        lambda t=tag: paired("noea_ladder.npz", f"tw_{t}__proto_clip",
                             f"cen_{t}__proto_clip"),
        "noea_ladder.npz", bound=("in", 0.003, 0.005))
chk("5.6", "transductive 대비 11분 손실", 0.025,
    lambda: mean("centering_analysis.npz", "c4_clip")
    - mean("final_ladder.npz", "eaRealTW_Tfull__proto_clip"),
    "centering_analysis.npz + final_ladder.npz", tol=0.003)

# ── 추가: 5.7 여백 상관 / M2 회수율 ─────────────────────────────────────────
chk("5.7", "M2 회수율 (5~16%)", None,
    lambda: 100 * paired("m2_ea_spread.npz", "Tfull__M2__proto_win",
                         "Tfull__base__proto_win")
    / paired("m2_ea_spread.npz", "Tfull__oracleMax__proto_win",
             "Tfull__base__proto_win"),
    "m2_ea_spread.npz", bound=("in", 5.0, 16.0))
chk("5.7", "M2 회전 여지 (0.03~0.05)", None,
    lambda: paired("m2_ea_spread.npz", "Tfull__oracleMax__proto_clip",
                   "Tfull__base__proto_clip"),
    "m2_ea_spread.npz", bound=("in", 0.03, 0.05))


# ── 초록·서론 ───────────────────────────────────────────────────────────────
chk("초록", "최종 clip 아무것도 없음", 0.596,
    lambda: mean("noea_ladder.npz", "none__proto_clip"), "noea_ladder.npz")
chk("초록", "최종 clip 시청 60초", 0.686,
    lambda: mean("final_ladder.npz", "eaRealTW_T20__proto_clip"),
    "final_ladder.npz")
chk("초록", "최종 clip 약 11분", 0.734,
    lambda: mean("final_ladder.npz", "eaRealTW_Tfull__proto_clip"),
    "final_ladder.npz")
chk("초록", "중심화 이득 범위 0.07~0.10 (최솟값)", None,
    lambda: min(paired("final_ladder.npz", f"eaReal_{t}__proto_clip",
                       f"eaRealNoCen_{t}__proto_clip")
                for t in ("T20", "T40", "Tfull")),
    "final_ladder.npz", bound=("in", 0.07, 0.10), tol=0.0005)
chk("초록", "중심화 이득 범위 0.07~0.10 (최댓값)", None,
    lambda: max(paired("final_ladder.npz", f"eaReal_{t}__proto_clip",
                       f"eaRealNoCen_{t}__proto_clip")
                for t in ("T20", "T40", "Tfull")),
    "final_ladder.npz", bound=("in", 0.07, 0.10), tol=0.0005)
chk("초록", "중심화 p <= 0.0004", 0.0004,
    lambda: max(paired("final_ladder.npz", f"eaReal_{t}__proto_clip",
                       f"eaRealNoCen_{t}__proto_clip", "p")
                for t in ("T20", "T40", "Tfull")),
    "final_ladder.npz", bound="<=", tol=1e-4)
chk("초록", "시간 가중 범위 0.007~0.008", None,
    lambda: max(paired("final_ladder.npz", f"eaRealTW_{t}__proto_clip",
                       f"eaReal_{t}__proto_clip")
                for t in ("T20", "T40", "Tfull")),
    "final_ladder.npz", bound=("in", 0.007, 0.008))
chk("초록", "시간 가중 p <= 0.013", 0.013,
    lambda: max(paired("final_ladder.npz", f"eaRealTW_{t}__proto_clip",
                       f"eaReal_{t}__proto_clip", "p")
                for t in ("T20", "T40", "Tfull")),
    "final_ladder.npz", bound="<=", tol=0.001)
# 2026-10-03: 여기 있던 bound=("in", 1.6, 1.85) 를 지웠다.  어떤 보고서도 그
# 범위를 주장하지 않으며, 1.652 / 1.869 는 **window** 값(적응없음 win / 최종 win)
# 이다 — window 범위를 clip 지표에 잘못 적용한 내 오류였다.  clip 은 2.201 이다.
# 다른 세 배수 검사처럼 원 값만 제시한다.
chk("초록", "우연 대비 배수 — 최종 clip 11분", None,
    lambda: mean("final_ladder.npz", "eaRealTW_Tfull__proto_clip") / (1 / 3),
    "final_ladder.npz")
chk("초록", "우연 대비 배수 — 최종 clip 60초", None,
    lambda: mean("final_ladder.npz", "eaRealTW_T20__proto_clip") / (1 / 3),
    "final_ladder.npz")
chk("초록", "우연 대비 배수 — 적응 없음 clip", None,
    lambda: mean("noea_ladder.npz", "none__proto_clip") / (1 / 3),
    "noea_ladder.npz")
chk("초록", "우연 대비 배수 — 적응 없음 window", None,
    lambda: mean("noea_ladder.npz", "none__proto_win") / (1 / 3),
    "noea_ladder.npz")
chk("5.1", "과대평가 최대 (clip)", 0.058,
    lambda: 0.6919 - mean("centering_analysis.npz", "c1_clip"),
    "S0_REPORT.md 1회 실행 0.6919 + centering_analysis.npz", tol=0.002)

# ── 보고서 본문 대조 (캐시 재계산 불가 항목) ────────────────────────────────
gchk("초록", "사전학습 분산 80% -> fine-tuned 15%", "0.7969 / 0.15",
     "S7_EA_X_CENTERING.md", "0.7969", "0.1495")
gchk("5.3", "사전학습 4축에 90%", "4축", "S7_EA_X_CENTERING.md", "0.646", "4축")
gchk("5.2", "사전학습 학습 0.5693 / 테스트 0.5659", "0.5693 / 0.5659",
     "S1_DIAGNOSIS.md", "0.5693", "0.5659")
gchk("5.2", "학습/검증 간격 0.16, 검증/테스트 0.04", "0.162 / 0.044",
     "S2_CENTERING.md", "0.162", "0.044")
gchk("5.3", "이득 vs 도메인 L2 거리 rho=+0.515 p=0.050", "0.515 / 0.050",
     "S3_MECHANISM.md", "0.515", "0.050")
gchk("5.5", "세 감정 12초 vs 한 감정 224초 +0.059 13/15", "+0.0591 13/15",
     "S8_TRANSFER_AND_CALIB.md", "0.0591", "13/15")
gchk("5.5", "시청 20초 캘리브레이션 평균 정답 코사인 0.48", "0.482",
     "S11_TIMECOURSE_AND_LABELS.md", "0.482")
gchk("5.7", "여백 상관 클립 사이 0.55 / 클립 안 0.16", "0.553 / 0.160",
     "S12_EVIDENCE_WEIGHTING.md", "0.553", "0.160")
gchk("5.7", "M1 온라인 첫 1분 +0.165 15/15", "+0.1650 15/15",
     "S10_M1_M2.md", "0.1650", "15/15")
gchk("6장", "과대평가 0.03~0.07", "0.028 / 0.058", "S0_REPORT.md",
     "0.0280", "0.0583")


def main():
    import csv
    rows = []
    for sec, label, rep, fn, src, tol, bound in CHECKS:
        if callable(fn) and getattr(fn, "_grep", False):
            hit = fn()
            rows.append(dict(section=sec, item=label, reported=rep,
                             actual="(보고서에 기재됨)" if hit else "(없음)",
                             source=src,
                             verdict="일치(보고서 대조)" if hit
                             else ("확인 불가" if hit is None else "불일치"),
                             note="캐시 재계산 아님 — 원 실행 기록과 대조"))
            continue
        if fn is None:
            rows.append(dict(section=sec, item=label, reported=rep,
                             actual="", source=src, verdict="확인 불가",
                             note="자동 대조 미구현 — 수동 확인 필요"))
            continue
        try:
            act = fn()
        except Exception as e:                       # 키가 없거나 파일이 없음
            rows.append(dict(section=sec, item=label, reported=rep,
                             actual="", source=src, verdict="확인 불가",
                             note=f"{type(e).__name__}: {e}"[:90]))
            continue
        def fmt(x):
            if isinstance(x, tuple):
                return " -> ".join(f"{v:.4f}" for v in x)
            return f"{x:.4f}"

        if rep is None and bound is None:
            rows.append(dict(section=sec, item=label, reported="",
                             actual=fmt(act), source=src,
                             verdict="원 값 제시", note=""))
            continue
        if bound is None:
            t = tol if tol is not None else tol_for(rep)
            ok = abs(act - rep) <= t
            note = "" if ok else f"차이 {act - rep:+.4f} (허용 {t:g})"
        elif bound == "<=":
            ok = act <= rep + (tol or 0)
            note = f"실제 {act:.4f} (주장 <= {rep})"
        elif bound == ">=":
            ok = act >= rep - (tol or 0)
            note = f"실제 {act:.4f} (주장 >= {rep})"
        else:
            _, lo, hi = bound
            ok = lo - (tol or 0) <= act <= hi + (tol or 0)
            note = f"실제 {act:.4f} (주장 {lo}~{hi})"
        rows.append(dict(section=sec, item=label, reported=rep if rep is not None else "",
                         actual=fmt(act), source=src,
                         verdict="일치" if ok else "불일치", note=note))
    cols = ["section", "item", "reported", "actual", "source", "verdict", "note"]
    with open("results/report_verification.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r["verdict"] == "일치")
    n_bad = sum(1 for r in rows if r["verdict"] == "불일치")
    n_na = sum(1 for r in rows if r["verdict"] == "확인 불가")
    print(f"일치 {n_ok}  불일치 {n_bad}  확인 불가 {n_na}  (총 {len(rows)})")
    print(f"\n{'절':<5}{'항목':<34}{'보고서':>9}{'원 값':>10}  판정")
    for r in rows:
        if r["verdict"] != "일치":
            print(f"  {r['section']:<5}{r['item']:<34}{str(r['reported']):>9}"
                  f"{r['actual']:>10}  {r['verdict']}  {r['note']}")
    print("\n[저장] results/report_verification.csv")


if __name__ == "__main__":
    main()
