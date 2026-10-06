"""교환 캘리브레이션 대조 요약 (2026-10-06) — analyze_calib_swap.py 결과, SEED-V 16명, ⓐ · ⓑ 시드 3개씩.

조건마다 피험자별 시드 평균 → (1) 팔 안에서 본인 대비 (빼는 평균의 출처가 얼마나 중요한가),
(2) 팔 사이 ⓑ − ⓐ (CAFT 의 이득이 본인 캘리브레이션에 묶여 있는가).  '적응 없음' 은 calib_protocol 결과.
검정은 피험자 단위 Wilcoxon (보정 전, 분야 관례).
    python summarize_calib_swap.py | tee results/calib_swap_summary.txt
"""
import numpy as np
from scipy import stats

A3 = ("seedv_noea_seed0", "seedv_noea_seed1", "seedv_noea_seed2")
B3 = ("caftb_seedv_noea", "caftb_seedv_noea_s1", "caftb_seedv_noea_s2")
C1 = ("caftc_seedv_noea",)
CONDS = (("none", "적응 없음"), ("pop", "집단 평균"), ("other_subj", "다른 사람 (같은 영상)"),
         ("other_sess", "본인 다른 세션"), ("own", "본인 (기존 중심화)"))


def get(tag, budget, cond, m="clip"):
    if cond == "none":
        z = np.load(f"results/{tag}_calib_protocol_r10.npz")
        return np.asarray(z["none__proto_clip" if m == "clip" else "none__proto_win"], float)
    z = np.load(f"results/{tag}_calib_swap.npz")
    return np.asarray(z[f"{budget}__{cond}__{m}"], float)


def avg(tags, budget, cond, m="clip"):
    return np.mean([get(t, budget, cond, m) for t in tags], axis=0)


def pair(b, a):
    d = b - a
    try:
        p = stats.wilcoxon(b, a).pvalue
    except ValueError:
        p = float("nan")
    return d.mean(), int((d > 0).sum()), p


subs = np.load(f"results/{A3[0]}_calib_swap.npz")["subjects"]
for t in A3 + B3 + C1:
    assert np.array_equal(np.load(f"results/{t}_calib_swap.npz")["subjects"], subs), t
n = len(subs)
print(f"SEED-V {n}명, 클립 정확도, 피험자마다 시드 평균 (ⓐ · ⓑ 시드 3개, ⓒ 시드 0)")

for budget in ("Tfull", "T20"):
    print(f"\n══ 캘리브레이션 {'끝까지' if budget == 'Tfull' else '20초'} ══")
    print(f"{'빼는 평균':22s}{'ⓐ 일반':>8s}{'ⓑ CAFT①':>10s}{'ⓒ ①+②':>9s}   ⓑ − ⓐ (향상, p)")
    for cond, lab in CONDS:
        a, b, c = avg(A3, budget, cond), avg(B3, budget, cond), avg(C1, budget, cond)
        dm, up, p = pair(b, a)
        print(f"{lab:22s}{a.mean():8.3f}{b.mean():10.3f}{c.mean():9.3f}   {dm:+.3f} ({up}/{n}, p={p:.4f})")
    print("  팔 안: 본인 − 다른 조건 (향상, p)")
    for nm, tags in (("ⓐ", A3), ("ⓑ", B3)):
        own = avg(tags, budget, "own")
        row = []
        for cond, lab in CONDS[:-1]:
            dm, up, p = pair(own, avg(tags, budget, cond))
            row.append(f"{lab.split(' (')[0]} {dm:+.3f} ({up}/{n}, p={p:.4f})")
        print(f"   {nm}: " + " · ".join(row))
    # 이득의 몫: 본인 중심화가 적응 없음 대비 얻는 것 중, 다른 사람 평균으로도 얻는 비율
    for nm, tags in (("ⓐ", A3), ("ⓑ", B3)):
        none_, own, oth = (avg(tags, budget, c).mean() for c in ("none", "own", "other_subj"))
        print(f"   {nm}: 본인 중심화 이득 {own - none_:+.3f} 중 다른 사람 평균으로 얻는 몫 {(oth - none_) / (own - none_):.0%}")
