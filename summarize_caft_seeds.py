"""CAFT ⓑ 대 기준선 ⓐ — SEED-V 16명, 시드 0·1·2 (2026-10-06).

caft_analyze.sh 가 만든 결과 파일만 읽는다 (현실적 캘리브레이션 r10 · SLA · 어긋남, 두 팔 같은 분석).
  ⓐ 일반 파인튜닝   results/seedv_noea_seed{0,1,2}_*.npz
  ⓑ CAFT ① (중심화) results/caftb_seedv_noea{,_s1,_s2}_*.npz
보고 (분야 관례): 피험자마다 시드 3개를 평균한 뒤 짝 비교 (윌콕슨 부호순위 · 향상 피험자 수),
시드별 짝 비교, 시드 수준 평균 ± 표준편차, 교차 9쌍 (ⓑ 시드 i − ⓐ 시드 j), 피험자별 표.
    python summarize_caft_seeds.py
"""
import numpy as np
from scipy import stats

A_TAGS = ("seedv_noea_seed0", "seedv_noea_seed1", "seedv_noea_seed2")
B_TAGS = ("caftb_seedv_noea", "caftb_seedv_noea_s1", "caftb_seedv_noea_s2")
FN = {"cp": "calib_protocol_r10", "sla": "sla", "mis": "misalignment"}
COLS = (("적응없음", "cp", "none__proto_clip"), ("중심화20초", "sla", "T20__center__clip"),
        ("중심화40초", "sla", "T40__center__clip"), ("중심화끝까지", "sla", "Tfull__center__clip"),
        ("+SLA20초", "sla", "T20__sla__clip"), ("+SLA끝까지", "sla", "Tfull__sla__clip"),
        ("창:중심화끝까지", "sla", "Tfull__center__win"), ("c", "mis", "c"))


def load(tag):
    return {k: np.load(f"results/{tag}_{f}.npz") for k, f in FN.items()}


Z = {t: load(t) for t in A_TAGS + B_TAGS}
subs = Z[A_TAGS[0]]["sla"]["subjects"]
for t in Z:
    for k in ("sla", "mis"):
        assert np.array_equal(Z[t][k]["subjects"], subs), (t, k)
    if "subjects" in Z[t]["cp"].files:
        assert np.array_equal(Z[t]["cp"]["subjects"], subs), (t, "cp")
n = len(subs)
print(f"SEED-V 피험자 {n}명, ⓐ {A_TAGS}  ⓑ {B_TAGS}")


def v(t, s, k):
    x = np.asarray(Z[t][s][k], float)
    assert x.shape == (n,), (t, k, x.shape)
    return x


def pair(x, y):
    d = x - y
    try:
        p = stats.wilcoxon(x, y).pvalue
    except ValueError:
        p = float("nan")
    return d.mean(), int((d > 0).sum()), int((d < 0).sum()), p


print(f"\n── 시드 3개 평균 (피험자마다 평균 → 짝 비교, {n}명) " + "─" * 30)
for lab, s, k in COLS:
    A = np.array([v(t, s, k) for t in A_TAGS]); B = np.array([v(t, s, k) for t in B_TAGS])
    dm, up, dn, p = pair(B.mean(0), A.mean(0))
    am, bm = A.mean(1), B.mean(1)          # 시드 수준 (16명 평균) 3개씩
    print(f"{lab:12s} ⓐ {am.mean():.3f}±{am.std(ddof=1):.3f}  ⓑ {bm.mean():.3f}±{bm.std(ddof=1):.3f}"
          f"  차 {dm:+.3f}  향상 {up}/{n} (하락 {dn})  p {p:.4f}")

print(f"\n── 시드별 짝 비교 ⓑs − ⓐs " + "─" * 40)
for lab, s, k in COLS:
    out = []
    for a, b in zip(A_TAGS, B_TAGS):
        dm, up, _, p = pair(v(b, s, k), v(a, s, k))
        out.append(f"{v(a, s, k).mean():.3f}→{v(b, s, k).mean():.3f} ({dm:+.3f}, {up}/{n}, p{p:.3f})")
    print(f"{lab:12s} " + "   ".join(out))

print(f"\n── 교차 9쌍 ⓑ 시드 i − ⓐ 시드 j (16명 평균 차) " + "─" * 20)
for lab, s, k in COLS:
    D = np.array([[v(b, s, k).mean() - v(a, s, k).mean() for a in A_TAGS] for b in B_TAGS])
    print(f"{lab:12s} 최소 {D.min():+.3f}  최대 {D.max():+.3f}  양수 {int((D > 0).sum())}/9")

print(f"\n── SLA 추가 이득 (끝까지: SLA − 중심화) " + "─" * 25)
for nm, tags in (("ⓐ", A_TAGS), ("ⓑ", B_TAGS)):
    g = [(v(t, "sla", "Tfull__sla__clip") - v(t, "sla", "Tfull__center__clip")).mean() for t in tags]
    print(f"{nm}: 시드 0·1·2 " + " ".join(f"{x:+.3f}" for x in g) + f"  (평균 {np.mean(g):+.3f})")

print("\n── 피험자별 중심화 끝까지 클립: ⓐ 시드 0 1 2 | ⓑ 시드 0 1 2 | 평균 차 · ⓑ 가 높은 시드 수 " + "─" * 5)
A = np.array([v(t, "sla", "Tfull__center__clip") for t in A_TAGS])
B = np.array([v(t, "sla", "Tfull__center__clip") for t in B_TAGS])
for i, s_ in enumerate(subs):
    print(f"  S{int(s_):<2d} " + " ".join(f"{x:.3f}" for x in A[:, i]) + " | " + " ".join(f"{x:.3f}" for x in B[:, i])
          + f" | {B[:, i].mean() - A[:, i].mean():+.3f} · {int((B[:, i] > A[:, i]).sum())}/3")
