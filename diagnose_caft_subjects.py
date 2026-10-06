"""CAFT 에서 꾸준히 떨어진 피험자 진단 (2026-10-06) — S6 · S14 (+ 섞인 S8) 대 나머지 13명, SEED-V, ⓐ · ⓑ 시드 3개씩.

(1) 결과 파일의 피험자별 값 (시드 평균): 적응 없음 · 본인 중심화 (끝까지) · 평가 데이터를 미리 쓴 상한 (전달식) ·
    정답으로 회전을 맞춘 상한 (oracle) · 본인 다른 세션 · 다른 사람 캘리브레이션 · 어긋남 c.
    → 떨어짐이 '캘리브레이션 평균이 대표성이 없어서' (본인 < 전달식 격차) 인지, '특징 자체' (전달식도 낮음) 인지,
      '방향' (c · oracle) 인지 가른다.
(2) 본인 중심화 (끝까지) 예측을 직접 뽑아 감정별 재현율 · 세션별 정확도 (analyze_sla 의 center 와 같은 추출 · 난수 · 채점).
    python diagnose_caft_subjects.py | tee results/caft_subject_diagnosis.txt
"""
import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("DATASET", "seedv")
sys.path.insert(0, os.getcwd())
from analyze_calib_rotation import draw
from analyze_centering import l2
from analyze_sla import load_fold, doms_of, calib_windows

A3 = ("seedv_noea_seed0", "seedv_noea_seed1", "seedv_noea_seed2")
B3 = ("caftb_seedv_noea", "caftb_seedv_noea_s1", "caftb_seedv_noea_s2")
CACHE = {"seedv_noea_seed0": "cache_ft_seedv_noea_seed0", "seedv_noea_seed1": "cache_ft_seedv_noea_seed1",
         "seedv_noea_seed2": "cache_ft_seedv_noea_seed2", "caftb_seedv_noea": "cache_ft_seedv_caftb_noea",
         "caftb_seedv_noea_s1": "cache_ft_seedv_caftb_noea_s1", "caftb_seedv_noea_s2": "cache_ft_seedv_caftb_noea_s2"}
EMO = ("disgust", "fear", "sad", "neutral", "happy")
FOCUS = (6, 14, 8)


def col(tag, f, k):
    return np.asarray(np.load(f"results/{tag}_{f}.npz")[k], float)


ITEMS = (("적응없음", "calib_protocol_r10", "none__proto_clip"), ("본인중심화", "sla", "Tfull__center__clip"),
         ("전달식상한", "calib_protocol_r10", "transductive__proto_clip"), ("회전oracle", "calib_rotation", "Tfull__oracle__clip"),
         ("본인다른세션", "calib_swap", "Tfull__other_sess__clip"), ("다른사람", "calib_swap", "Tfull__other_subj__clip"),
         ("어긋남c", "misalignment", "c"))
subs = np.load(f"results/{A3[0]}_sla.npz")["subjects"]
rest = [i for i, s in enumerate(subs) if s not in FOCUS]
print("── (1) 피험자별 (시드 평균) — ⓐ → ⓑ" + "─" * 40)
print(f"{'':6s}" + "".join(f"{lab:>20s}" for lab, _, _ in ITEMS))
V = {lab: (np.mean([col(t, f, k) for t in A3], 0), np.mean([col(t, f, k) for t in B3], 0)) for lab, f, k in ITEMS}
for s in FOCUS:
    i = int(np.flatnonzero(subs == s)[0])
    print(f"S{s:<5d}" + "".join(f"{V[lab][0][i]:9.3f}→{V[lab][1][i]:.3f}" for lab, _, _ in ITEMS))
print(f"{'나머지':6s}" + "".join(f"{V[lab][0][rest].mean():9.3f}→{V[lab][1][rest].mean():.3f}" for lab, _, _ in ITEMS))
print("  (전달식 − 본인 = 캘리브레이션 평균의 대표성 손실; oracle = 정답으로 회전까지 맞춘 천장)")
for s in FOCUS:
    i = int(np.flatnonzero(subs == s)[0])
    g = [V["전달식상한"][j][i] - V["본인중심화"][j][i] for j in (0, 1)]
    print(f"  S{s}: 전달식 − 본인 ⓐ {g[0]:+.3f} · ⓑ {g[1]:+.3f}")
g = [(V["전달식상한"][j][rest] - V["본인중심화"][j][rest]).mean() for j in (0, 1)]
print(f"  나머지: 전달식 − 본인 ⓐ {g[0]:+.3f} · ⓑ {g[1]:+.3f}")


# ── (2) 예측을 직접 뽑아 감정별 · 세션별 ─────────────────────────────────────
def predictions(tag, s, n_rep=10):
    f = os.path.join(CACHE[tag], f"S{s}_seed{tag_seed(tag)}.npz")
    ts, sd, meta, lab, Z, cidx, train, val, P, T = load_fold(f)
    assert ts == s
    doms = doms_of(meta, ts)
    by = defaultdict(list)
    for k in sorted(cidx):
        if k[0] == ts:
            by[((k[0], k[1]), int(lab[cidx[k][0]]))].append(k)
    rng = np.random.default_rng(1000 * ts + sd)      # analyze_sla 와 같은 시드 · 소비
    out = []                                          # (세션, 참, 예측)
    for _ in range(n_rep):
        held = draw(by, doms, rng)
        hs = {k for v in held.values() for k in v}
        ev = [k for k in sorted(cidx) if k[0] == ts and k not in hs]
        mu = {d: Z[calib_windows(held, cidx, lab, d, "full", T)[0]].mean(0) for d in doms}
        for k in ev:
            x = l2((Z[cidx[k]] - mu[(k[0], k[1])]).mean(0)[None])
            out.append((int(k[1]), int(lab[cidx[k][0]]), int((x @ P.T).argmax(1)[0])))
    return np.array(out)


def tag_seed(tag):
    return {"seedv_noea_seed0": 0, "seedv_noea_seed1": 1, "seedv_noea_seed2": 2,
            "caftb_seedv_noea": 0, "caftb_seedv_noea_s1": 1, "caftb_seedv_noea_s2": 2}[tag]


print("\n── (2) 본인 중심화 (끝까지) 예측 — 시드 3개 · 반복 10 합산 " + "─" * 20)
for s in FOCUS:
    pa = np.concatenate([predictions(t, s) for t in A3]); pb = np.concatenate([predictions(t, s) for t in B3])
    i = int(np.flatnonzero(subs == s)[0])
    acc_a, acc_b = (pa[:, 1] == pa[:, 2]).mean(), (pb[:, 1] == pb[:, 2]).mean()
    print(f"S{s}: 정확도 ⓐ {acc_a:.3f} → ⓑ {acc_b:.3f}  (결과 파일 {V['본인중심화'][0][i]:.3f} → {V['본인중심화'][1][i]:.3f})")
    rec = lambda p, c: (p[p[:, 1] == c, 2] == c).mean()
    print("   감정별 재현율 " + "  ".join(f"{EMO[c]} {rec(pa, c):.2f}→{rec(pb, c):.2f}" for c in range(5)))
    sess = sorted(set(pa[:, 0]))
    print("   세션별 정확도 " + "  ".join(
        f"세션{a} {(pa[pa[:, 0] == a, 1] == pa[pa[:, 0] == a, 2]).mean():.2f}→{(pb[pb[:, 0] == a, 1] == pb[pb[:, 0] == a, 2]).mean():.2f}"
        for a in sess))
    cnt = np.bincount(pb[:, 2], minlength=5) / len(pb)
    cnta = np.bincount(pa[:, 2], minlength=5) / len(pa)
    print("   예측 분포 " + "  ".join(f"{EMO[c]} {cnta[c]:.2f}→{cnt[c]:.2f}" for c in range(5)))

# ── (3) 기제: 피험자별 정확도 변화가 방향 일치 c 변화를 따라가는가 · (4) SLA 가 되살리는가 ─────────────
from scipy import stats

dacc = V["본인중심화"][1] - V["본인중심화"][0]
dc = V["어긋남c"][1] - V["어긋남c"][0]
r, p = stats.spearmanr(dc, dacc)
print(f"\n── (3) 피험자 16명: Δ정확도 (ⓑ − ⓐ, 본인 중심화 끝까지) 대 Δc — Spearman ρ = {r:+.2f} (p = {p:.4f})")
print("   c 가 내려간 피험자: " + ", ".join(f"S{int(s)} (Δc {dc[i]:+.3f}, Δ정확도 {dacc[i]:+.3f})"
                                   for i, s in enumerate(subs) if dc[i] < 0))
sla = (np.mean([col(t, "sla", "Tfull__sla__clip") for t in A3], 0), np.mean([col(t, "sla", "Tfull__sla__clip") for t in B3], 0))
dsla = sla[1] - sla[0]
print(f"── (4) 그 위 자극 시점 정렬 (SLA, 끝까지): 하락 피험자 " + (", ".join(
    f"S{int(s)} ({dsla[i]:+.3f})" for i, s in enumerate(subs) if dsla[i] < 0) or "없음"))
for s in FOCUS:
    i = int(np.flatnonzero(subs == s)[0])
    print(f"   S{s}: 중심화 ⓐ {V['본인중심화'][0][i]:.3f} → ⓑ {V['본인중심화'][1][i]:.3f}  |  + SLA ⓐ {sla[0][i]:.3f} → ⓑ {sla[1][i]:.3f}")
