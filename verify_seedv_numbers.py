"""V1_SEEDV_REPLICATION.md (+ V4 결과 절) 의 수치를 원본 npz/csv 에서 다시 계산해 대조한다.

보고서에 적은 값을 손으로 옮기다 틀리는 것, 그리고 재실행으로 수치가 바뀐 것을
잡는 용도다.  SEED 쪽 verify_report_numbers.py 와 같은 방식이다.
"""
from __future__ import annotations

import csv
import collections
import statistics as st
import sys

import numpy as np
from scipy import stats

CHECKS: list[tuple[str, float, float, float]] = []   # (라벨, 주장, 실제, 허용)
MISSING: list[str] = []


def chk(label, claimed, actual, tol=5e-5):
    CHECKS.append((label, claimed, actual, tol))


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    return d.mean(), p, int((d > 0).sum()), len(d)


def load(path):
    try:
        return np.load(path)
    except FileNotFoundError:
        MISSING.append(path)
        return None


# ══ 1. A팔 raw CSV ═══════════════════════════════════════════════════════
rows = list(csv.DictReader(open("results/seedv_a02.csv")))
subj = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for m in ("accuracy", "clip_accuracy", "f1_macro", "clip_f1_macro",
              "window_balanced", "clip_balanced"):
        subj[int(r["subject"])][m].append(float(r[m]))
chk("A팔 runs", 48, len(rows), 0)
chk("A팔 피험자", 16, len(subj), 0)
for m, claimed in (("accuracy", 0.3038), ("clip_accuracy", 0.3273),
                   ("window_balanced", 0.3067), ("f1_macro", 0.2663),
                   ("clip_f1_macro", 0.2631)):
    v = [st.mean(subj[s][m]) for s in sorted(subj)]
    chk(f"A팔 {m}", claimed, st.mean(v), 5e-5)
sd = [st.stdev(subj[s]["accuracy"]) for s in subj]
sdc = [st.stdev(subj[s]["clip_accuracy"]) for s in subj]
chk("A팔 시드sd 창", 0.0233, st.mean(sd), 5e-5)
chk("A팔 시드sd 클립", 0.0511, st.mean(sdc), 5e-5)

# ══ 2. 중심화 (기준 1·2·6) ═══════════════════════════════════════════════
C = load("results/seedv_centering_analysis.npz")
if C is not None:
    for k, claimed in (("c1_clip", 0.3273), ("c2_clip", 0.3731),
                       ("c3_clip", 0.3218), ("c4_clip", 0.4208),
                       ("c1_win", 0.3040), ("c2_win", 0.3304),
                       ("c3_win", 0.3027), ("c4_win", 0.3297)):
        chk(f"조건 {k}", claimed, float(C[k].mean()))
    # 조건1 이 A팔 raw 와 일치해야 한다 (head/중심화없음 = raw 재현)
    raw_clip = st.mean([st.mean(subj[s]["clip_accuracy"]) for s in sorted(subj)])
    chk("조건1 clip == A팔 raw clip", raw_clip, float(C["c1_clip"].mean()), 5e-4)
    for lbl, a, b, dc, pc, wc in (
            ("조건4−조건1 clip", "c4_clip", "c1_clip", +0.0935, 0.0005, 15),
            ("조건4−조건1 win", "c4_win", "c1_win", +0.0257, 0.0000, 16),
            ("조건3−조건1 clip", "c3_clip", "c1_clip", -0.0056, 0.2801, 6),
            ("조건2−조건1 clip", "c2_clip", "c1_clip", +0.0458, 0.0027, 13)):
        d, p, w, n = paired(C[a], C[b])
        chk(f"{lbl} Δ", dc, d, 5e-5)
        chk(f"{lbl} p", pc, p, 5e-5)
        chk(f"{lbl} 이긴수", wc, w, 0)
        chk(f"{lbl} n", 16, n, 0)
    for k, claimed in (("role_train", 0.9029), ("role_val", 0.4384),
                       ("role_test", 0.4236), ("ceil_logreg", 0.5431),
                       ("ceil_subject", 0.5079), ("ceil_session", 0.4560)):
        if k in C.files:
            chk(f"보완 {k}", claimed, float(np.asarray(C[k]).mean()))
        else:
            MISSING.append(f"centering:{k}")

# ══ 3. 시간 곡선 (기준 5) ════════════════════════════════════════════════
T = load("results/seedv_timecourse.npz")
if T is not None:
    bins = ["0-20초", "20-40초", "40-80초", "80-160초", "160-끝초"]
    acc_claimed = [0.2692, 0.3026, 0.3378, 0.3434, 0.3524]
    mar_claimed = [0.1466, 0.2079, 0.2804, 0.2958, 0.3402]
    for b, ac, mc in zip(bins, acc_claimed, mar_claimed):
        chk(f"시간곡선 acc {b}", ac, float(T[f"acc__{b}"].mean()))
        chk(f"시간곡선 여백 {b}", mc, float(T[f"marg__{b}"].mean()))
    for b, dc, pc, wc in (("20-40초", +0.0334, 0.0034, 14),
                          ("40-80초", +0.0686, 0.0000, 16),
                          ("80-160초", +0.0743, 0.0001, 15),
                          ("160-끝초", +0.0832, 0.0002, 14)):
        d, p, w, n = paired(T[f"acc__{b}"], T["acc__0-20초"])
        chk(f"시간곡선 {b} Δ", dc, d, 5e-5)
        chk(f"시간곡선 {b} p", pc, p, 5e-5)
        chk(f"시간곡선 {b} 이긴수", wc, w, 0)

# ══ 4. 단일 감정 (기준 4) ════════════════════════════════════════════════
P = load("results/seedv_calib_protocol.npz")
if P is not None:
    names = ("disgust", "fear", "sad", "neutral", "happy")
    claimed = ((-0.0282, 0.0027, 2), (-0.0460, 0.0001, 1), (-0.0469, 0.0002, 1),
               (-0.0525, 0.0000, 0), (-0.0687, 0.0008, 1))
    full = P["Tfull__proto_clip"]
    for c, (nm, (dc, pc, wc)) in enumerate(zip(names, claimed)):
        d, p, w, n = paired(P[f"single{c}__proto_clip"], full)
        chk(f"단일 {nm} Δ", dc, d, 5e-5)
        chk(f"단일 {nm} p", pc, p, 5e-5)
        chk(f"단일 {nm} 이긴수", wc, w, 0)
    mean_single = np.mean([P[f"single{c}__proto_clip"] for c in range(5)], axis=0)
    d, p, w, n = paired(mean_single, full)
    chk("단일 5클래스평균 Δ", -0.0485, d, 5e-5)
    chk("단일 5클래스평균 p", 0.0001, p, 5e-5)
    chk("단일 5클래스평균 이긴수", 1, w, 0)

# ══ 5. 시간 가중 (기준 3) ════════════════════════════════════════════════
E = load("results/seedv_evidence.npz")
if E is not None:
    # 조건4 prototype clip, (a) 균등 대비 (b) 시간 가중
    for tag, ac, bc, dc, pc in (("T20", 0.3910, 0.3974, +0.0063, 0.274),
                                ("T40", 0.3990, 0.4019, +0.0028, 0.326)):
        ka, kb = f"{tag}__proto__uniform", f"{tag}__proto__time"
        if ka in E.files and kb in E.files:
            chk(f"시간가중 {tag} (a)", ac, float(E[ka].mean()), 5e-5)
            chk(f"시간가중 {tag} (b)", bc, float(E[kb].mean()), 5e-5)
            d, p, w, n = paired(E[kb], E[ka])
            chk(f"시간가중 {tag} Δ", dc, d, 5e-5)
            chk(f"시간가중 {tag} p", pc, p, 1e-3)
        else:
            MISSING.append(f"evidence:{ka}")

# ══ 6. few-shot 7a ═══════════════════════════════════════════════════════
F = load("results/seedv_fewshot.npz")
if F is not None:
    base = {1: 0.3261, 2: 0.3284}
    meth = {"b_mix": (0.0404, 0.0546), "c_lr": (0.0679, 0.1079),
            "c_lrreg": (-0.0072, 0.0012), "d_head": (0.0322, 0.0540)}
    for k in (1, 2):
        ka = f"k{k}__a_none__win"
        if ka not in F.files:
            MISSING.append(f"fewshot:{ka}"); continue
        chk(f"7a k={k} 기준선", base[k], float(F[ka].mean()), 5e-5)
        for m, (d1, d2) in meth.items():
            km = f"k{k}__{m}__win"
            if km not in F.files:
                MISSING.append(f"fewshot:{km}"); continue
            d, p, w, n = paired(F[km], F[ka])
            chk(f"7a k={k} {m} Δ", (d1, d2)[k - 1], d, 5e-5)

# ══ 7. few-shot 7b ═══════════════════════════════════════════════════════
S = load("results/seedv_fewshot_sessions.npz")
if S is not None:
    for k, bc, lc, dc, pc, wc in ((1, 0.3281, 0.3447, +0.0165, 0.3225, 10),
                                  (2, 0.3301, 0.3823, +0.0522, 0.0034, 12)):
        kb, kl = f"i__k{k}__a_none__win", f"i__k{k}__c_lr__win"
        if kb not in S.files:
            MISSING.append(f"sessions:{kb}"); continue
        chk(f"7b k={k} 라벨없음", bc, float(S[kb].mean()), 5e-5)
        chk(f"7b k={k} 로지스틱", lc, float(S[kl].mean()), 5e-5)
        d, p, w, n = paired(S[kl], S[kb])
        chk(f"7b k={k} Δ", dc, d, 5e-5)
        chk(f"7b k={k} p", pc, p, 5e-5)
        chk(f"7b k={k} 이긴수", wc, w, 0)

# ══ 8. 기준 1 정식 · 기준 8 (2026-10-04) ══════════════════════════════════
R = load("results/seedv_ea_realistic.npz")
N = load("results/seedv_calib_protocol_noea_r10.npz")
if R is not None and N is not None:
    for t, (b, a, dc, pc, wc) in {"T20": (0.3190, 0.3849, 0.0658, 0.0010, 13),
                                  "T40": (0.3184, 0.3928, 0.0744, 0.0004, 14),
                                  "Tfull": (0.3181, 0.3972, 0.0792, 0.0000, 16)}.items():
        x, y = R[f"eaReal_{t}__proto_clip"], R[f"eaRealNoCen_{t}__proto_clip"]
        d, p_, w, n = paired(x, y)
        chk(f"기준1정식 {t} EA만", b, float(y.mean()), 5e-5)
        chk(f"기준1정식 {t} +중심화", a, float(x.mean()), 5e-5)
        chk(f"기준1정식 {t} Δ", dc, d, 5e-5)
        chk(f"기준1정식 {t} p", pc, p_, 5e-5)
        chk(f"기준1정식 {t} 이긴수", wc, w, 0)
    for t, dc in (("T20", 0.0168), ("T40", 0.0242), ("Tfull", 0.0302)):
        d, p_, w, n = paired(R[f"eaReal_{t}__proto_clip"], N[f"{t}__proto_clip"])
        chk(f"기준8 {t} Δ(EA−noEA)", dc, d, 5e-5)
        chk(f"기준8 {t} p≥0.14", 1.0, float(p_ >= 0.14), 0)

# ══ 9. V4 CR (V4_CALIB_ROTATION_PREREG.md 결과 절) ══════════════════════════
def _holm(ps):
    ps = np.asarray(ps, float); n = len(ps); out = np.empty(n); m = 0.0
    for r, i in enumerate(np.argsort(ps)):
        m = max(m, min(1.0, ps[i] * (n - r))); out[i] = m
    return out

V4 = {  # (데이터셋, 예산): (중심화, CR, oracle, Δ, p, Holm, 이긴수)  — no-EA 팔 clip
    ("", "T20"): (0.6673, 0.6639, 0.7583, -0.0035, 0.211, 0.422, 5),
    ("", "T40"): (0.6770, 0.6720, 0.7738, -0.0050, 0.109, 0.328, 5),
    ("", "Tfull"): (0.6966, 0.6960, 0.8039, -0.0006, 0.955, 0.955, 6),
    ("seedv_", "T20"): (0.3695, 0.3766, 0.6975, 0.0071, 0.078, 0.313, 10),
    ("seedv_", "T40"): (0.3710, 0.3902, 0.7089, 0.0192, 0.0076, 0.038, 14),
    ("seedv_", "Tfull"): (0.3667, 0.3987, 0.7200, 0.0319, 0.0027, 0.016, 15),
}
Zs = {pre: load(f"results/{pre}calib_rotation_noea.npz") for pre in ("", "seedv_")}
if all(z is not None for z in Zs.values()):
    ps = []
    for (pre, t), v in V4.items():
        z = Zs[pre]; c, r, o = (z[f"{t}__{m}__clip"] for m in ("center", "cr", "oracle"))
        d, p_, w, n = paired(r, c); ps.append(p_)
        tag = f"V4 {pre or 'seed_'}{t}"
        for lab_, cl, ac in (("중심화", v[0], c.mean()), ("CR", v[1], r.mean()),
                             ("oracle", v[2], o.mean()), ("Δ", v[3], d)):
            chk(f"{tag} {lab_}", cl, float(ac), 5e-5)
        chk(f"{tag} p", v[4], p_, 5e-4 if v[4] >= 0.01 else 5e-5)
        chk(f"{tag} 이긴수", v[6], w, 0)
    for (k, v), h in zip(V4.items(), _holm(ps)):
        chk(f"V4 {k[0] or 'seed_'}{k[1]} Holm", v[5], float(h), 5e-4)
    # 창 (no-EA) 과 EA 팔 clip 의 Δ
    for pre, arm, met, vals in (("", "noea", "win", (0.0000, -0.0022, 0.0025)),
                                ("seedv_", "noea", "win", (0.0055, 0.0077, 0.0177)),
                                ("", "ea", "clip", (-0.0077, -0.0112, 0.0001)),
                                ("seedv_", "ea", "clip", (0.0075, 0.0080, 0.0208))):
        z = load(f"results/{pre}calib_rotation_{arm}.npz")
        if z is None:
            continue
        for t, cl in zip(("T20", "T40", "Tfull"), vals):
            d, p_, w, n = paired(z[f"{t}__cr__{met}"], z[f"{t}__center__{met}"])
            chk(f"V4 {pre or 'seed_'}{arm} {met} {t} Δ", cl, d, 5e-5)

# ══ 10. V4 부록 A (no-EA 팔 labelcalib) ════════════════════════════════════
LA = {pre: load(f"results/{pre}labelcalib_noea.npz") for pre in ("", "seedv_")}
if all(z is not None for z in LA.values()):
    # 규칙 1~3: (L2, L3, L3−L2, p, L3 이긴 수)
    for pre, (l2v, l3v, dv, pv, wv) in (("", (0.7057, 0.7107, 0.0050, 0.349, 10)),
                                         ("seedv_", (0.3994, 0.4049, 0.0055, 0.140, 9))):
        z = LA[pre]; a, b = z["Tfull__L3__proto_clip"], z["Tfull__L2__proto_clip"]
        d, p_, w, n = paired(a, b)
        tag = f"V4A {pre or 'seed_'}"
        chk(tag + "L2", l2v, float(b.mean())); chk(tag + "L3", l3v, float(a.mean()))
        chk(tag + "L3−L2", dv, d); chk(tag + "L3−L2 p", pv, p_, 5e-4); chk(tag + "L3 이긴수", wv, w, 0)
    # 규칙 4: L3 vs 기준선 Δ 와 Holm (족 6)
    L3T = {("", "T20"): (-0.0074, 0.52), ("", "T40"): (-0.0014, 1.00), ("", "Tfull"): (0.0148, 0.0089),
           ("seedv_", "T20"): (0.0024, 1.00), ("seedv_", "T40"): (0.0204, 0.052),
           ("seedv_", "Tfull"): (0.0378, 0.0009)}
    ps, ds = [], []
    for (pre, t) in L3T:
        z = LA[pre]; d, p_, w, n = paired(z[f"{t}__L3__proto_clip"], z[f"{t}__base__proto_clip"])
        ps.append(p_); ds.append(d)
    for ((pre, t), (dv, hv)), d, h in zip(L3T.items(), ds, _holm(ps)):
        chk(f"V4A L3 {pre or 'seed_'}{t} Δ", dv, d)
        chk(f"V4A L3 {pre or 'seed_'}{t} Holm", hv, float(h), 5e-3 if hv >= 0.1 else 5e-4 if hv >= 0.01 else 5e-5)
    # V4 와 부호가 다른 SEED L2 클립 전체
    d, p_, w, n = paired(LA[""]["Tfull__L2__proto_clip"], LA[""]["Tfull__base__proto_clip"])
    chk("V4A SEED L2 Tfull Δ", 0.0098, d); chk("V4A SEED L2 Tfull p", 0.017, p_, 5e-4)
    # 규칙 5: L4
    for pre, (hv, d2v, d4v) in (("", (0.7174, 0.0451, 0.0214)), ("seedv_", (0.4003, 0.0577, 0.0333))):
        z = LA[pre]; h = z["Tfull__L4__head_clip"]
        chk(f"V4A {pre or 'seed_'}L4 head_clip", hv, float(h.mean()))
        chk(f"V4A {pre or 'seed_'}L4 vs 조건2", d2v, paired(h, z["Tfull__base__head_clip"])[0])
        chk(f"V4A {pre or 'seed_'}L4 vs 조건4", d4v, paired(h, z["Tfull__base__proto_clip"])[0])

# ══ 11. subject-dependent (README · NEXT.md) ═════════════════════════════
for path, nrun, vals in (("results/a02_sd.csv", 45, (0.6536, 0.7506)),
                         ("results/seedv_a02_sd.csv", 48, (0.4123, 0.5222))):
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        MISSING.append(path); continue
    chk(f"SD {path} runs", nrun, len(rows), 0)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for m in ("accuracy", "clip_accuracy"):
            by[int(r["subject"])][m].append(float(r[m]))
    for m, cl in zip(("accuracy", "clip_accuracy"), vals):
        chk(f"SD {path} {m}", cl, st.mean([st.mean(by[s_][m]) for s_ in sorted(by)]), 5e-5)

# ══ 12. V6 SEED-IV (V6_SEEDIV_SLA_PREREG.md 결과 절) ═════════════════════════
MZ, CPz, SLz = (load(f) for f in ("results/seediv_misalignment_noea.npz", "results/seediv_calib_protocol_noea_r10.npz",
                                   "results/seediv_sla_noea.npz"))
if MZ is not None and CPz is not None and SLz is not None:
    chk("V6 c", 0.428, float(MZ["c"].mean()), 5e-4)
    chk("V6 적응 없음 clip", 0.3704, float(CPz["none__proto_clip"].mean()))
    for t, (cv, dv) in (("T20", (0.4213, 0.0509)), ("T40", (0.4264, 0.0561)), ("Tfull", (0.4303, 0.0600))):
        d, p_, w, n = paired(CPz[f"{t}__proto_clip"], CPz["none__proto_clip"])
        chk(f"V6 중심화 {t} clip", cv, float(CPz[f"{t}__proto_clip"].mean()))
        chk(f"V6 중심화 {t} Δ", dv, d); chk(f"V6 중심화 {t} p", 0.0001, p_, 1e-4); chk(f"V6 중심화 {t} 향상", 14, w, 0)
    chk("V6 전달식 − Tfull", 0.0152, float(CPz["transductive__proto_clip"].mean() - CPz["Tfull__proto_clip"].mean()))
    for k, (dfull, dnone) in enumerate(((-0.0441, 0.0159), (-0.0329, 0.0270), (-0.0323, 0.0277), (-0.0457, 0.0143))):
        a = CPz[f"single{k}__proto_clip"]
        chk(f"V6 단일 {k} vs 전체", dfull, paired(a, CPz["Tfull__proto_clip"])[0])
        chk(f"V6 단일 {k} vs 없음", dnone, paired(a, CPz["none__proto_clip"])[0])
    SLV = {"T20": (0.4214, 0.4305, 0.0091, 0.1070, 9), "T40": (0.4265, 0.4321, 0.0056, 0.1514, 11),
           "Tfull": (0.4307, 0.4616, 0.0308, 0.0256, 11)}
    ps = []
    for t, (cv, sv, dv, pv, wv) in SLV.items():
        d, p_, w, n = paired(SLz[f"{t}__sla__clip"], SLz[f"{t}__center__clip"]); ps.append(p_)
        chk(f"V6 SLA {t} center", cv, float(SLz[f"{t}__center__clip"].mean()))
        chk(f"V6 SLA {t} SLA", sv, float(SLz[f"{t}__sla__clip"].mean()))
        chk(f"V6 SLA {t} Δ", dv, d); chk(f"V6 SLA {t} p", pv, p_, 5e-4); chk(f"V6 SLA {t} 향상", wv, w, 0)
    for t, hv, h in zip(SLV, (0.2140, 0.2140, 0.0767), _holm(ps)):
        chk(f"V6 SLA {t} Holm", hv, float(h), 5e-4)
    chk("V6 L3−center Tfull", 0.0307, paired(SLz["Tfull__l3__clip"], SLz["Tfull__center__clip"])[0])
    chk("V6 L3−center Tfull p", 0.048, paired(SLz["Tfull__l3__clip"], SLz["Tfull__center__clip"])[1], 5e-4)
    chk("V6 SLA−L3 Tfull", 0.0001, paired(SLz["Tfull__sla__clip"], SLz["Tfull__l3__clip"])[0])
    chk("V6 SLA+L3−L3 Tfull", 0.0024, paired(SLz["Tfull__slal3__clip"], SLz["Tfull__l3__clip"])[0])
    d, p_, w, n = paired(SLz["Tfull__sla__win"], SLz["Tfull__center__win"])
    chk("V6 SLA win Tfull Δ", 0.0179, d); chk("V6 SLA win Tfull p", 0.0054, p_, 5e-4); chk("V6 SLA win Tfull 향상", 13, w, 0)
    g = SLz["Tfull__sla__clip"] - SLz["Tfull__center__clip"]
    chk("V6 탐색 Spearman SEED-IV", -0.457, float(stats.spearmanr(MZ["c"], g).correlation), 5e-4)
else:
    MISSING.append("V6 SEED-IV 결과 파일")

# ══ 보고 ═════════════════════════════════════════════════════════════════
bad = [(l, c, a, t) for l, c, a, t in CHECKS if abs(c - a) > t]
print(f"검사 {len(CHECKS)}개,  불일치 {len(bad)}개,  키 없음 {len(MISSING)}개\n")
if bad:
    print(f"{'항목':<34}{'보고':>11}{'실제':>11}{'차이':>11}")
    for l, c, a, t in bad:
        print(f"  {l:<32}{c:>11.4f}{a:>11.4f}{a-c:>+11.4f}")
if MISSING:
    print("\n키/파일 없음:")
    for m in MISSING:
        print(f"  {m}")
with open("results/seedv_verification.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "claimed", "actual", "diff", "tol", "ok"])
    for l, c, a, t in CHECKS:
        w.writerow([l, f"{c:.6f}", f"{a:.6f}", f"{a-c:+.6f}", t,
                    "OK" if abs(c - a) <= t else "MISMATCH"])
print(f"\n[저장] results/seedv_verification.csv")
sys.exit(1 if bad else 0)
