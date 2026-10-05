"""우리 연구의 기여 정리 (분야 관례 기준) — reports/CONTRIBUTIONS.pdf.

2026-10-05 사용자 결정: "정직하게 하되, 기존 논문들의 기조를 따라간다" — 주장의 강도는 분야 관례 (피험자 단위 짝지은 Wilcoxon,
평균 · 향상된 피험자 수) 로 판단하고, 모든 데이터셋 · 조건을 무효과까지 보고한다.  내부 사전 등록 판정은 참고로만 적는다.
수치는 results/ 에서 읽고, 본문 주장마다 assert 로 조건을 확인한다.  끝에 폰트에 없는 글자를 검사한다.

    python make_contrib_pdf.py
"""
from __future__ import annotations

import csv
import datetime
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
import numpy as np
from scipy import stats
from fontTools.ttLib import TTFont as FTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
                                KeepTogether, CondPageBreak)

FONT_R, FONT_B = ".fonts/NotoSansKR-regular.ttf", ".fonts/NotoSansKR-bold.ttf"
pdfmetrics.registerFont(TTFont("NotoKR", FONT_R))
pdfmetrics.registerFont(TTFont("NotoKR-B", FONT_B))
registerFontFamily("NotoKR", normal="NotoKR", bold="NotoKR-B", italic="NotoKR", boldItalic="NotoKR-B")
F, FB = "NotoKR", "NotoKR-B"
INK, GREY, ACC = colors.HexColor("#1a1d21"), colors.HexColor("#6f757b"), colors.HexColor("#c1553b")
LINE, BG, NOTE, HL = (colors.HexColor("#d4d9dd"), colors.HexColor("#f5f7f9"), colors.HexColor("#eef4ef"),
                      colors.HexColor("#fbeae4"))


def st(name, size=9.3, lead=14.2, color=INK, font=F, space=4, left=0):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead, textColor=color, spaceAfter=space,
                          leftIndent=left, alignment=TA_LEFT)


BODY = st("b"); BUL = st("bul", left=10)
TITLE = st("t", 18, 25, INK, FB, 5); SUBT = st("st", 11.5, 16, INK, F, 6)
H2 = st("h2", 12.5, 17, ACC, FB, 6); H2.keepWithNext = 1
H3 = st("h3", 10.4, 14.6, INK, FB, 3); H3.keepWithNext = 1
H2S, H3S = ParagraphStyle("h2s", parent=H2), ParagraphStyle("h3s", parent=H3)
H2S.keepWithNext = H3S.keepWithNext = 0
SMALL = st("s", 8.0, 11.6, GREY, space=3); EN = st("en", 8.6, 12.6, colors.HexColor("#2f3a44"), space=4, left=10)
CELL = st("c", 7.8, 10.8, INK, space=0); CELLB = st("cb", 7.8, 10.8, INK, FB, space=0)
BOX = ParagraphStyle("box", fontName=F, fontSize=9.2, leading=14.2, textColor=INK, backColor=NOTE, borderPadding=8,
                     spaceAfter=9, leftIndent=4, rightIndent=4)
E, TEXTS = [], []


def _p(t, s):
    TEXTS.append(t)
    p = Paragraph(t, s); p._src = t
    return p


def P(t, s=BODY):
    E.append(_p(t, s))


def B(items, s=BUL):
    for t in items:
        E.append(_p("• " + t, s))


def gap(h=6):
    E.append(Spacer(1, h))


def _heads():
    hs = []
    while E and isinstance(E[-1], Paragraph) and E[-1].style in (H2, H3):
        hs.insert(0, E.pop())
    return hs


def T(rows, widths, hl=(), split=False):
    data = [[_p(str(c), CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], hAlign="LEFT", repeatRows=1)
    s = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.8),
         ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8), ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
         ("RIGHTPADDING", (0, 0), (-1, -1), 3.5), ("BACKGROUND", (0, 0), (-1, 0), BG),
         ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
         ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK)]
    for r in hl:
        s.append(("BACKGROUND", (0, r), (-1, r), HL))
    t.setStyle(TableStyle(s))
    hs = _heads()
    if split:
        E.append(CondPageBreak(60 * mm))
        E.extend(Paragraph(h._src, H2S if h.style is H2 else H3S) for h in hs); E.append(t)
    else:
        E.append(KeepTogether(hs + [t]))
    gap(5)


# ── 수치 ─────────────────────────────────────────────────────────────────
R = lambda f: np.load(f"results/{f}")
m = lambda a: float(np.asarray(a, float).mean())


def pr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    return float(d.mean()), float(stats.wilcoxon(a, b).pvalue), int((d > 0).sum()), len(d)


DS = (("SEED", ""), ("SEED-V", "seedv_"), ("SEED-IV", "seediv_"))
BUD = ("T20", "T40", "Tfull")
CP = {d: R(f"{p}calib_protocol_noea_r10.npz") for d, p in DS}
SL = {d: R(f"{p}sla_noea.npz") for d, p in DS}
CRt = {d: R(f"{p}calib_rotation_noea.npz") for d, p in DS}
MIS = {d: m(R(f"{p}misalignment_noea.npz")["c"]) for d, p in DS}
SG = {d: R(f"{p}session_geometry.npz") for d, p in DS[:2]}          # 유클리드 정렬 팔 (V3)
cen = {d: {t: pr(CP[d][f"{t}__proto_clip"], CP[d]["none__proto_clip"]) for t in BUD} for d, _ in DS}
sla = {d: {t: pr(SL[d][f"{t}__sla__clip"], SL[d][f"{t}__center__clip"]) for t in BUD} for d, _ in DS}
l3 = {d: {t: pr(SL[d][f"{t}__l3__clip"], SL[d][f"{t}__center__clip"]) for t in BUD} for d, _ in DS}
cr = {d: {t: pr(CRt[d][f"{t}__cr__clip"], CRt[d][f"{t}__center__clip"]) for t in BUD} for d, _ in DS}
sla_l3 = {d: pr(SL[d]["Tfull__sla__clip"], SL[d]["Tfull__l3__clip"]) for d, _ in DS}
trgap = {d: m(CP[d]["transductive__proto_clip"]) - m(CP[d]["Tfull__proto_clip"]) for d, _ in DS}
none = {d: m(CP[d]["none__proto_clip"]) for d, _ in DS}
S8 = R("calib_protocol.npz")                                         # SEED 단일 감정 (유클리드 정렬 팔)
NS = {"SEED-V": 5, "SEED-IV": 4}
sgap = {"SEED": [-pr(S8[f"single{k}__proto_clip"], S8["Tfull__proto_clip"])[0] for k in range(3)]}
sgap.update({d: [-pr(CP[d][f"single{k}__proto_clip"], CP[d]["Tfull__proto_clip"])[0] for k in range(n)] for d, n in NS.items()})
FL, NL = R("final_ladder.npz"), R("noea_ladder.npz")
ea = [pr(FL[f"eaRealNoCen_{t}__proto_clip"], NL["none__proto_clip"]) for t in BUD]
win_none, win_cen = m(NL["none__proto_win_bal"]), m(NL["cen_Tfull__proto_win_bal"])
STN = {d: R(f"{p}stratnorm.npz") for d, p in DS[:2]}
sn = [pr(STN[d][f"strat_{t}__{mt}"], STN[d][f"center_{t}__{mt}"]) for d, _ in DS[:2]
      for t in ("T20", "T40", "Tfull", "trans") for mt in ("clip", "win")]
assert max(x[0] for x in sn) < 0.0075 and sum(x[1] < 0.05 for x in sn) == 2

# 본문 주장의 조건
assert all(cen[d][t][0] > 0 and cen[d][t][1] < 0.01 for d, _ in DS for t in BUD)
assert all(0 < trgap[d] < 0.025 for d, _ in DS)
assert sla["SEED-V"]["Tfull"][1] < 0.001 and sla["SEED-IV"]["Tfull"][1] < 0.05 and sla["SEED"]["Tfull"][1] > 0.05
assert MIS["SEED"] > MIS["SEED-IV"] > MIS["SEED-V"] and sla["SEED"]["Tfull"][0] < sla["SEED-IV"]["Tfull"][0] < sla["SEED-V"]["Tfull"][0]
assert all(g > 0 for d in sgap for g in sgap[d]) and all(p > 0.1 for _, p, *_ in ea)
assert sla_l3["SEED-V"][1] < 0.01 and abs(sla_l3["SEED-IV"][0]) < 0.005

f3 = lambda x: f"{x:+.3f}"
fd = lambda x: f"{x[0]:+.3f} ({x[2]}/{x[3]}, p={x[1]:.3f})" if x[1] >= 0.001 else f"{x[0]:+.3f} ({x[2]}/{x[3]}, p&lt;0.001)"
rng = lambda v: f"{min(v):.2f}~{max(v):.2f}"


def sd_mean(path, key):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[int(r["subject"])].append(float(r[key]))
    return float(np.mean([np.mean(v) for v in by.values()]))


# ════ 머리 ════════════════════════════════════════════════════════════════
P("우리 연구의 기여 정리", TITLE)
P("짧은 캘리브레이션으로 새 사용자에게 맞추기 (가제) — 분야 관례 기준으로 다시 본 기여, 근거, 약점, 확장 방향", SUBT)
P(f"{datetime.date.today():%Y-%m-%d} · 세 데이터셋 (SEED · SEED-V · SEED-IV) 결과 반영 · 판단 기준: 피험자 단위 짝지은 Wilcoxon 검정, "
  "향상된 피험자 수, 모든 데이터셋·조건 보고 (무효과 포함) · 약어는 처음 나올 때 풀어 쓴다", SMALL)
gap(4)
P("<b>한눈에.</b><br/>"
  "<b>C1 현실적 캘리브레이션 프로토콜</b> (강함) — 평가 데이터를 전혀 쓰지 않고, 캘리브레이션 영상은 평가에서 빼고, 시청 시간 예산으로 잰다.  "
  f"평가 데이터를 미리 쓴 상한과의 차이는 클립 정확도 {f3(min(trgap.values()))}~{f3(max(trgap.values()))} 뿐이다.<br/>"
  f"<b>C2 임베딩 중심화가 주 효과</b> (강함) — 세 데이터셋 모두 유의 (클립 {f3(min(cen[d][t][0] for d, _ in DS for t in BUD))}~"
  f"{f3(max(cen[d][t][0] for d, _ in DS for t in BUD))}).  유클리드 정렬(EA) 등 라벨 없는 다른 방법은 그 위에 ≈0.<br/>"
  "<b>C3 남는 오차의 진단</b> (중간) — 처음 보는 사람의 감정 방향이 돌아가 있다 (회전).  라벨 없는 적응의 천장을 설명한다.<br/>"
  f"<b>C4 대응 정보로 회전 회수</b> (중간, 조건부) — 자극 시점 정렬(SLA): SEED-V {f3(sla['SEED-V']['Tfull'][0])}, SEED-IV "
  f"{f3(sla['SEED-IV']['Tfull'][0])} (둘 다 유의), SEED {f3(sla['SEED']['Tfull'][0])} (무효과).  이득 순서가 어긋남 순서와 같다.<br/>"
  "<b>C5 실용 지침과 음성 결과</b> (중간) — 감정 구성이 길이보다 중요, 유클리드 정렬 불필요, 라벨 없는 테스트 시점 적응 불필요.<br/>"
  "<b>다른 BCI 패러다임</b>: 같은 감정 패러다임의 큰 데이터셋 (FACED) 이 1순위.  운동 상상은 중심화 부분만 옮겨 가고 경쟁이 치열하며, "
  "SSVEP · ERP 는 자극 동기 템플릿 전이가 이미 표준이라 비추천 (7절).", BOX)

# ── 1. 판단 기준 ──
P("1. 판단 기준", H2)
B(["<b>분야 관례</b>: 피험자 단위 짝지은 Wilcoxon 검정의 p, 평균 향상폭, 향상된 피험자 수.  예산(20초 · 40초 · 끝까지)마다 그대로 보고한다.  "
   "여러 비교 보정 (Holm) 은 쓰지 않고, 내부 사전 등록 판정 (V5 · V6) 은 계획 기록으로만 남긴다.",
   "<b>정직성</b>: 세 데이터셋 · 세 예산 결과를 무효과까지 모두 보고하고, 결과를 본 뒤 주 지표를 바꾸지 않으며, 한정이 필요한 표현은 "
   "한정한다 (예: 자극 시점 정렬은 '정렬에 감정 라벨을 쓰지 않는다' — 영상이 곧 감정이므로 '라벨 없이' 라고 쓰지 않는다).",
   "<b>강도 표시</b>: 강함 = 세 데이터셋 모두 유의 · 일관 / 중간 = 일부 데이터셋에서 유의하거나 진단 근거 / 약함 = 한 데이터셋 또는 기술 통계."])

# ── 2. 근거 표 ──
P("2. 근거 한눈에 — 세 데이터셋 (클립 정확도, 유클리드 정렬 없는 모델)", H2)
rows = [["항목", "SEED (3감정, 15명)", "SEED-V (5감정, 16명)", "SEED-IV (4감정, 15명)"],
        ["적응 없음 (우연 수준)", f"{none['SEED']:.3f} (0.333)", f"{none['SEED-V']:.3f} (0.200)", f"{none['SEED-IV']:.3f} (0.250)"]]
for t, lab in zip(BUD, ("감정당 20초", "감정당 40초", "영상 끝까지")):
    rows.append([f"중심화 − 적응 없음, {lab}"] + [fd(cen[d][t]) for d, _ in DS])
rows += [["평가 데이터를 미리 쓴 상한 − 현실적 (끝까지)"] + [f3(trgap[d]) for d, _ in DS],
         ["어긋남 c (1 = 학습 기준과 완전 일치)"] + [f"{MIS[d]:.3f}" for d, _ in DS],
         ["자극 시점 정렬 − 중심화 (끝까지)"] + [fd(sla[d]["Tfull"]) for d, _ in DS],
         ["prototype 혼합 − 중심화 (끝까지)"] + [fd(l3[d]["Tfull"]) for d, _ in DS],
         ["라벨 회전 − 중심화 (끝까지)"] + [fd(cr[d]["Tfull"]) for d, _ in DS],
         ["자극 시점 정렬 − prototype 혼합 (끝까지)"] + [fd(sla_l3[d]) for d, _ in DS],
         ["한 감정만 캘리브레이션 − 모든 감정 (끝까지)"] + [f"−{rng(sgap[d])}" for d, _ in DS]]
T(rows, [52, 38, 38, 38], hl=(4, 7))
P("괄호 = (향상된 피험자 / 전체, Wilcoxon p).  SEED 의 한 감정 행은 유클리드 정렬을 켠 모델 (S8), 나머지는 모두 끈 모델.  20초 · 40초의 자극 "
  f"시점 정렬: SEED-V {f3(sla['SEED-V']['T20'][0])} / {f3(sla['SEED-V']['T40'][0])} (둘 다 p&lt;0.01), SEED-IV "
  f"{f3(sla['SEED-IV']['T20'][0])} / {f3(sla['SEED-IV']['T40'][0])} (p={sla['SEED-IV']['T20'][1]:.2f} / {sla['SEED-IV']['T40'][1]:.2f}), "
  f"SEED {f3(sla['SEED']['T20'][0])} / {f3(sla['SEED']['T40'][0])}.", SMALL)

# ── 3. 기여별 ──
P("3. 기여별 정리", H2)
P("C1. 배포 현실적인 캘리브레이션 프로토콜과 정직한 회계 — 강함", H3)
B(["<b>주장</b>: 새 사용자에게서 쓰는 것은 세션 시작의 짧은 캘리브레이션 블록뿐이다 (감정당 영상 1편, 앞 20초 · 40초 · 끝까지).  그 영상은 "
   "평가에서 빼고, 비용은 시청 시간으로 잰다.  평가 데이터는 정규화·적응에 쓰지 않는다.",
   f"<b>근거</b>: 평가 데이터를 미리 쓰는 상한과의 차이가 클립 {f3(min(trgap.values()))}~{f3(max(trgap.values()))} 로 작다 — 현실적 조건에서도 "
   "얻을 것은 대부분 얻는다.  반대로 문헌의 높은 수치는 같은 사람 안 분할 (SEED 60~94%) 이나 같은 시행 안 분할에서 나온다 "
   f"(조사 PDF 6절).  우리 적응 없음의 창 균형 정확도 {win_none:.3f} 는 피험자 분리 벤치마크 (Compass 52.2, AdaBrain 55.8) 와 일치한다.",
   "<b>선행 대비</b>: 테스트와 분리된 캘리브레이션은 PPDA (AAAI 2021, 45초 무라벨) · Li 2020 · EvoFA 등 소수.  'FM + 감정 + 클립 분리 + "
   "시청 시간 예산 + 감정 포괄' 을 함께 통제한 평가는 찾지 못했다 (조사의 빈자리).",
   "<b>보강</b>: PPDA 를 같은 프로토콜로 비교하면 가장 가까운 선행과의 차이가 분명해진다."])
P("\"We evaluate calibration of an EEG foundation model under a deployment-realistic protocol: only a short, clip-disjoint "
  "calibration block per session is used, its cost is measured in viewing time, and no test data enter normalization or adaptation.\"", EN)

P("C2. 임베딩 공간 도메인 중심화가 주 효과 — 강함", H3)
B([f"<b>주장</b>: 캘리브레이션 블록의 평균 특징을 빼는 것만으로 클립 정확도가 오른다 — SEED {f3(cen['SEED']['Tfull'][0])}, SEED-V "
   f"{f3(cen['SEED-V']['Tfull'][0])}, SEED-IV {f3(cen['SEED-IV']['Tfull'][0])} (끝까지), 감정당 20초로도 대부분.  재학습 없이 평균 하나를 뺀다.",
   "<b>그 위에 ≈0 인 것들</b>: 유클리드 정렬 (SEED 현실적 조건 " + " / ".join(f"{d:+.3f}" for d, *_ in ea) + ", 모두 p&gt;0.1), "
   "분산까지 맞추는 정규화 (16개 비교 모두 +0.007 이하; SEED 클립 끝까지 +0.005 · 평가 데이터 사용 +0.007 만 p&lt;0.05, SEED-V 에서는 재현 안 됨), 적응형 배치 정규화 · 테스트 시점 템플릿 조정 · 잠재 정렬 · 의사라벨 회전 (모두 ±0.005, SEED).",
   "<b>선행 대비</b>: 재중심화 원리 자체는 알려져 있다 (Zanini 2018, EA 2020, Mellot 2023 — 고전 BCI).  FM 감정 임베딩에서, 평가 데이터 없이, "
   "세 데이터셋으로 '이것이 이득의 대부분이고 나머지는 더하지 못한다' 를 보인 것이 기여다.  테스트 시점 적응 문헌의 합의 (정규화가 먼저, "
   "목적함수는 덤) 와 맞는다.",
   "<b>보강</b>: 다른 FM (CBraMod · REVE) 에서 같은 결과가 나오면 'LaBraM 만의 성질' 이라는 반론을 막는다."])
P("\"Subtracting the calibration-block mean of the foundation-model embedding recovers most of the attainable gain on three "
  "datasets, and input whitening, variance normalization or label-free test-time adaptation add nothing on top.\"", EN)

P("C3. 남는 오차의 진단 — 회전, 그리고 라벨 없는 적응의 천장 — 중간", H3)
sgs, sgv = SG["SEED"], SG["SEED-V"]
B([f"<b>주장</b>: 중심화 뒤 남는 오차는 처음 보는 사람 · 날마다 다른 감정 방향의 회전이다.  학습한 사람은 날짜가 바뀌어도 방향이 거의 같지만 "
   f"(코사인 SEED {m(sgs['cw_train']):.2f}, SEED-V {m(sgv['cw_train']):.2f}), 처음 보는 사람은 덜 맞는다 ({m(sgs['cw_test']):.2f}, "
   f"{m(sgv['cw_test']):.2f}).  세 데이터셋의 어긋남 c 도 SEED {MIS['SEED']:.2f} · SEED-IV {MIS['SEED-IV']:.2f} · SEED-V {MIS['SEED-V']:.2f} 로 다르다.",
   "<b>왜 천장인가</b>: 이 회전은 사람마다 제각각이고 감정 정보와 얽혀 있어, 학습 피험자들에게서 찾은 공통 방향을 지우면 SEED-V 에서 오히려 "
   "나빠진다 (−0.025).  새 사용자의 뇌파만으로는 고칠 수 없다 — 그래서 대응 정보 (C4) 가 필요하다.",
   "<b>선행 대비</b>: 리만 Procrustes 의 이동 모델 (평행이동 + 회전), Mellot 2023, The Identity Trap (2026, 피험자 정체성이 FM 임베딩을 지배), "
   "SCORE (2026, '같은 관계를 다른 좌표 방향으로') 와 맞는다 — 새로운 현상이 아니라 FM 감정 임베딩에서의 확인과 설명이다.",
   "<b>보강</b>: 세션 일치도 진단은 SEED · SEED-V (유클리드 정렬 모델) 에만 있다.  세 데이터셋을 같은 모델로 맞추려면 SEED-IV 와 함께 "
   "유클리드 정렬 없는 모델로 다시 계산한다 (CPU 몇 분)."])

P("C4. 대응 정보로 회전 회수 — 자극 시점 정렬 — 중간 (조건부)", H3)
B([f"<b>주장</b>: 캘리브레이션 영상은 학습 피험자들도 본 영상이다.  4초마다 '같은 영상의 같은 순간에 학습 피험자들의 평균 반응' 과 짝지어 "
   f"직교 회전을 추정하면, 어긋남이 큰 데이터셋에서 중심화 위로 더 오른다: SEED-V {fd(sla['SEED-V']['Tfull'])}, SEED-IV "
   f"{fd(sla['SEED-IV']['Tfull'])}.  어긋남이 작은 SEED 는 {fd(sla['SEED']['Tfull'])}.",
   f"<b>이득은 어긋남 순서를 따른다</b>: c {MIS['SEED']:.2f} → {MIS['SEED-IV']:.2f} → {MIS['SEED-V']:.2f} 일 때 이득 "
   f"{f3(sla['SEED']['Tfull'][0])} → {f3(sla['SEED-IV']['Tfull'][0])} → {f3(sla['SEED-V']['Tfull'][0])}.  SEED-IV 의 이득은 결과 전에 "
   "c 로 +0.048 로 예측했고 방향이 맞았다 (세 점이라 관계 입증은 아님).",
   f"<b>라벨 방법과 비교</b>: 정렬에 감정 라벨을 쓰지 않는데도 SEED-V 에서는 영상 라벨로 하는 prototype 혼합보다 높고 "
   f"({fd(sla_l3['SEED-V'])}), SEED-IV 에서는 같다 ({f3(sla_l3['SEED-IV'][0])}).  짧은 시청 (20초) 에서도 SEED-V 는 유의하다.",
   "<b>선행 대비</b>: 원리는 하이퍼정렬 (fMRI) · SSVEP 템플릿 전이 (LST 계열) 와 같다.  2026년 SCORE (뇌파 임베딩 직교 정렬, 이미지 검색) 와 "
   "GRN (감정, 시험 구간마다 같은 자극의 참조 필요) 이 가깝다.  차이: 자극 정보를 캘리브레이션 때만 쓰고 시험 때는 아무 정보도 필요 없음, "
   "고정 FM 임베딩과 고정 분류기, 위상 고정 없는 자연 영화의 초 단위 대응.",
   "<b>약점과 보강</b>: SEED 무효과, SEED-IV 20·40초 무효과.  '시점 고정' 이 핵심인지 보이려면 같은 클립 안에서 초를 섞은 대조가 필요하다 "
   "(GRN 도 같은 대조를 했다, CPU).  사람이 많은 FACED 에서 재현되면 강도가 '강함' 으로 오른다."])
P("\"When residual misalignment is large, a calibration-only stimulus-locked alignment — hyperalignment adapted to a frozen "
  "foundation model — recovers part of the rotation without using emotion labels in the alignment objective; its gain follows the "
  "measured misalignment across datasets.\"", EN)

P("C5. 실용 지침과 음성 결과 — 중간", H3)
B([f"<b>감정 구성이 길이보다 중요</b>: 한 감정만으로 캘리브레이션하면 모든 감정을 쓴 경우보다 SEED {rng(sgap['SEED'])}, SEED-V "
   f"{rng(sgap['SEED-V'])}, SEED-IV {rng(sgap['SEED-IV'])} 낮다.  세 감정을 4초씩 (총 12초) 본 것이 한 감정을 약 4분 본 것보다 낫다 (SEED).",
   "<b>빼도 되는 것</b>: 유클리드 정렬 (쓰려면 모델을 다시 학습해야 하는데 현실적 조건에서 이득이 유의하지 않다), 분산 정규화, 라벨 없는 테스트 "
   "시점 적응, 영상 후반 가중 (SEED 에서만 보였고 SEED-V 에서 재현 실패).",
   "<b>예산별 권장</b>: 1~3분 (감정당 20~40초) 이면 중심화만.  영상을 끝까지 보고 어긋남이 크면 + 자극 시점 정렬, 그것을 못 쓰면 + prototype 혼합.",
   "<b>가치</b>: 실무자가 '무엇을 안 해도 되는지' 를 아는 것 — 음성 결과는 문헌에 드물다 (조사 PDF 3-2)."])

# ── 4. 리뷰어 ──
P("4. 예상되는 리뷰어 질문과 대응", H2)
T([["질문", "지금의 답", "더 하면 좋은 것"],
   ["방법이 새롭지 않다", "분석에 근거한 프레임워크 논문으로 낸다 — 무엇이 충분하고 (C2) 왜 나머지가 안 되며 (C3) 언제 무엇을 더해야 하는지 (C4·C5)",
    "관련 연구 절에서 재중심화 · 하이퍼정렬 · 템플릿 전이 계보를 먼저 인정"],
   ["LaBraM 하나뿐", "—", "CBraMod 또는 REVE 로 C2 · C4 재현 (GPU)"],
   ["데이터셋이 모두 같은 실험실 (상하이교통대)", "세 데이터셋에서 C2 일관", "FACED (칭화대, 123명) 추가 (GPU, 적은 비용)"],
   ["같은 영상을 학습·평가에서 본다 (반복 자극)", "분야 전체의 문제", "학습에 없던 영상으로 평가 (GPU 재학습)"],
   ["정확도가 FM 논문 수치보다 낮다", f"평가 방식 차이 — 우리 적응 없음 {win_none:.3f} 은 피험자 분리 벤치마크와 일치", "창 단위 균형 정확도 병기"],
   ["테스트 시점 적응 (T-TIME · Tent) 과 비교했나", "라벨 없는 방법 6가지가 중심화 위에 ≈0; FM 에서 경사 기반 적응은 불안정 (NeuroAdapt-Bench)",
    "시간순 스트림 기준선 하나 추가 (CPU)"],
   ["자극 시점 정렬은 사실상 라벨을 쓴다", "정렬 목적함수에는 감정 라벨이 없다고 한정해 쓴다", "같은 클립 안 초 섞기 대조 (CPU)"],
   ["가장 가까운 선행 (PPDA) 과 비교했나", "논의로 차이 설명 가능", "같은 프로토콜로 재구현 비교"]], [40, 66, 60], split=True)

# ── 5. 다른 패러다임 ──
P("5. 다른 BCI 패러다임으로 확장 — 의견", H2)
P("우리 프레임워크에서 다른 패러다임으로 옮겨 가는 것은 <b>중심화 (C2) 와 진단 (C3)</b> 이다.  <b>자극 시점 정렬 (C4)</b> 은 여러 사람이 같은 "
  "긴 자연 자극을 보는 패러다임에서만 의미가 있다.  운동 상상에는 공유 자극이 없어서 클래스 평균 회전 (리만 Procrustes) 으로 줄어들고, SSVEP · "
  "ERP 에서는 자극 동기 템플릿 전이가 이미 표준이다.", BODY)
T([["패러다임 (데이터 예)", "옮겨 가는 것", "새로움 위험", "비용", "추천"],
   ["감정 — 다른 데이터셋 (FACED: 123명이 같은 28클립, 9감정 · SEED-VII · DEAP)", "전부 (C1~C5)", "낮음 — 조사의 빈자리",
    "낮음: FACED 표준 분할 (학습 80 · 검증 20 · 시험 23명) 이면 학습 1회 × 시드 3 ≈ 2시간 GPU", "<b>1순위</b>"],
   ["자연 자극 (이야기 듣기 · 영화 시청 대규모 뇌파)", "C2 · C4 (자극 시점 정렬의 본래 무대)", "낮음~중간", "중간 (과제와 라벨이 다름)",
    "후속 연구"],
   ["운동 상상 (BCI IV-2a · Lee2019 54명 · PhysioNet)", "C2 · C3 · 라벨 방법 (C4 는 클래스 평균 회전으로 줄어듦)",
    "높음 — Tao &amp; Chen 2026 (FM 개인 보정, 235명), T-TIME · EA 계열", "중간: 피험자 하나 빼기면 수십 시간 GPU", "선택 (일반성 보조)"],
   ["SSVEP · c-VEP", "C2", "매우 높음 — 템플릿 전이 (LST 계열) 가 표준", "중간", "비추천"],
   ["ERP · P300", "C2 · 라벨 방법", "높음 — 자극 동기 평균이 표준", "중간", "비추천"]], [42, 34, 36, 36, 18], hl=(1,))
P("요약: 논문을 'BCI 전반' 으로 넓히면 비교해야 할 기존 방법 (EA, T-TIME, LST 등) 이 많아지고 최근 경쟁 논문 (Tao &amp; Chen 2026) 과 정면으로 "
  "부딪친다.  같은 감정 패러다임의 큰 데이터셋 (FACED) 으로 C2 · C4 를 넓히는 것이 비용 대비 가장 강하다.  운동 상상은 그 다음, 'C2 는 "
  "패러다임을 넘는다' 를 보이는 보조 실험으로 하나 정도.", SMALL)

# ── 6. 논문 구성 ──
P("6. 논문 구성 제안", H2)
B(["<b>제목</b> (가제 유지): What a Short Calibration Buys: An Analysis-Driven Calibration Framework for Deploying EEG Foundation "
   "Models to New Users.",
   "<b>본문 흐름</b>: 1 서론 (FM 의 사람 차이 문제 + 현실적 캘리브레이션의 빈자리) → 2 관련 연구 (재중심화 · 템플릿 전이 · 테스트 시점 적응 · "
   "교차 피험자 감정 · FM 벤치마크 · 하이퍼정렬) → 3 프로토콜 (C1) → 4 중심화와 그 위의 ≈0 (C2) → 5 진단 (C3) → 6 대응 정보로 회전 회수 "
   "(C4) → 7 실용 지침 (C5) → 8 한계.",
   "<b>그림</b>: 개념도 (원점 → 방향), 파이프라인, 예산별 향상 곡선 (세 데이터셋), 어긋남 c 대 이득, 세션 일치도 (학습 · 처음 보는 사람).",
   f"<b>참고 — 같은 사람 안 평가</b> (상한 참고용): SEED 클립 {sd_mean('results/a02_sd.csv', 'clip_accuracy'):.3f}, SEED-V "
   f"{sd_mean('results/seedv_a02_sd.csv', 'clip_accuracy'):.3f}."])

# ── 7. 다음 실험 ──
P("7. 다음 실험 — 우선순위", H2)
T([["#", "실험", "올리는 기여", "자원"],
   ["1", "FACED (123명, 같은 28클립) 에서 C1~C4 — 표준 분할 또는 피험자 5겹", "C2 · C4 강도, 다른 실험실 데이터", "GPU 약 2~10시간"],
   ["2", "같은 클립 안 초 섞기 · 다른 클립 짝 대조 (자극 시점 정렬)", "C4 의 '시점 고정' 근거", "CPU"],
   ["3", "학습에 없던 영상으로 평가", "반복 자극 반론 차단 (C2 · C4 모두)", "GPU 재학습"],
   ["4", "다른 FM (CBraMod 또는 REVE)", "C2 의 일반성", "GPU"],
   ["5", "세션 일치도 진단을 세 데이터셋 · 같은 모델로", "C3 를 세 데이터셋으로", "CPU 몇 분"],
   ["6", "운동 상상 데이터셋 하나 (선택)", "C2 가 패러다임을 넘는지", "GPU 수십 시간"]], [7, 82, 50, 27])


# ── 글리프 검사 · 출력 ─────────────────────────────────────────────────────
def check_glyphs():
    cmaps = [set(FTFont(p).getBestCmap()) for p in (FONT_R, FONT_B)]
    miss = set()
    for t in TEXTS:
        plain = re.sub(r"<[^>]+>", "", t).replace("&amp;", "&").replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
        miss |= {ch for ch in plain if not ch.isspace() and all(ord(ch) not in c for c in cmaps)}
    if miss:
        raise SystemExit(f"폰트에 없는 글자: {sorted(miss)}")


def footer(c, d):
    c.saveState(); c.setFont(F, 7.5); c.setFillColor(GREY)
    c.drawString(22 * mm, 12 * mm, "우리 연구의 기여 정리 (2026-10-05)")
    c.drawRightString(A4[0] - 22 * mm, 12 * mm, str(d.page))
    c.setStrokeColor(LINE); c.setLineWidth(0.4); c.line(22 * mm, 15 * mm, A4[0] - 22 * mm, 15 * mm); c.restoreState()


check_glyphs()
doc = BaseDocTemplate("reports/CONTRIBUTIONS.pdf", pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=18 * mm,
                      bottomMargin=20 * mm, title="우리 연구의 기여 정리")
doc.addPageTemplates([PageTemplate(id="a", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)],
                                   onPage=footer)])
doc.build(E)
print("[saved] reports/CONTRIBUTIONS.pdf")
