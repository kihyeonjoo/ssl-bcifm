"""연구 전체 총정리 (가제 포함, 누구나 이해할 수 있게) — reports/OVERVIEW.pdf.

전문 용어는 처음 나올 때 "풀어 쓴 이름(약어)" 으로 소개하고 끝에 용어 풀이를 둔다.  수치는 results/ 와 dataset_config 에서
직접 읽는다 (V3 의 사영 제거 −0.025 만 결과 파일이 없어 V3_SESSION_GEOMETRY_PREREG.md 2b 표에서 옮겼다).

    python make_overview_pdf.py
"""
from __future__ import annotations

import csv
import datetime
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
import numpy as np
from scipy import stats
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
                                KeepTogether, Image)

from dataset_config import _ALL as CFGS

pdfmetrics.registerFont(TTFont("NotoKR", ".fonts/NotoSansKR-regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoKR-B", ".fonts/NotoSansKR-bold.ttf"))
registerFontFamily("NotoKR", normal="NotoKR", bold="NotoKR-B", italic="NotoKR", boldItalic="NotoKR-B")
F, FB = "NotoKR", "NotoKR-B"
INK, GREY, ACC = colors.HexColor("#1a1d21"), colors.HexColor("#6f757b"), colors.HexColor("#c1553b")
LINE, BG, NOTE = colors.HexColor("#d4d9dd"), colors.HexColor("#f5f7f9"), colors.HexColor("#eef4ef")


def st(name, size=9.5, lead=14.6, color=INK, font=F, space=4, left=0, align=TA_LEFT):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead, textColor=color, spaceAfter=space,
                          leftIndent=left, alignment=align)


BODY = st("b"); BUL = st("bul", left=10)
TITLE = st("t", 19, 26, INK, FB, 6); SUBT = st("st", 12, 17, INK, F, 6); ENG = st("en", 9.6, 14, GREY, F, 10)
H2 = st("h2", 12.5, 17, ACC, FB, 6); H2.keepWithNext = 1
H3 = st("h3", 10.2, 14.5, INK, FB, 3); H3.keepWithNext = 1
SMALL = st("s", 8.2, 12, GREY, space=3)
CELL = st("c", 8.2, 11.4, INK, space=0); CELLB = st("cb", 8.2, 11.4, INK, FB, space=0)
BOX = ParagraphStyle("box", fontName=F, fontSize=9.6, leading=15, textColor=INK, backColor=NOTE, borderPadding=8,
                     spaceAfter=9, leftIndent=4, rightIndent=4)
E = []


def P(t, s=BODY):
    E.append(Paragraph(t, s))


def B(items):
    for t in items:
        E.append(Paragraph("• " + t, BUL))


def gap(h=6):
    E.append(Spacer(1, h))


def _with_headings(group):
    """바로 앞의 제목들 (H2, H3) 을 표·그림과 한 덩어리로 묶어 제목만 쪽 끝에 남지 않게 한다."""
    while E and isinstance(E[-1], Paragraph) and E[-1].style in (H2, H3):
        group.insert(0, E.pop())
    E.append(KeepTogether(group))


def T(rows, widths, hl=()):
    data = [[Paragraph(str(c), CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], hAlign="LEFT")
    s = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.2),
         ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 4),
         ("BACKGROUND", (0, 0), (-1, 0), BG), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
         ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE),
         ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK)]
    for r in hl:
        s.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fbeae4")))
    t.setStyle(TableStyle(s))
    _with_headings([t]); gap(5)


def fig(path, cap, width=166):
    w, h = PILImage.open(path).size
    _with_headings([Image(path, width=width * mm, height=width * mm * h / w), Paragraph(cap, SMALL)]); gap(6)


# ── 수치 ─────────────────────────────────────────────────────────────────
L = lambda f: np.load(f"results/{f}")
m = lambda a: float(np.asarray(a, float).mean())


def pr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    return float(d.mean()), float(stats.wilcoxon(a, b).pvalue), int((d > 0).sum()), len(d)


def holm(ps):
    ps = np.asarray(ps, float); n = len(ps); out = np.empty(n); mx = 0.0
    for r, i in enumerate(np.argsort(ps)):
        mx = max(mx, min(1.0, ps[i] * (n - r))); out[i] = mx
    return out


DS = (("SEED", ""), ("SEED-V", "seedv_"))
BUD = (("T20", "감정당 20초"), ("T40", "감정당 40초"), ("Tfull", "영상 끝까지"))
DS3 = DS + (("SEED-IV", "seediv_"),)                                  # 세 번째 데이터셋 (V6, 2026-10-05)
NCFG = {"SEED": CFGS["seed"], "SEED-V": CFGS["seedv"], "SEED-IV": CFGS["seediv"]}
SEC = {d: {"T20": 20.0, "T40": 40.0, "Tfull": c.median_clip_sec} for d, c in NCFG.items()}   # 감정당 시청 초 (중앙값)
MIN = {d: {t: s * NCFG[d].n_classes / 60 for t, s in SEC[d].items()} for d in SEC}
PAIRS = {d: {t: int(s // NCFG[d].sec_per_win) * NCFG[d].n_classes for t, s in SEC[d].items()} for d in SEC}
CP = {d: L(f"{p}calib_protocol_noea_r10.npz") for d, p in DS3}
SL = {d: L(f"{p}sla_noea.npz") for d, p in DS3}
CRt = {d: L(f"{p}calib_rotation_noea.npz") for d, p in DS3}
SG = {d: L(f"{p}session_geometry.npz") for d, p in DS}
MIS = {d: L(f"{p}misalignment_noea.npz") for d, p in DS3}
STN = {d: L(f"{p}stratnorm.npz") for d, p in DS}
TC = {d: L(f"{p}timecourse.npz") for d, p in DS}
C4 = L("centering_analysis.npz"); FL = L("final_ladder.npz"); NL = L("noea_ladder.npz"); VR = L("seedv_ea_realistic.npz")
CP32 = L("calib_protocol_noea_r10_fp32.npz"); S8 = L("calib_protocol.npz")
cen = {d: {t: pr(CP[d][f"{t}__proto_clip"], CP[d]["none__proto_clip"]) for t, _ in BUD} for d, _ in DS3}
sla = {d: {t: pr(SL[d][f"{t}__sla__clip"], SL[d][f"{t}__center__clip"]) for t, _ in BUD} for d, _ in DS3}
cr = {d: {t: pr(CRt[d][f"{t}__cr__clip"], CRt[d][f"{t}__center__clip"]) for t, _ in BUD} for d, _ in DS3}
l3 = {d: {t: pr(SL[d][f"{t}__l3__clip"], SL[d][f"{t}__center__clip"]) for t, _ in BUD} for d, _ in DS3}
sla_holm4 = holm([sla["SEED-IV"][t][1] for t, _ in BUD])[2]            # V6 주 판정 (Holm 족 3) 의 클립 전체
sl3_4 = pr(SL["SEED-IV"]["Tfull__sla__clip"], SL["SEED-IV"]["Tfull__l3__clip"])
# 한 감정 캘리브레이션 (no-EA r10): 모든 감정 끝까지 대비 낮은 폭의 범위
nsing = {"SEED-V": 5, "SEED-IV": 4}
sgap = {d: [-pr(CP[d][f"single{k}__proto_clip"], CP[d]["Tfull__proto_clip"])[0] for k in range(nsing[d])] for d in nsing}
snone = {d: [pr(CP[d][f"single{k}__proto_clip"], CP[d]["none__proto_clip"])[0] for k in range(nsing[d])] for d in nsing}
pred4 = 0.0065 + (0.754 - m(MIS["SEED-IV"]["c"])) / (0.754 - 0.322) * (0.0610 - 0.0065)     # V6 사전 등록 예측식
sl3 = pr(SL["SEED-V"]["Tfull__sla__clip"], SL["SEED-V"]["Tfull__l3__clip"])
both = pr(SL["SEED-V"]["Tfull__slal3__clip"], SL["SEED-V"]["Tfull__sla__clip"])
single_full = [pr(S8[f"single{k}__proto_clip"], S8["Tfull__proto_clip"]) for k in range(3)]   # 유클리드 정렬 팔 (S8)
single_none = [pr(S8[f"single{k}__proto_clip"], S8["none__proto_clip"]) for k in range(3)]
# S8 의 키 T2 = 감정별 4초 (총 12초) — S8 표의 0.6596 과 같은 값으로 대응을 확인했다
assert abs(m(S8["T2__proto_clip"]) - 0.6596) < 5e-5
comp = pr(S8["T2__proto_clip"], np.mean([S8[f"single{k}__proto_clip"] for k in range(3)], axis=0))
sf = f"{min(-x[0] for x in single_full):.2f}~{max(-x[0] for x in single_full):.2f}"   # 모든 감정 대비 낮은 폭
ps16 = [pr(STN[d][f"strat_{t}__{mt}"], STN[d][f"center_{t}__{mt}"])[1]
        for d, _ in DS for t in ("T20", "T40", "Tfull", "trans") for mt in ("clip", "win")]
cen_pmax = max(cen[d][t][1] for d, _ in DS3 for t, _ in BUD)
share = {d: cen[d]["T20"][0] / cen[d]["Tfull"][0] for d, _ in DS3}
assert all(cen["SEED-IV"][t][2] >= 12 for t, _ in BUD) and 0.05 < sla_holm4 < 0.1 and sla["SEED-IV"]["Tfull"][1] < 0.05
assert all(g > 0 for d in sgap for g in sgap[d]) and all(x > -0.005 for x in snone["SEED-IV"]) and abs(sl3_4[0]) < 0.005
assert cen_pmax < 0.01 and all(x[0] <= 0 and x[2] == 0 for x in single_full) and all(x[0] <= 0 for x in single_none)
assert sla["SEED-V"]["T20"][1] < 0.05 and sl3[1] < 0.05 and both[1] > 0.05 and 0.95 <= share["SEED-V"] <= 1.1
assert all(cr[d]["T20"][1] > 0.05 and l3[d]["T20"][1] > 0.05 for d, _ in DS)


def sd_mean(path, key):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[int(r["subject"])].append(float(r[key]))
    return float(np.mean([np.mean(v) for v in by.values()]))


fmt = lambda x: f"{x[0]:+.3f} ({x[2]}/{x[3]})"                       # 표 안: 향상폭 (향상된 사람 / 전체)
fmtw = lambda x: f"{x[0]:+.3f} ({x[3]}명 중 {x[2]}명 향상)"           # 본문
mn = lambda d, t: f"{MIN[d][t]:.1f}".rstrip("0").rstrip(".") if t != "Tfull" else f"약 {MIN[d][t]:.0f}"

# ════ 표지 ════════════════════════════════════════════════════════════════
gap(18)
P("가제", SMALL)
P("짧은 캘리브레이션으로 새 사용자에게 맞추기", TITLE)
P("EEG 파운데이션 모델 감정 인식을 위한, 분석에 근거한 캘리브레이션 프레임워크", SUBT)
P("What a Short Calibration Buys: An Analysis-Driven Calibration Framework for Deploying EEG Foundation Models "
  "to New Users", ENG)
P(f"연구 총정리 · {datetime.date.today():%Y-%m-%d} · 상태: SEED · SEED-V · SEED-IV 분석 완료", SMALL)
gap(8)
P("<b>한 줄 요약.</b>  뇌파로 감정을 읽는 인공지능은 처음 보는 사람에게 약하다.  새 사용자가 처음에 잠깐 영상을 보는 "
  "'캘리브레이션' 시간 동안 무엇을 얼마나 고칠 수 있는지를, 평가 데이터를 전혀 미리 쓰지 않는 현실적 조건에서 측정했다.  "
  f"<b>평균 하나를 빼는 것만으로</b> 정확도가 크게 오르고 (SEED {cen['SEED']['Tfull'][0]:+.3f}, SEED-V "
  f"{cen['SEED-V']['Tfull'][0]:+.3f}, SEED-IV {cen['SEED-IV']['Tfull'][0]:+.3f} — 세 데이터셋 모두 사전 등록 기준으로 재현), "
  "더 복잡한 라벨 없는 방법은 여기에 더하는 것이 없다.  남는 오차는 사람·날마다 다른 '감정 방향의 회전' 이고, 같은 영상의 "
  f"같은 순간을 짝지어 일부 되찾을 수 있다 (SEED-V {sla['SEED-V']['Tfull'][0]:+.3f}, SEED-IV {sla['SEED-IV']['Tfull'][0]:+.3f}) — "
  "다만 효과가 데이터셋마다 다르고 사전 등록한 재현 기준은 통과하지 못했다.", BOX)

# ── 1. 한 쪽 요약 ──
P("1. 한 쪽 요약", H2)
T([["", "내용"],
   ["문제", "뇌파(EEG) 감정 인식 모델은 학습에 없던 사람에게 성능이 크게 떨어진다.  실제로 쓰려면 새 사용자에게 맞추는 "
    "과정(캘리브레이션)이 필요한데, 그 시간이 곧 사용자의 부담이다."],
   ["질문", "① 짧은 캘리브레이션으로 무엇을 얼마나 얻는가?  ② 무엇이 남고 왜 남는가?  ③ 남은 것을 이미 쓴 시간 안에서 "
    "되찾을 수 있는가?"],
   ["방법", "대규모 사전학습 뇌파 모델(LaBraM)을 다른 사람들의 데이터로 미세조정한 뒤, 새 사람에게는 캘리브레이션 데이터만 "
    "쓰고 평가 데이터는 절대 미리 보지 않는다.  비용은 시청 시간으로 잰다.  결과를 보기 전에 판정 기준을 고정하고 "
    "(사전 등록), 다른 데이터셋에서 재현한다."],
   ["결과 ①", f"캘리브레이션 녹화의 <b>평균을 빼는 것(도메인 중심화)</b> 하나로 크게 오른다 — SEED "
    f"{fmtw(cen['SEED']['Tfull'])}, SEED-V {fmtw(cen['SEED-V']['Tfull'])}, SEED-IV {fmtw(cen['SEED-IV']['Tfull'])} (영상 끝까지 본 경우).  더 복잡한 라벨 없는 "
    "방법들은 여기에 더하는 것이 없다."],
   ["결과 ②", "남는 오차는 사람·날마다 다르게 '돌아가 있는' 감정 방향이다.  새 사용자의 뇌파만 보고는 이것을 고칠 수 "
    "없어서, 중심화가 라벨 없는 적응의 천장이 된다.  고치려면 추가 정보가 필요하다."],
   ["결과 ③", "추가 정보로 감정 라벨 대신 <b>'어떤 영상의 몇 초를 보고 있었나'</b> 를 쓴다.  새 사용자와 학습 피험자들이 "
    "같은 영상의 같은 순간에 보인 반응을 짝지어 회전을 추정하면 (자극 시점 정렬) SEED-V 에서 "
    f"{fmtw(sla['SEED-V']['Tfull'])} 더 오른다 — 감정 라벨을 쓰는 방법보다 높다.  SEED-IV 는 {sla['SEED-IV']['Tfull'][0]:+.3f} 로 "
    f"방향은 같지만 사전 등록 기준에 못 미쳤고, 회전이 작은 SEED 에서는 효과가 없다 ({sla['SEED']['Tfull'][0]:+.3f})."],
   ["의미", "각 단계를 언제 쓰고 언제 빼야 하는지가 측정으로 정해진 <b>캘리브레이션 프레임워크</b>.  방법 하나하나는 기존 "
    "기법이지만, 무엇이 필요하고 무엇이 불필요한지를 보인 것이 기여다."]], [20, 146])

# ── 2. 배경 ──
P("2. 배경 — 쉬운 말로", H2)
P("2-1. 뇌파로 감정 읽기", H3)
P("머리에 62개의 전극을 붙이고 뇌의 전기 신호(EEG)를 기록하면서, 슬프거나 즐거운 감정을 유도하는 영화 장면을 보여 준다.  "
  "기록을 4초 단위의 <b>창(window)</b> 으로 잘라 인공지능이 감정을 맞히게 한다.  영상 한 편 전체를 <b>클립(clip)</b> 이라 "
  "부르고, 클립 안의 모든 창을 모아 한 번 내린 판정도 따로 잰다 (사람이 영상 한 편을 보는 동안의 감정을 맞히는 셈).")
P("2-2. 처음 보는 사람이 어려운 이유", H3)
P("같은 감정이라도 사람마다 뇌파가 다르게 기록된다 — 머리 모양, 전극 위치, 피부 상태, 개인의 반응 방식이 다르기 때문이다.  "
  "같은 사람도 날마다 전극을 다시 붙이면 달라진다.  그래서 여러 사람으로 학습한 모델도 <b>처음 보는 사람</b>에게는 성능이 "
  "크게 떨어진다.  이를 평가하는 표준 방법이 <b>피험자 하나 빼기 교차검증(LOSO)</b> 이다: 한 사람을 완전히 빼 두고 나머지로 "
  "학습한 뒤 빠진 사람으로 시험하고, 모든 사람에 대해 돌아가며 반복한다.")
P("2-3. 파운데이션 모델", H3)
P("수천 시간의 다양한 뇌파로 미리 학습해 둔 큰 모델을 <b>파운데이션 모델(FM)</b> 이라 한다.  이 연구는 공개 모델 "
  "LaBraM 을 감정 분류에 맞게 다른 사람들의 데이터로 미세조정해 쓴다.  모델은 4초 창마다 200개의 숫자로 된 특징을 낸다.")
P("2-4. 캘리브레이션 — 비용은 시청 시간", H3)
P("새 사용자는 쓰는 날마다 처음에 감정마다 영상 한 편을 보며 짧게 녹화한다 (전극을 다시 붙이면 신호가 달라지므로 날마다 "
  "한다).  그 데이터로 모델을 그날의 그 사람에게 맞춘다.  사용자가 화면 앞에 앉아 있어야 하는 시간이 곧 비용이므로, 이 "
  "연구는 효과를 <b>시청 시간</b>에 대해 잰다: 감정당 20초 · 40초 · 영상 끝까지.  모두 합치면 SEED (감정 3개) 는 "
  f"{mn('SEED', 'T20')}분 · {mn('SEED', 'T40')}분 · {mn('SEED', 'Tfull')}분, SEED-V (감정 5개) 는 {mn('SEED-V', 'T20')}분 · "
  f"{mn('SEED-V', 'T40')}분 · {mn('SEED-V', 'Tfull')}분, SEED-IV (감정 4개) 는 {mn('SEED-IV', 'T20')}분 · {mn('SEED-IV', 'T40')}분 · "
  f"{mn('SEED-IV', 'Tfull')}분이다 (영상 길이는 중앙값 기준).")
P("2-5. 기존 연구가 놓친 것", H3)
P("많은 연구가 평가할 데이터를 정규화나 적응에 <b>미리</b> 쓴다 (전달식, transductive).  실제 사용에서는 미래의 데이터를 "
  "미리 볼 수 없으므로, 이렇게 얻은 성능은 실제보다 좋게 나온다.  이 연구는 이것을 엄격히 금지했다.  같은 모델 (SEED, "
  f"유클리드 정렬을 켠 모델) 에서 그 차이는 클립 정확도 {m(C4['c4_clip']):.3f} (평가 데이터 사용) 대 "
  f"{m(FL['eaReal_Tfull__proto_clip']):.3f} (캘리브레이션만, {mn('SEED', 'Tfull')}분) 이다.")

# ── 3. 연구 질문 ──
P("3. 연구 질문", H2)
T([["#", "질문", "답 (요약)"],
   ["Q1", "짧은 캘리브레이션으로 무엇을 얼마나 얻는가?",
    f"평균 빼기(중심화).  모든 감정을 포함하면 감정당 20초로도 이득의 대부분을 얻는다 (SEED 는 끝까지 본 효과의 "
    f"{share['SEED']:.0%}, SEED-IV 는 {share['SEED-IV']:.0%}, SEED-V 는 같은 수준)"],
   ["Q2", "무엇이 남고 왜 남는가?", "사람·날마다 다른 감정 방향의 회전.  처음 보는 사람에게 일반화가 안 되는 부분이다"],
   ["Q3", "남은 것을 이미 쓴 시간 안에서 되찾을 수 있는가?",
    "'어떤 영상의 몇 초' 정보로 같은 순간을 짝지으면 일부 (회전이 큰 데이터셋에서).  사전 등록 재현 기준은 미달"]],
  [10, 70, 86])

# ── 4. 데이터와 평가 ──
P("4. 데이터와 평가 방법", H2)
T([["데이터셋", "사람", "감정", "영상", "역할"],
   ["SEED", "15명 × 3일", "3 (부정·중립·긍정)", "15편, 세 날 모두 같은 영상", "개발"],
   ["SEED-V", "16명 × 3일", "5 (혐오·공포·슬픔·중립·행복)", "날마다 다른 15편", "<b>외부 검증</b> — LaBraM 사전학습에 없음"],
   ["SEED-IV", "15명 × 3일", "4 (중립·슬픔·공포·행복)", "날마다 다른 24편", "세 번째 재현"]], [22, 22, 44, 44, 34])
B(["<b>평가</b>: 피험자 하나 빼기 교차검증 (LOSO).  새 사람의 녹화 날마다, 감정당 영상 1편을 캘리브레이션으로 쓰고 "
   "(앞 20초 · 40초 · 끝까지), 그 영상은 평가에서 뺀 뒤 나머지 영상으로 창 정확도와 클립 정확도를 잰다.  어떤 영상을 "
   "캘리브레이션으로 쓸지 무작위로 10번 바꿔 뽑고, 모델은 시드 3개로 반복해 평균한다.",
   "<b>공정성 장치</b>: 결과를 보기 전에 판정 기준을 문서로 고정 (사전 등록) · 사람 단위의 짝지은 통계 검정 · 여러 비교를 "
   "한꺼번에 할 때의 보정 (Holm) · 이 표본으로 감지할 수 있는 최소 효과 (최소 감지 효과, MDE) 를 미리 계산 · 주요 보고서 "
   "수치를 결과 파일과 자동 대조 (268개 항목)."])

# ── 5. 프레임워크 ──
P("5. 프레임워크 — 원점을 맞추고, 방향을 맞춘다", H2)
P("새 사용자의 특징은 학습 피험자들과 비교할 때 <b>통째로 밀려 있고 (오프셋)</b>, 거기에 더해 감정 방향이 <b>돌아가 있다 "
  "(회전)</b>.  지도 두 장을 겹치는 일에 비유하면, 먼저 원점을 맞추고 그다음 북쪽을 맞추는 것이다.")
fig("figs/fig_concept.png", "그림 1.  개념 도식 (실제 데이터가 아님).  (a) 학습 피험자들의 감정 위치와 기준점(별).  (b) 새 사용자는 "
    "통째로 밀리고 돌아가 있다.  (c) 중심화(평균 빼기)로 원점은 맞지만 방향은 어긋나 있다.  (d) 자극 시점 정렬로 방향까지 맞춘다.")
T([["단계", "하는 일", "언제", "근거 (클립 정확도)"],
   ["① 캘리브레이션 설계", "감정마다 영상 1편, 모든 감정 포함, 본 것은 다 쓴다", "항상",
    f"한 감정만 쓰면 모든 감정을 쓸 때보다 낮다 (SEED {sf}, SEED-V · SEED-IV 0.02~0.06)"],
   ["② 도메인 중심화", "캘리브레이션 녹화의 평균 특징을 뺀다 (원점 맞춤)", "<b>항상</b>", "주 효과, 라벨 없음"],
   ["③ 자극 시점 정렬 (SLA)", "같은 영상의 같은 순간끼리 짝지어 회전을 추정해 되돌린다 (방향 맞춤)",
    "학습 피험자들이 본 영상으로 캘리브레이션하고, 어긋남이 클 때",
    f"SEED-V {sla['SEED-V']['Tfull'][0]:+.3f}, SEED-IV {sla['SEED-IV']['Tfull'][0]:+.3f}, SEED {sla['SEED']['Tfull'][0]:+.3f}"],
   ["④ prototype 혼합 (L3)", "캘리브레이션 영상의 감정 라벨로 기준점을 그 사람 쪽으로 옮긴다",
    "③을 못 쓰고, 감정당 40초 이상 볼 때",
    f"SEED-V {l3['SEED-V']['T40'][0]:+.3f} (40초) ~ {l3['SEED-V']['Tfull'][0]:+.3f} (끝까지), SEED-IV {l3['SEED-IV']['Tfull'][0]:+.3f} (끝까지)"],
   ["⑤ 뺄 것", "유클리드 정렬(EA) · 분산 정규화 · 라벨 없는 다른 적응 · ③ 위에 ④ 더하기", "—", "효과 없음"],
   ["⑥ 평가 원칙", "평가 데이터 미사용 · 사전 등록 · 다중 데이터셋 재현", "항상", "신뢰의 근거"]], [30, 58, 40, 38],
  hl=(2, 3))
fig("figs/fig_pipeline.png", "그림 2.  파이프라인.  위: 새 사용자의 추론 경로 — 4초 창이 왼쪽에서 오른쪽으로 흐른다 (① 유클리드 정렬은 "
    "입력 단계라 모델도 그 입력으로 학습해야 해서 주 파이프라인에서는 끈다).  가운데: 캘리브레이션이 각 단계에 공급하는 것 (색 = "
    "쓰는 정보).  아래: 학습 때 다른 사람들로 미리 만들어 두는 것.", width=144)

# ── 6. 결과 ──
P("6. 결과", H2)
P("6-1. 평균 하나만 빼도 크게 오른다 (주 결과)", H3)
rows = [["클립 정확도", "적응 없음"] + [b for _, b in BUD]]
for d, _ in DS3:
    rows.append([d, f"{m(CP[d]['none__proto_clip']):.3f}"] +
                [f"{m(CP[d][f'{t}__proto_clip']):.3f} ({cen[d][t][0]:+.3f}, {cen[d][t][2]}/{cen[d][t][3]})" for t, _ in BUD])
T(rows, [24, 22, 40, 40, 40])
vr = [pr(VR[f"eaReal_{t}__proto_clip"], VR[f"eaRealNoCen_{t}__proto_clip"]) for t, _ in BUD]
B([f"우연 수준 (아무렇게나 찍을 때): SEED {1 / 3:.3f}, SEED-V {1 / 5:.3f}, SEED-IV {1 / 4:.3f}.  괄호 안은 향상폭과 '향상된 사람 수 / "
   f"전체'.  아홉 칸 모두 통계적으로 유의하다 (모두 p &lt; 0.01).",
   "<b>사전학습에 없던 SEED-V 와 세 번째 데이터셋 SEED-IV 에서도 재현됐다.</b>  둘 다 결과를 보기 전에 정한 기준으로 판정했다 "
   "(SEED-V 판정은 유클리드 정렬을 켠 모델로 " + " / ".join(f"{d:+.3f}" for d, *_ in vr) + ", SEED-IV 는 세 예산 모두 15명 중 "
   f"{min(cen['SEED-IV'][t][2] for t, _ in BUD)}명 이상 향상).",
   f"<b>구성이 길이보다 중요하다.</b>  세 감정을 4초씩 (총 12초) 본 것이 한 감정을 약 4분 본 것보다 낫다 "
   f"({comp[0]:+.3f}, {comp[3]}명 중 {comp[2]}명; SEED, 유클리드 정렬을 켠 모델).  한 감정만으로 캘리브레이션하면 모든 감정을 끝까지 "
   f"쓴 경우보다 낮다 — SEED {sf} (15명 전원), SEED-V {min(sgap['SEED-V']):.2f}~{max(sgap['SEED-V']):.2f}, SEED-IV "
   f"{min(sgap['SEED-IV']):.2f}~{max(sgap['SEED-IV']):.2f}.  SEED 에서는 중심화를 안 한 것과 같거나 더 나빴고, SEED-V · SEED-IV "
   "에서는 그보다는 조금 낫다."])
fig("figs/fig_summary.png", "그림 3.  (a) 적응 없음 → 중심화 (영상 끝까지 캘리브레이션), 세 데이터셋.  (b–d) 중심화 위에 더한 이득: "
    "라벨 회전 · prototype 혼합 (둘 다 감정 라벨 사용) · 자극 시점 정렬 (정렬에 감정 라벨을 쓰지 않음).  패널은 어긋남 c 가 작은 "
    "데이터셋부터 (SEED → SEED-IV → SEED-V) — 어긋남이 클수록 자극 시점 정렬의 이득이 커진다.  막대는 95% 신뢰구간.", width=150)
P("6-2. 효과가 없던 것", H3)
ea = [pr(FL[f"eaRealNoCen_{t}__proto_clip"], NL["none__proto_clip"]) for t, _ in BUD]
T([["방법", "결과", "뜻"],
   ["유클리드 정렬 (EA, 모델에 넣기 전 입력을 정규화하는 표준 방법)", "SEED " + " / ".join(f"{d:+.3f} (p={p:.2f})" for d, p, *_ in ea)
    + f" — 유의하지 않음 (감지 한계 {NCFG['SEED'].mde:.3f} 아래)", "현실적 조건에서는 뺄 수 있다.  쓰려면 모델을 그 입력으로 다시 "
    "학습해야 한다"],
   ["분산까지 맞추는 정규화 (stratified normalization)", f"보정 후 유의한 것 {int((holm(ps16) < 0.05).sum())}/{len(ps16)}",
    "평균만으로 충분하다"],
   ["라벨 없는 다른 방법 6가지 (의사라벨 회전, 적응형 배치 정규화, 테스트 시점 템플릿 조정 등)", "모두 중심화 대비 ±0.005 안 "
    "(SEED)", "남은 오차는 새 사용자의 뇌파만으로는 못 고친다"],
   ["영상 후반에 가중치 (시간 가중)", "SEED 에서만 작게 보였고 SEED-V 에서 재현 실패", "이득으로 주장하지 않는다"]], [52, 64, 50])
P("6-3. 왜 그런가 — 남은 오차의 정체", H3)
T([["같은 사람의 다른 날 사이 감정 방향 일치도 (코사인, 1 = 완전 일치)", "학습한 사람", "처음 보는 사람 (검증)",
    "처음 보는 사람 (시험)"]] +
  [[d, f"{m(SG[d]['cw_train']):.3f}", f"{m(SG[d]['cw_val']):.3f}", f"{m(SG[d]['cw_test']):.3f}"] for d, _ in DS],
  [74, 30, 32, 30])
P("모델은 학습한 사람들의 날짜별 차이는 거의 맞춰 놓지만, <b>처음 보는 사람</b>에게서는 감정 방향이 날마다 돌아가 있다 "
  f"(특히 SEED-V {m(SG['SEED-V']['cw_test']):.2f}).  이 회전은 사람마다 제각각이어서, 학습 피험자들에게서 찾은 공통 회전 "
  "방향을 지우면 SEED-V 에서는 오히려 나빠진다 (−0.025).  그래서 새 사용자의 뇌파만으로 할 수 있는 최선은 중심화다.")
P("6-4. 감정 라벨을 쓰면", H3)
tc = {d: (m(TC[d]["acc__0-20초"]), m(TC[d]["acc__160-끝초"])) for d, _ in DS}
P("캘리브레이션 영상은 실험자가 고른 것이라 각 영상의 감정 (자극 라벨) 을 안다.  이것을 쓰면 SEED-V 에서 <b>볼수록 효과가 "
  f"커진다</b>: 감정당 20초에서는 효과가 없고 (라벨 회전 {cr['SEED-V']['T20'][0]:+.3f}, prototype 혼합 "
  f"{l3['SEED-V']['T20'][0]:+.3f}), 40초에서 {cr['SEED-V']['T40'][0]:+.3f} / {l3['SEED-V']['T40'][0]:+.3f}, 끝까지 보면 "
  f"{cr['SEED-V']['Tfull'][0]:+.3f} / {l3['SEED-V']['Tfull'][0]:+.3f}.  SEED 에서는 효과가 없다.  짧을 때 약한 이유는 영상 "
  "앞부분에 감정이 아직 뇌파에 덜 실려 있기 때문으로 보인다 — 영상 안 구간별 창 정확도가 SEED "
  f"{tc['SEED'][0]:.3f} (첫 20초) → {tc['SEED'][1]:.3f} (160초 이후), SEED-V {tc['SEED-V'][0]:.3f} → {tc['SEED-V'][1]:.3f} "
  "로 오른다.  가장 단순한 prototype 혼합이 회전보다 조금 높아서, 한계는 방법보다 라벨의 양 (감정당 영상 1편) 에 있는 것으로 "
  "보인다.")
P("6-5. 자극 시점 정렬 — 같은 영상의 같은 순간을 짝짓기", H3)
P("새 사용자가 본 캘리브레이션 영상은 학습 피험자들도 본 같은 영상이다.  그래서 4초마다 '같은 장면을 볼 때 학습 피험자들은 "
  "평균적으로 이런 반응이었다' 는 짝이 생긴다.  감정 라벨은 쓰지 않고 '어떤 영상의 몇 초' 라는 정보만 쓴다 — 그래서 뇌파만 "
  f"쓰는 방법의 천장을 넘을 수 있다.  이 짝 (감정당 20초면 {min(v['T20'] for v in PAIRS.values())}~"
  f"{max(v['T20'] for v in PAIRS.values())}개, 끝까지 보면 약 {min(v['Tfull'] for v in PAIRS.values())}~"
  f"{max(v['Tfull'] for v in PAIRS.values())}개) 으로 회전을 추정해 되돌린다 — 뇌영상 분야의 hyperalignment, SSVEP 의 템플릿 전이와 "
  "같은 원리를 감정 영상 캘리브레이션에 옮긴 것이다.")
rows = [["중심화 대비 향상"] + [b for _, b in BUD]]
for d, _ in DS3:
    rows.append([d] + [fmt(sla[d][t]) for t, _ in BUD])
T(rows, [40, 42, 42, 42])
B([f"<b>SEED-V: 감정 라벨 없이, 라벨을 쓰는 방법보다 높다</b> (끝까지 볼 때 prototype 혼합보다 {sl3[0]:+.3f}, "
   f"p={sl3[1]:.3f}).  감정당 20초만 봐도 작동한다 ({sla['SEED-V']['T20'][0]:+.3f}, p={sla['SEED-V']['T20'][1]:.3f}).  "
   f"둘을 합쳐도 더 오르지 않는다 ({both[0]:+.3f}).",
   f"<b>SEED-IV: 방향은 같지만 기준 미달.</b>  끝까지 볼 때 {fmtw(sla['SEED-IV']['Tfull'])} (p={sla['SEED-IV']['Tfull'][1]:.3f}) "
   f"이지만, 세 예산을 함께 보정하면 p={sla_holm4:.3f} 라 사전 등록한 '재현' 에 못 미쳤다.  라벨을 쓰는 prototype 혼합과 같은 크기다 "
   f"({sl3_4[0]:+.3f} 차이).",
   "<b>SEED: 효과가 없다.</b>  결과를 보기 전에 정한 기준 (두 데이터셋 모두에서 효과) 으로 '실패' 로 판정했다.",
   f"<b>왜 갈렸나</b>: 처음 보는 사람의 감정 방향이 학습 기준점과 얼마나 일치하는지 (1 = 완전 일치) 재면 SEED "
   f"{m(MIS['SEED']['c']):.2f}, SEED-IV {m(MIS['SEED-IV']['c']):.2f}, SEED-V {m(MIS['SEED-V']['c']):.2f} 이고, 이득도 이 순서를 따른다 "
   f"({sla['SEED']['Tfull'][0]:+.3f} → {sla['SEED-IV']['Tfull'][0]:+.3f} → {sla['SEED-V']['Tfull'][0]:+.3f}).  SEED-IV 의 이득은 결과 "
   f"전에 {pred4:+.3f} 로 예측했고 관측은 {sla['SEED-IV']['Tfull'][0]:+.3f} 였다 (방향은 맞고 크기는 약 2/3).  세 점이라 관계를 입증하지는 "
   "못하고, 같은 데이터셋 안에서는 사람별 어긋남이 이득을 거의 예측하지 못했다 (탐색)."])
P("6-6. 재현과 점검", H3)
cpf = pr(CP32["Tfull__proto_clip"], CP32["none__proto_clip"])
B(["<b>SEED-V 재현</b>: SEED 에서 얻은 결론 9개를 사전 등록하고 대보니 8개 재현, 1개 실패 (시간 가중 — 이득으로 주장하지 않음).",
   "<b>SEED-IV 재현</b>: 주 결과 (중심화) 를 사전 등록 기준 (세 예산 모두 향상 · p &lt; 0.05 · 15명 중 12명 이상) 으로 재현 — "
   + " / ".join(f"{cen['SEED-IV'][t][0]:+.3f}" for t, _ in BUD) + f", 모두 {min(cen['SEED-IV'][t][2] for t, _ in BUD)}/15.",
   f"<b>계산 정밀도 점검</b>: 특징을 더 정밀한 계산(fp32)으로 다시 뽑아도 결론이 같다 (중심화 효과 "
   f"{cen['SEED']['Tfull'][0]:+.3f} → {cpf[0]:+.3f}).",
   f"<b>참고 — 한 사람 안에서 학습·평가</b> (피험자 내 평가): SEED 클립 {sd_mean('results/a02_sd.csv', 'clip_accuracy'):.3f}, "
   f"SEED-V {sd_mean('results/seedv_a02_sd.csv', 'clip_accuracy'):.3f}.  평가 구성이 달라 위 표와 직접 비교하지는 않는다."])

# ── 7. 실용 지침 ──
P("7. 실용 지침 — 상황별 권장 구성", H2)
T([["상황", "권장", "근거 (클립 정확도)"],
   ["항상", "감정마다 영상 1편, 모든 감정 포함 → <b>중심화</b>",
    f"적응 없음 대비 SEED {min(x[0] for x in cen['SEED'].values()):+.3f}~{max(x[0] for x in cen['SEED'].values()):+.3f}, "
    f"SEED-V {min(x[0] for x in cen['SEED-V'].values()):+.3f}~{max(x[0] for x in cen['SEED-V'].values()):+.3f}, "
    f"SEED-IV {min(x[0] for x in cen['SEED-IV'].values()):+.3f}~{max(x[0] for x in cen['SEED-IV'].values()):+.3f}"],
   ["캘리브레이션 영상이 학습 피험자들이 본 영상이고, 새 사용자의 어긋남이 클 때", "+ <b>자극 시점 정렬</b>",
    f"중심화 대비 SEED-V {sla['SEED-V']['T20'][0]:+.3f} (20초) ~ {sla['SEED-V']['Tfull'][0]:+.3f} (끝까지), "
    f"SEED-IV {sla['SEED-IV']['Tfull'][0]:+.3f} (기준 미달), SEED {sla['SEED']['Tfull'][0]:+.3f}"],
   ["자극 시점 정렬을 쓸 수 없고, 감정당 40초 이상 볼 때", "+ prototype 혼합 (영상의 감정 라벨 사용)",
    f"중심화 대비 SEED-V {l3['SEED-V']['T40'][0]:+.3f} (40초) ~ {l3['SEED-V']['Tfull'][0]:+.3f} (끝까지), SEED-IV "
    f"{l3['SEED-IV']['Tfull'][0]:+.3f} (끝까지), SEED 효과 없음"],
   ["뺄 것", "유클리드 정렬, 분산 정규화, 라벨 없는 다른 적응, 자극 시점 정렬 위에 prototype 혼합", "더하는 것이 없다"]],
  [56, 54, 56])
P("주의: '어긋남이 클 때' 를 캘리브레이션 데이터만으로 판단하는 규칙은 아직 없다 — 사람별 어긋남으로는 예측이 안 됐다.  또 prototype 혼합의 효과는 학습 "
  "피험자들이 본 영상에서 잰 것이라, 학습에 없던 영상에서는 아직 검증되지 않았다.", SMALL)

# ── 8. 기여 · 한계 · 다음 ──
P("8. 이 연구의 기여와 한계", H2)
P("기여", H3)
B(["<b>정직한 회계</b> — 평가 데이터를 전혀 쓰지 않는 현실적 캘리브레이션과 시청 시간 예산.  기존 방식과의 차이를 숫자로 보였다.",
   "<b>무엇이 충분하고 왜 그런가</b> — 오차를 오프셋 · 분산 · 회전으로 나눠, 평균 빼기 하나가 라벨 없는 적응의 천장인 이유를 "
   "설명했다.",
   "<b>남은 것을 언제 어떻게 되찾나</b> — 회전을 추정하려면 대응 정보가 필요하다.  '어떤 영상의 몇 초' 를 쓰는 자극 시점 정렬은 "
   "어긋남이 큰 데이터셋에서 효과가 났지만 사전 등록 재현 기준은 통과하지 못했다.  원리는 뇌영상 hyperalignment · SSVEP 템플릿 "
   "전이와 같고, 자극 정보를 캘리브레이션 때만 쓰는 형태로 감정 FM 에 옮긴 것이 차이다 (CALIBRATION_SURVEY.pdf).",
   "<b>엄격성</b> — 사전 등록, 세 데이터셋 재현, 음성 결과까지 보고, 수치 자동 검증 (268개 항목)."])
P("한계", H3)
B(["방법 하나하나는 기존 기법이다 — 새 알고리즘이 아니라 '무엇을 언제 쓸지' 가 기여다.",
   "모델이 LaBraM 하나이고, 데이터셋이 모두 같은 실험실(상하이교통대) 것이다.",
   "<b>반복 자극 문제</b> — 학습과 평가에서 같은 영상을 보므로 모델이 감정이 아니라 영상을 알아볼 가능성이 있다 (분야 전체의 "
   "문제).  보지 못한 영상으로 평가해 확인할 계획이다.",
   f"사람 수가 15~16명이라, 이 표본으로 확실히 잡을 수 있는 최소 효과가 SEED 에서 약 {NCFG['SEED'].mde:.3f} 다.  그보다 작은 "
   "효과는 놓칠 수 있다.",
   "자극 시점 정렬은 두 번의 사전 등록 판정 (V5 · V6) 을 모두 통과하지 못했다 — 뚜렷한 효과는 SEED-V 에서만이고, 언제 켤지 정하는 "
   "규칙도 없다."])
P("다음 단계", H3)
B(["문헌이 요구하는 대조 실험 (CPU): 시점을 섞은 자극 시점 정렬, 남의 캘리브레이션으로 중심화, 변환 형태 비교, 감정 구성 불균형.",
   "보지 못한 영상으로 평가 (반복 자극 문제 확인) — GPU 재학습 필요.",
   "다른 FM (CBraMod · REVE) 과 FACED (123명이 같은 28클립) 로 일반화 — GPU 필요."])

# ── 9. 용어 풀이 ──
P("9. 용어 풀이", H2)
T([["용어", "뜻"],
   ["EEG (뇌파)", "머리에 붙인 전극으로 기록한 뇌의 전기 신호"],
   ["창 (window) · 클립 (clip)", "4초 단위로 자른 뇌파 조각 · 영상 한 편 전체 (클립 판정은 그 안의 모든 창을 모아 한 번)"],
   ["피험자 하나 빼기 교차검증 (LOSO)", "한 사람을 빼고 학습해 그 사람으로 시험하기를 모든 사람에 대해 반복하는 평가"],
   ["파운데이션 모델 (FM) · LaBraM", "대량의 뇌파로 미리 학습한 큰 모델 · 이 연구가 쓴 공개 모델"],
   ["캘리브레이션", "새 사용자가 쓰는 날 처음에 짧게 녹화해 모델을 그 사람에게 맞추는 과정"],
   ["전달식 (transductive)", "평가할 데이터를 미리 보고 정규화·적응에 쓰는 방식 — 실제 사용에서는 불가능"],
   ["라벨 없는 적응", "새 사용자의 뇌파만 보고 모델을 맞추는 방법 (감정 라벨도, 무슨 영상을 봤는지도 쓰지 않음)"],
   ["도메인 중심화", "캘리브레이션 녹화의 평균 특징을 모든 특징에서 빼기 (원점 맞춤).  가장 단순한 라벨 없는 적응"],
   ["유클리드 정렬 (EA)", "모델에 넣기 전의 뇌파를 채널별 진폭이 맞도록 정규화하는 기존 방법"],
   ["자극 시점 정렬 (SLA)", "같은 영상의 같은 순간에 대한 반응끼리 짝지어 회전을 추정하고 되돌리기 (방향 맞춤)"],
   ["prototype · prototype 혼합 (L3)", "감정마다의 기준점 · 그 기준점을 캘리브레이션 영상의 감정 평균 쪽으로 옮기기"],
   ["사전 등록", "결과를 보기 전에 판정 기준과 예측을 문서로 고정해 두는 것"],
   ["Holm 보정 · 최소 감지 효과 (MDE)", "여러 검정을 한꺼번에 할 때 우연한 '유의' 를 막는 보정 · 이 표본 크기로 잡을 수 있는 가장 작은 효과"]],
  [52, 114])

# ── 더 자세한 문서 ──
gap(4)
P("<b>더 자세한 문서 (reports/)</b> — METHOD.pdf: 방법의 정확한 정의 · 수식 · 학습 설정  ·  REVIEW.pdf: 선행연구 · 방법별 "
  "기여 평가 · 논문 방향  ·  CALIBRATION_SURVEY.pdf: 캘리브레이션 분야 지형 조사  ·  SUMMARY_SEED_SEEDV.pdf: 짧은 결과 요약  ·  V1~V6 (*_PREREG.md 등): 각 실험의 사전 등록과 판정  ·  "
  "NEXT.md, README.md: 작업 일지와 실행 명령.", SMALL)


def footer(c, d):
    c.saveState(); c.setFont(F, 7.5); c.setFillColor(GREY)
    c.drawString(22 * mm, 12 * mm, "짧은 캘리브레이션으로 새 사용자에게 맞추기 — 연구 총정리 (가제)")
    c.drawRightString(A4[0] - 22 * mm, 12 * mm, str(d.page))
    c.setStrokeColor(LINE); c.setLineWidth(0.4); c.line(22 * mm, 15 * mm, A4[0] - 22 * mm, 15 * mm); c.restoreState()


doc = BaseDocTemplate("reports/OVERVIEW.pdf", pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm,
                      bottomMargin=20 * mm, title="짧은 캘리브레이션으로 새 사용자에게 맞추기 — 연구 총정리")
doc.addPageTemplates([PageTemplate(id="a", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)],
                                   onPage=footer)])
doc.build(E)
print("[saved] reports/OVERVIEW.pdf")
