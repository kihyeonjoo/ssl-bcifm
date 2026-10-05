"""캘리브레이션 연구 지형 조사 — reports/CALIBRATION_SURVEY.pdf.

다섯 갈래 병렬 조사 (reports/survey_raw/A1~A5) 를 종합한다.  문헌 수치는 원 보고서에서 옮기고, 우리 수치는 results/ 에서 읽는다.
핵심 5편 (SCORE, GRN, mdJPT, Tao & Chen, EEG-Arena) 은 종합할 때 arXiv 원문으로 다시 확인했다.
PDF 는 약어를 쓰되 처음 나올 때 풀어 쓴다 (doc_terms 를 거치지 않는다).  끝에 폰트에 없는 글자가 있는지 검사한다.

    python make_survey_pdf.py
"""
from __future__ import annotations

import datetime
import os
import re
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
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
LINE, BG, NOTE, OURS = (colors.HexColor("#d4d9dd"), colors.HexColor("#f5f7f9"), colors.HexColor("#eef4ef"),
                        colors.HexColor("#fbeae4"))
LINKC = "#2c5f8a"


def st(name, size=9.3, lead=14.2, color=INK, font=F, space=4, left=0):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead, textColor=color, spaceAfter=space,
                          leftIndent=left, alignment=TA_LEFT)


BODY = st("b"); BUL = st("bul", left=10)
TITLE = st("t", 18, 25, INK, FB, 5); SUBT = st("st", 11.5, 16, INK, F, 6)
H2 = st("h2", 12.5, 17, ACC, FB, 6); H2.keepWithNext = 1
H3 = st("h3", 10.2, 14.5, INK, FB, 3); H3.keepWithNext = 1
H2S, H3S = ParagraphStyle("h2s", parent=H2), ParagraphStyle("h3s", parent=H3)   # 긴 표 앞 제목 (표와 묶지 않음)
H2S.keepWithNext = H3S.keepWithNext = 0
SMALL = st("s", 8.0, 11.6, GREY, space=3); REF = st("r", 7.7, 11.0, INK, space=1.5, left=8)
CELL = st("c", 7.7, 10.6, INK, space=0); CELLB = st("cb", 7.7, 10.6, INK, FB, space=0)
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


def _with_headings(group):
    while E and isinstance(E[-1], Paragraph) and E[-1].style in (H2, H3):
        group.insert(0, E.pop())
    E.append(KeepTogether(group))


def T(rows, widths, hl=(), split=False):
    data = [[_p(str(c), CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], hAlign="LEFT", repeatRows=1)
    s = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.8),
         ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8), ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
         ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
         ("BACKGROUND", (0, 0), (-1, 0), BG), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
         ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE),
         ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK)]
    for r in hl:
        s.append(("BACKGROUND", (0, r), (-1, r), OURS))
    t.setStyle(TableStyle(s))
    if split:                                   # 긴 표는 쪽을 넘겨도 되게: 남은 자리가 60 mm 미만일 때만 새 쪽
        heads = []
        while E and isinstance(E[-1], Paragraph) and E[-1].style in (H2, H3):
            heads.insert(0, E.pop())
        E.append(CondPageBreak(60 * mm))
        E.extend(Paragraph(h._src, H2S if h.style is H2 else H3S) for h in heads); E.append(t)
    else:
        _with_headings([t])
    gap(5)


def L(label, url):
    return f'<link href="{url.replace("&", "&amp;")}" color="{LINKC}">{label}</link>'


O = lambda t: f'<font color="#b2432b"><b>{t}</b></font>'          # 우리 방법 표시

# ── 우리 수치 (results/) ─────────────────────────────────────────────────
R = lambda f: np.load(f"results/{f}")
m = lambda a: float(np.asarray(a, float).mean())
NL, CPs, CPv = R("noea_ladder.npz"), R("calib_protocol_noea_r10.npz"), R("seedv_calib_protocol_noea_r10.npz")
SLs, SLv = R("sla_noea.npz"), R("seedv_sla_noea.npz")
ours = dict(
    s_win_none=m(NL["none__proto_win_bal"]), s_win_cen=m(NL["cen_Tfull__proto_win_bal"]),
    s_clip_none=m(CPs["none__proto_clip"]), s_clip_cen=m(CPs["Tfull__proto_clip"]), s_clip_tr=m(CPs["transductive__proto_clip"]),
    s_wacc_cen=m(CPs["Tfull__proto_win"]), s_wacc_tr=m(CPs["transductive__proto_win"]),
    v_win_none=m(CPv["none__proto_win_bal"]), v_win_cen=m(CPv["Tfull__proto_win_bal"]), v_win_tr=m(CPv["transductive__proto_win_bal"]),
    v_clip_none=m(CPv["none__proto_clip"]), v_clip_cen=m(CPv["Tfull__proto_clip"]), v_clip_tr=m(CPv["transductive__proto_clip"]),
    v_sla=m(SLv["Tfull__sla__clip"]) - m(SLv["Tfull__center__clip"]), s_sla=m(SLs["Tfull__sla__clip"]) - m(SLs["Tfull__center__clip"]))
gap_s, gap_v = ours["s_clip_tr"] - ours["s_clip_cen"], ours["v_clip_tr"] - ours["v_clip_cen"]
assert 0 < gap_s < 0.03 and 0 < gap_v < 0.03, (gap_s, gap_v)

# ════ 머리 ════════════════════════════════════════════════════════════════
P("캘리브레이션 연구 지형 조사", TITLE)
P("새 사용자에게 뇌파 모델을 맞추는 방법들 — 고전 BCI 부터 파운데이션 모델까지, 그리고 우리 연구의 위치", SUBT)
P(f"{datetime.date.today():%Y-%m-%d} · 다섯 갈래 병렬 조사 (갈래당 20~50편) · 갈래별 원 보고서: reports/survey_raw/ · "
  "약어는 처음 나올 때 풀어 쓴다", SMALL)
gap(4)
P("<b>한눈에.</b><br/>"
  "<b>① \"요즘은 캘리브레이션만 잘하면 된다\" — 절반만 맞다.</b>  사람이 바뀌는 것이 병목이라는 쪽은 강하게 지지된다: 파운데이션 "
  "모델(FM)이 내세우는 감정 수치는 대부분 같은 사람 안에서 나눈 평가이고, 사람을 분리하면 SEED 가 50% 안팎으로 떨어진다 "
  f"(우리 LaBraM 의 적응 없는 창 균형 정확도 {ours['s_win_none']:.3f} 도 문헌과 일치).  FM 임베딩은 피험자 정체성이 지배하고 미세조정이 "
  "그것을 키운다.  그러나 \"캘리브레이션만\" 은 아니다: 다른 사람들로 학습하는 양을 4배로 늘리면 개인 캘리브레이션 이득이 1~2%p 로 "
  "준다는 보고가 있고 (2026-09), 감정 전용 사전학습이 일반 FM 을 이긴다.  정확한 문장은 <b>\"강한 모집단 표현 + 사람 차이만 겨냥한 "
  "가벼운 캘리브레이션\"</b> 이다.<br/>"
  "<b>② 분야를 가로지르는 합의 세 가지.</b>  (a) 평균 맞추기(재중심화)가 이득의 대부분이다.  (b) 회전은 '대응' — 라벨, 짝지은 "
  "데이터, 공유 자극의 시점 — 이 있어야 추정된다.  (c) 무거운 적응(경사 기반 테스트 시점 적응)은 불안정하다.  우리 결과 셋 "
  "(중심화가 대부분, 남는 것은 회전, 라벨 없는 적응은 ≈0) 이 이 합의와 정확히 겹친다.<br/>"
  "<b>③ 빈자리는 실재한다.</b>  'FM + 감정 + 평가 데이터 미사용 + 짧은 캘리브레이션 블록' 을 평가한 연구는 세 갈래 조사가 "
  "독립적으로 찾지 못했다 — 우리 연구의 자리다.<br/>"
  "<b>④ 새로움 주장은 좁혀야 한다.</b>  자극 시점 정렬(SLA)의 원리는 fMRI 하이퍼정렬, SSVEP 템플릿 전이(10년 계보)와 같고, 2026년에 "
  "뇌파 임베딩 직교 정렬(SCORE)과 감정 인식의 같은 자극 참조(GRN)가 나왔다.  지킬 수 있는 것은 '자극 정보를 캘리브레이션 때만 쓰고, "
  "시험 때는 아무 정보도 필요 없는, 고정 FM 임베딩의 회전' 과 '언제 필요한지' 다 (4·5절).<br/>"
  "<b>⑤ 바로 할 대조 실험 (CPU).</b>  시간 섞기 자극 시점 정렬 · 남의 캘리브레이션으로 중심화 · 사상 형태 비교 · 클래스 불균형.  "
  f"오라클 상한은 이미 있다: 테스트 세션 전체로 중심화해도 클립 정확도가 SEED {gap_s:+.3f}, SEED-V {gap_v:+.3f} 뿐 더 높다.", BOX)

# ── 1. 조사 방법 ──
P("1. 조사 방법과 읽는 법", H2)
T([["갈래", "범위", "핵심 질문"],
   ["A1 고전 BCI", "운동 상상 · ERP · SSVEP/c-VEP 의 캘리브레이션 단축 (2015–2026)", "정렬·템플릿 전이는 새 사용자 데이터를 얼마나, 어떻게 쓰나"],
   ["A2 적응", "테스트 시점 적응(TTA) · 소스 없는 적응(SFDA) · 온라인 비지도 적응", "평가 데이터를 미리 쓰지 않으면 이득이 남는가"],
   ["A3 감정", "교차 피험자·세션 감정 인식의 캘리브레이션 · 적은 데이터 적응 · 평가 함정", "테스트와 분리된 캘리브레이션을 쓰는 연구는 무엇인가"],
   ["A4 FM", "뇌파 FM 과 벤치마크, 새 사용자 적응", "FM 시대에 캘리브레이션이 병목인가"],
   ["A5 정렬", "같은 자극을 이용한 사람 간 정렬 (fMRI · MEG · ECoG · EEG)", "자극 시점 정렬은 새로운가"]], [24, 78, 64])
P("모든 논문은 제목·저자·연도·출처를 arXiv · Crossref · 출판사 페이지로 확인한 것만 실었다.  조사 중 세션 공용 웹 검색 한도(200회)가 "
  "소진돼 뒤쪽 검증은 arXiv API 와 Crossref 로 했다 — 2026 하반기 저널 논문이 빠졌을 수 있다 (10절).  새 사용자 데이터를 어떻게 쓰는지가 "
  "이 분야를 가르는 가장 중요한 축이라, 아래 표기를 문서 전체에 쓴다.", BODY)
T([["표기", "새 사용자에게서 쓰는 것", "실사용 가능?"],
   ["없음", "아무것도 (도메인 일반화, 학습 단계에서만 정렬)", "가능"],
   ["블록", "테스트와 분리된 짧은 캘리브레이션 블록, 라벨 없음", "가능"],
   ["블록+", "캘리브레이션 블록 + 자극 정보 (어떤 자극의 몇 초인지) 또는 소량 라벨", "가능 (보정 절차 필요)"],
   ["온라인", "평가 데이터 스트림을 도착 순서대로 (라벨 없음)", "가능 (스트림이 길어야 효과)"],
   ["전달식", "평가 데이터 전체를 미리 (정규화·적응에)", "불가능 — 성능이 부풀려진다"]], [20, 104, 42])

# ── 2. 지형 행렬 ──
P("2. 지형 행렬 — 어디를 고치고, 새 사용자의 무엇을 쓰나", H2)
T([["고치는 곳", "없음", "블록 (라벨 없음)", "블록+ (자극 정보·소량 라벨)", "온라인 · 전달식"],
   ["입력 신호·공분산", "tt-CCA (남의 SSVEP 템플릿 그대로)",
    "유클리드 정렬(EA) · 리만 재중심화를 블록으로 추정, 의사-온라인 EA (Junqueira 2024)",
    "SSVEP: LST · sd-LST · stCCA · SSVEP-DAN · OS-SSVEP.  리만 Procrustes(RPA) 회전 · 라벨 정렬(LA)",
    "온라인 EA (Wimpff 2024, T-TIME), OACCA / 오프라인 EA · SPDIM"],
   ["특징·임베딩 정규화", "학습 통계만 쓰는 정규화 (Zhou 2026)",
    O("도메인 중심화") + ", PPDA (개인 인코더, 45초)", "개인 어댑터의 가산 오프셋 (Tao &amp; Chen 2026, 라벨)",
    "온라인 AdaBN, CLISA 적응형 정규화 / 계층 정규화 · Personal-Zscore · 잠재 정렬 · SATTC · z-점수 (Apicella 2023)"],
   ["특징·임베딩 회전·사상", "CLISA · CL-SSTER · TA2CL · mdJPT (학습 단계에서 같은 자극 정렬, 새 사용자는 그대로)", "—",
    O("자극 시점 정렬(SLA)") + ", " + O("라벨 회전") + ", 하이퍼정렬 · SRM (fMRI · ECoG), STM (Li 2020)",
    "SCORE (직교, 라벨 없는 배치) / GRN (같은 자극 참조, 시험 자극 정보 필요)"],
   ["분류기·prototype", "—", "—", O("prototype 혼합") + ", REVE 클래스 평균 분류기, ProtoNet · EvoFA",
    "T3A · LAME / PR-PL · SHOT"],
   ["모델 파라미터", "도메인 일반화 (DResNet, MAT)", "—",
    "같은 사람 안 미세조정 (Compass), 개인 LoRA · 어댑터, 메타학습 (MAML 계열)",
    "Tent · T-TIME · NeuroTTT / 대부분의 도메인 적응 (MS-MDA, DANN), SFDA"]],
  [24, 30, 34, 42, 36])
P(f"빨간 굵은 글씨가 우리 방법이다.  표의 오른쪽 열로 갈수록 실사용과 멀어진다.  우리 세 방법은 모두 '블록' · '블록+' 열에 있다.", SMALL)

# ── 3. 갈래별 ──
P("3. 갈래별 핵심", H2)
P("3-1. 고전 BCI — 재중심화는 표준, 자극 동기 템플릿 전이는 10년 된 주류", H3)
B(["<b>재중심화 합의.</b>  LDA 평균의 비지도 추적 (Vidaurre 2011) → 리만 재중심화 (Zanini 2018) → EA (He &amp; Wu 2020) → SPD 도메인별 "
   "배치 정규화 (Kobler 2022) → <b>Mellot 2023: \"새 피험자에는 재중심화가 가장 폭넓게 유효하고, 회전은 짝지은 데이터가 있을 때만\"</b> — "
   "우리 결론 (중심화가 대부분 + 남는 회전) 과 거의 같다.  Lopes 2026 은 \"EA 의 이득은 곧 재중심화\" 라고 정리한다.",
   "<b>SSVEP 템플릿 전이.</b>  tt-CCA (2015, 남의 평균 템플릿을 그대로) → LST (2019/2021, 새 사용자의 몇 회 자극 반응에 맞춰 "
   "원천 시행을 선형 사상) → stCCA (2020, 9시행) → sd-LST (2023, 모든 자극에 공통인 사상 하나) → SSVEP-DAN · OS-SSVEP (2024, 비선형 · "
   "자극당 1회).  '새 사용자의 자극 동기 반응을 다른 사람의 같은 자극 템플릿과 짝지어 변환을 추정' 하는 발상이 우리 SLA 와 같다.  "
   "쟁점은 음의 전이와 원천 피험자 선택 (SSVEP-DAN 이 건식 전극 원천에서 LST 의 음의 전이를 관찰).",
   "<b>직교 프로크루스테스도 이미 있다.</b>  ALPHA (2022) 가 SSVEP 공간 패턴을 직교 회전으로 정렬 (같은 사용자의 장치 간).  RPA (2019) · "
   "접공간 정렬 (2022) 은 클래스 평균으로 회전을 추정한다.",
   "<b>짧은 보정의 현재 수준.</b>  SSVEP 40클래스를 9~24초, c-VEP 를 1분 미만으로 보정한다.  c-VEP FM (Behboodi 2026) 은 무보정 71.8% "
   "→ 약 43초 보정 92% (완전 보정 93.7%) — FM 위에서도 짧은 보정이 큰 몫을 한다."])
T([["논문", "새 사용자 데이터", "방법", "핵심 결과", "우리와의 관계"],
   ["Mellot 2023 (Imaging Neurosci.)", "블록 + 짝 데이터", "공분산 재중심·재스케일·회전 단계별 비교",
    "새 피험자는 재중심화가 핵심, 회전은 짝 데이터가 있어야", "중심화 + 회전 구조의 직접 선례"],
   ["LST, Chiang 2021 (JNE)", "블록+ (자극당 2–5회)", "원천 시행 → 새 사용자 템플릿, 제약 없는 선형", "자극당 2회로 약 78→90% (습식)",
    "SLA 의 SSVEP 판 (방향·수준·제약이 다름)"],
   ["sd-LST, Bian 2023 (TNSRE)", "블록+ (약 10시행)", "모든 자극 공통의 사상 하나", "40타깃을 약 10시행으로", "공통 변환 하나 = SLA 와 같은 설계"],
   ["ALPHA, Liu 2022 (TBME)", "같은 사용자 다른 장치", "직교 프로크루스테스 + CORAL", "습식→건식에서 완전 보정 TRCA 이상",
    "직교 정렬의 SSVEP 선례"],
   ["RPA, Rodrigues 2019 (TBME)", "블록 + 라벨 소수", "재중심 · 스케일 · 클래스 평균 회전", "8개 데이터셋 243명", "라벨 회전(CR)의 선례"],
   ["LA, He &amp; Wu 2020 (TNSRE)", "블록+ (클래스당 1개)", "클래스 조건부 평균 정렬", "클래스당 라벨 1개로 동작", "prototype 혼합의 선례"]],
  [30, 26, 38, 38, 34])

P("3-2. 테스트 시점 적응 — \"정규화가 먼저, 목적함수는 덤\"", H3)
B(["<b>정렬이 이득 대부분.</b>  Wimpff 2024 (운동 상상 교차 피험자): 원본 57.4 → EA 62.5 → 리만 정렬+BN 교체 65.4, 그 위 엔트로피 최소화는 "
   "레이블 스무딩 없이는 65.3 (≈0).  T-TIME (2024): EA 가 가장 큰 이득, <b>T3A 는 −5~−9점</b>, 오프라인 EA 는 증분 EA 보다 0.2~0.8점만 높다.  "
   "Bakas 2025: EA · AdaBN · 잠재 정렬의 이득 크기가 비슷해 서로를 대체한다.  Junqueira 2024: EA 위에 24시행 라벨로 미세조정해도 평균 이득 없음.",
   "<b>단서.</b>  비라벨 스트림이 길면 (100~200시행, 이진 운동 상상) 정보 최대화 계열이 +3~6점을 낸다 → 우리 '≈0' 은 <b>짧고 클래스 균형인 "
   "블록</b> 조건으로 한정해야 한다.  재중심화는 클래스 비율이 바뀌면 실패한다 (SPDIM, ICLR 2025; OSPDIM 2026) — 감정당 1클립 블록은 "
   "설계상 균형이라 이 위험을 피한다.  보정 블록이 테스트 조건을 대표하지 못하면 블록 기반 적응이 원본보다 나빠질 수 있다 (RAP 2025).",
   "<b>FM 의 테스트 시점 적응은 불안정.</b>  NeuroAdapt-Bench (MLHC 2026, CBraMod · REVE): Tent · SHOT 이 크게 악화, 평균이 양수인 것은 T3A 뿐 "
   "(감정 · LaBraM 없음).  NeuroTTT 는 LaBraM 에서 운동 상상 0.411→0.451.  NeuroOnline 의 SEED-V 큰 이득 (41→53) 은 라벨을 쓰고 같은 사람 안 "
   "분할이라 우리와 비교할 수 없다.",
   "<b>일반 기계학습의 실패 보고</b> (LAME · NOTE · RDumb · TTAB): 시간 상관 · 불균형 스트림에서 파라미터 갱신 적응이 원본보다 나빠진다.  SEED "
   "류는 클립 하나가 몇 분간 한 감정이라 정확히 그 조건이다 — 스트림 적응보다 분리된 블록이 나은 이유가 된다.",
   "<b>감정 인식</b>: Apicella 2023 — SEED 에서 정규화만으로 도메인 적응과 같거나 낫다 (81.5%, 단 전달식).  감정 SFDA 는 대부분 전달식 "
   "교차 데이터셋 (Imtiaz &amp; Khan, TAFFC 2026)."])

P("3-3. 감정 인식 — 전달식이 주류, 분리된 캘리브레이션은 소수", H3)
B(["<b>주류는 평가 데이터를 쓰는 도메인 적응</b>: TCA/TPT (2016) → DAN (2018) → MS-MDA (2021) → PR-PL (TAFFC 2024).  2026 리뷰는 '진정한 "
   "플러그 앤 플레이' 를 미래 과제로 꼽는다.",
   "<b>테스트와 분리된 보정을 쓰는 연구</b>: PPDA (AAAI 2021) — 세션 앞 45초 무라벨, SEED 0.854 → 0.867, 15~95초 사이 평탄.  SEED 는 감정 순서가 "
   "고정이라 이 45초는 첫 클립 하나 (한 감정) 다 → 우리 '감정 포괄이 길이보다 중요' 결과로 다시 읽을 수 있다 (우리 추론).  <b>제1저자가 LaBraM "
   "공저자</b>라 반드시 인용·비교.  Li 2020 (TCYB, 보정 세션 소량 라벨 + 스타일 전이 사상, +12.7%p), Lin 2017 · 2020, EvoFA 2024 (세션 1 의 "
   "1·5샷, 다음 세션 시험).",
   "<b>퓨샷 수치는 부풀려져 있다</b>: FACE (2025) SEED-V 10샷 98.95% — 같은 시행 안 무작위 분할이라 인접 창이 학습과 시험에 섞인다.  "
   "시행 내 누출로 +25~36%p (Lei 2025), 체크포인트를 시험 풀에서 고르면 +6.2%p (Suo 2026), 216편 중 19% 는 분할이 불명확 (Kukhilava 2025).",
   "<b>공유 자극은 학습 단계에만</b>: CLISA (TAFFC 2022) · CL-SSTER (2024) · TA2CL (2026) · mdJPT (NeurIPS 2025).  CLISA 는 미지 자극 시험에서 "
   "SEED 86.4 → 77.4 — 반복 자극 문제의 크기를 보여 주는 선례다.  TA2CL 은 사람마다 반응이 몇 초씩 어긋난다고 지적한다 (SLA 의 '같은 초' 가정).",
   "<b>주의</b>: 반복 자극 혼입 (Kilgallen 2025) 은 사물 범주 데이터 연구라 감정 데이터 결과로 인용하면 부정확하다 — 'SEED 의 클립 반복 구조' 와 "
   "연결하는 식으로 인용한다."])

P("3-4. 파운데이션 모델 — 사람 차이가 병목, 그러나 무거운 개인화는 해법이 아니다", H3)
B(["<b>대표 수치 대부분이 같은 사람 안 분할</b>: LaBraM · CBraMod · CSBrain · CodeBrain · NeuroLM · EEG-FM-Bench 의 SEED/SEED-V.  사람을 분리한 "
   "감정 표준은 FACED 분할 (CBraMod 0.551 → CodeBrain 0.594).",
   f"<b>사람을 분리하면 SEED 50% 안팎</b>: Compass LOSO — CBraMod 53.6, LaBraM 52.2, ShallowConv 53.4.  AdaBrain — LaBraM 55.8 (같은 피험자 "
   f"다른 시행이면 70.9).  우리 LaBraM 적응 없음 {ours['s_win_none']:.3f} (창 균형 정확도) → 중심화 {ours['s_win_cen']:.3f}.  EEG-Arena (2026-09, "
   "FM 30개) 는 같은 사람 안 69~94 대 분리 48~58 을 보고한다 (본문 수치, 직접 재확인 못 함).",
   "<b>피험자 정체성이 지배</b> (Identity Trap 2026): 피험자 분산이 무작위 기준의 13~89배, 미세조정하면 더 커지고, 피험자 축을 지우면 판별이 "
   "+6~12%p.  피험자마다 대비 방향이 다르다 — 우리 '중심화 + 남는 회전' 과 맞물린다.  SCORE 도 \"사람들은 같은 관계를 다른 좌표 방향으로 표현한다\" 고 쓴다.",
   "<b>무거운 개인화는 답이 아니다</b>: Compass 의 클래스당 영상 1개 같은 사람 안 미세조정에서 LaBraM 47.0 — LOSO 52.2 보다 낮다.  "
   "\"최소 보정 또는 무보정 적응이 미해결 핵심 과제\" (Compass, National Science Review 2026).",
   "<b>반론</b>: Tao &amp; Chen (2026-09-28, CBraMod · REVE · LaBraM, 운동 상상 235명) — 개인 어댑터 +1.5~5.4%p 가 모집단 학습량 4배에서 "
   "+1.0~2.0%p 로 준다; 소량 라벨 보정은 불안정.  <b>'남의 어댑터' 와 '모집단 규모' 대조를 요구한다.</b>  mdJPT — 감정 전용 다중 데이터셋 "
   "사전학습이 일반 FM 을 이긴다 (표현 품질도 중요)."])
T([["논문", "평가", "핵심 결과", "우리에게 주는 것"],
   ["EEG-FM-Compass (NSR 2026)", "LOSO 무보정 + 클래스당 영상 1개 미세조정", "SEED LaBraM 52.2 / 47.0; 특화 모델이 경쟁적", "같은 보정량의 무거운 기준선"],
   ["AdaBrain-Bench (2025)", "피험자 분리 / 다중 피험자 / 소수샷", "피험자 간 변동이 세션 간보다 어렵다", "가설 지지"],
   ["The Identity Trap (2026)", "LaBraM · CBraMod · REVE, 학습 fold 만으로 진단", "피험자 분산 13~89배, 축 제거 +6~12%p", "진단 지표 차용"],
   ["Tao &amp; Chen (2026-09)", "운동 상상 235명, 개인 vs 남의 어댑터", "모집단 4배면 개인 이득 1~2%p", "교환·규모 대조 필요"],
   ["mdJPT (NeurIPS 2025)", "데이터셋 단위 제외, 소수 피험자로 분류기", "같은 자극·시각 정렬 손실; 시각 맞춤 65.0 vs 비정렬 55.5", "SLA 의 사전학습 판"],
   ["NeuroAdapt-Bench (MLHC 2026)", "온라인·전달식 TTA (감정 없음)", "Tent · SHOT 악화, T3A 만 양수", "'라벨 없는 적응 ≈0' 의 맥락"],
   ["REVE (NeurIPS 2025)", "FACED 분리 + 운동 상상 소수샷", "동결 임베딩 + 클래스 평균 분류기", "prototype 접근의 FM 선례"]],
  [34, 44, 50, 38])

P("3-5. 같은 자극을 이용한 사람 간 정렬 — SLA 의 계보", H3)
B(["<b>원전</b>: 하이퍼정렬 (Haxby 2011, 영화 시점별 반응 + 직교 프로크루스테스), 공유 반응 모델 (SRM, Chen 2015, 피험자별 직교 사상 + k차원 "
   "공유 반응), 적은 데이터 새 피험자 (Jiahui 2020), ProMises (2023, 단위 행렬 쪽 수축의 원리적 판).",
   "<b>다른 모달리티의 새 피험자 정렬</b>: MEG M-CCA (2017), ECoG SRM 으로 새 참가자를 회전 (Nat. Comput. Sci. 2026), fMRI 무성영화 70분으로 "
   "새 참가자 정렬 후 디코더 이전 (Tang &amp; Huth 2025), 최적수송 정렬 (Thual 2023).",
   "<b>뇌파 임베딩의 직교 정렬 — SCORE (2026-08)</b>: 고정 인코더에서 라벨 없는 대상 뇌파로 직교 변환을 추정 (뇌파→이미지 검색).  분석 "
   "실험에서 직교 28.2% &gt; 릿지 20.0% &gt; 무정렬 16.9% — 수백 쌍 규모에서는 직교가 낫다.",
   "<b>감정 — GRN (ICANN 2026)</b>: SEED LOSO 87.9%.  \"공명은 같은 자극 시간축에 맞춘 샘플 사이에서만 계산\" — 시험 구간마다 자극·시점을 "
   "알아야 한다 (SEED 에서는 사실상 라벨).  변환은 없다.  자극을 어긋나게 맞추면 84.8% — 우리 시간 섞기 대조의 본보기.",
   "<b>감정 반응의 피험자 간 상관은 작다</b>: 0.03~0.06 (CL-SSTER), 피크가 각성 장면에 몰림 (Dmochowski 2012), valence 가 높을수록 낮음 "
   "(Hajlaoui 2018) — SLA 효과가 데이터셋마다 다른 이유의 실마리."])

# ── 4. 주장별 판정 ──
P("4. 우리 연구의 위치 — 주장별 새로움 판정", H2)
T([["우리 주장", "이미 알려진 것", "지킬 수 있는 것", "권장 표현"],
   ["도메인 중심화가 이득 대부분", "재중심화 원리 (Vidaurre · Zanini · EA · Mellot), 감정 정규화 (계층 정규화 · Personal-Zscore · Apicella — 대부분 "
    "전달식), 분리 블록 (PPDA)", "FM 임베딩에서, 평가 데이터 없이, 클립 분리 · 초 단위 예산으로 이득을 정량화; 감정 구성이 길이보다 중요",
    "잘 알려진 재중심화를 FM 감정 임베딩에서 현실적 프로토콜로 재확인·정량화"],
   ["남는 오차는 회전", "RPA 의 이동 모델 (평행이동 + 스케일 + 회전), Mellot, Identity Trap, SCORE ('다른 좌표 방향')",
    "FM 감정 임베딩에서의 진단 (학습 0.97 · 검증 0.85 · 시험 0.83 / SEED-V 0.86 · 0.44 · 0.41) 과 '라벨 없는 적응의 천장' 설명",
    "RPA 계열 이동 모델이 FM 임베딩에서도 성립; 회전에 어떤 대응이 필요한지 비교"],
   ["자극 시점 정렬(SLA)", "하이퍼정렬 · SRM (원리 동일), SSVEP LST 계열, SCORE (뇌파 임베딩 직교), GRN (감정 · 같은 자극), mdJPT · CLISA (학습 단계)",
    "자극 정보를 캘리브레이션 때만 사용, 시험 데이터·시험 자극 정보 불필요; 고정 FM + 고정 분류기; 위상 고정 없는 자연 영화의 4초 시점 대응; "
    "효과의 조건부성과 그 예측 (V6)",
    "hyperalignment 를 FM 배치용 캘리브레이션으로 — known-stimulus calibration"],
   ["'감정 라벨 없이'", "SEED 류에서는 클립 = 감정.  SSVEP 문헌 기준으로 자극당 보정 시행은 라벨 있는 보정",
    "정렬 목적함수에 클래스 통계를 쓰지 않음", "'정렬에 감정 라벨을 쓰지 않는다; 단 보정 클립의 정체는 클래스를 내포' 로 한정"],
   ["prototype 혼합", "스타일 전이 사상 (2013), Li 2020 (감정), LA, ProtoNet, REVE 클래스 평균, T3A",
    "클립 수준의 약한 라벨, 예산 곡선, 조건부 결과 (SEED-V 40초 이상)", "퓨샷 표준 기법의 현실적 평가로 서술"],
   ["라벨 없는 적응 ≈0", "지지: Wimpff · T-TIME · Bakas · Junqueira · NeuroAdapt-Bench · LAME/NOTE.  단서: 긴 스트림의 정보 최대화 +3~6, "
    "NeuroAdapt-Bench 의 T3A 양수, 불균형 시 SPDIM",
    "짧고 클래스 균형인, 테스트와 분리된 블록 조건에서의 결론", "조건을 명시해 주장 (긴 온라인 스트림까지 일반화하지 않음)"],
   ["현실적 평가 프로토콜", "분리 보정: PPDA · Li 2020 · Lin · EvoFA; Compass 의 클래스당 영상 1개",
    "클립 분리 + 감정 포괄 + 초 단위 예산을 함께 통제한 체계적 평가 (없음 확인), FM + 감정", "'처음' 대신 '함께 통제한 체계적 평가'"]],
  [26, 50, 50, 40], hl=(3,), split=True)

# ── 5. SLA 정밀 비교 ──
P("5. 자극 시점 정렬과 가장 가까운 방법들", H2)
T([["방법", "분야", "정렬 대상", "사상", "대응 단위", "새 사용자 데이터", "시험 때 필요"],
   ["하이퍼정렬 2011 · SRM 2015", "fMRI", "복셀 반응", "직교 (SRM: k차원)", "영화 시점", "공유 영화 시청", "없음"],
   ["tt-CCA 2015", "SSVEP", "원신호 템플릿", "없음 (그대로)", "자극", "없음", "없음"],
   ["LST 2021 · sd-LST 2023", "SSVEP", "원신호 (원천→대상)", "제약 없는 선형", "표본 시점 (위상 고정)", "자극당 2–5회 / 약 10회", "없음"],
   ["ALPHA 2022", "SSVEP", "공간 패턴", "직교 + CORAL", "자극", "같은 사용자 다른 장치", "없음"],
   ["RPA 2019 · TSA 2022", "운동 상상 등", "공분산 · 접공간", "직교 회전", "클래스 평균", "라벨 소수", "없음"],
   ["CLISA 2022 · mdJPT 2025", "감정 EEG", "딥 표현 (학습 단계)", "대조학습", "같은 자극·시각", "없음 (적응 안 함)", "없음"],
   ["GRN 2026", "감정 EEG", "위상 동기 특징", "변환 없음", "같은 자극 시간축", "—", "<b>시험 구간의 자극·시점</b>"],
   ["SCORE 2026", "뇌파→이미지", "고정 인코더 임베딩", "직교 + 항등 정규화", "이미지 랜드마크", "배치 시점 대상 뇌파 (라벨 없음)", "이미지 갤러리"],
   [O("SLA (우리)"), "감정 EEG", "FM 분류 토큰 (상위 k 부분공간)", "직교 + 단위 행렬 쪽 수축", "클립 × 초 (4초 창)",
    "감정당 1클립 20초~전체", "<b>없음</b>"]],
  [27, 15, 26, 22, 22, 30, 24], hl=(9,))
P("SLA 를 개선할 아이디어 (선행에서): ProMises 식 수축 R = polar(X<super>T</super>T + κI) 로 직교성을 유지한 채 보정 길이에 자동 적응 · 피험자 간 상관이 높은 "
  "초에 가중 · ±1~2초 지연 허용 (TA2CL) · 템플릿을 SRM/일반화 프로크루스테스로 정제 · 시점 앵커와 클래스 평균 앵커 혼합.  모두 SEED-V 결과를 "
  "본 뒤의 변경이라, 쓰려면 SEED-IV 나 FACED 처럼 새 데이터에서 미리 고정해 검증해야 한다.", SMALL)

# ── 6. 수치 읽는 법 ──
P("6. 문헌 수치를 읽는 법 — 평가 방식이 숫자를 정한다", H2)
T([["평가 방식", "예", "SEED 수준", "우리와 비교?"],
   ["같은 사람 안 (시행 분할)", "EEG-FM-Bench (EEGPT 70.8, LaBraM 61.6), NeuroLM 표의 LaBraM 73.2, EEG-Arena REVE 89.3", "60~94", "불가"],
   ["같은 시행 안 무작위 퓨샷", "FACE: SEED 3샷 91.7, SEED-V 10샷 98.95", "90+", "불가 (누출)"],
   ["사람 분리 + 전달식 도메인 적응", "MS-MDA 89.6, PR-PL 93.1 (차분 엔트로피 특징, 평가 데이터 사용)", "85~93", "불가"],
   ["사람 분리, 적응 없음", f"Compass LaBraM 52.2 · CBraMod 53.6, AdaBrain LaBraM 55.8, 우리 {100 * ours['s_win_none']:.1f} (창 균형 정확도)",
    "48~58", "가능 — 일치"],
   ["사람 분리 + 분리 블록 캘리브레이션 (우리)", f"창 {100 * ours['s_win_cen']:.1f}, 클립 {100 * ours['s_clip_cen']:.1f} (중심화, 영상 끝까지)",
    "—", "기준"],
   ["(상한) 테스트 세션 전체로 중심화", f"클립 {100 * ours['s_clip_tr']:.1f} (SEED), {100 * ours['v_clip_tr']:.1f} (SEED-V)", "—",
    f"현실적 대비 +{100 * gap_s:.1f} / +{100 * gap_v:.1f}%p 뿐"]],
  [38, 74, 20, 34], hl=(5,))
P("논문에서는 클립 정확도와 함께 창 단위 균형 정확도를 병기해야 벤치마크와 비교된다.  FM 이 과제 특화 모델보다 나은지는 문헌이 엇갈린다 "
  "(Compass: 특화 모델이 경쟁적, Lee 2025: 이득 0.9~1.2%p, EEG-Arena 초록: FM 이 우세하나 파라미터를 키운 이득은 없음) — 우리 논지에는 영향이 작다.", SMALL)

# ── 7. 대조 실험 ──
P("7. 리뷰어 대비 — 문헌이 요구하는 대조 실험 (우선순위)", H2)
T([["#", "실험", "막는 질문", "근거", "자원"],
   ["1", "시간 섞기 SLA: 같은 클립 안 초 섞기 · 같은 감정의 다른 클립 · 무작위 짝", "이득이 '시점 고정' 에서 오나, '영상 = 감정' 정보에서 오나",
    "GRN 소거, mdJPT E.2, A1·A5", "CPU"],
   ["2", "교환 캘리브레이션: 다른 사람·다른 날의 캘리브레이션 평균으로 중심화", "개인 정보인가, 아무 평균이나 빼도 되나", "Tao &amp; Chen 2026", "CPU"],
   ["3", "사상 형태: 직교 / 최소제곱 (LST 식) / 클래스 평균 회전 (RPA 식) / ProMises 수축", "왜 직교인가", "SCORE, LST, RPA, ProMises", "CPU"],
   ["4", "클래스 불균형: 감정 하나 빼기, 감정별 길이 불균형", "중심화의 클래스 균형 가정", "SPDIM, OSPDIM, Bakas", "CPU"],
   ["5", "스트림 대조: 지수이동평균 중심화 · T3A · LAME 를 시간순 스트림에서", "온라인 적응이면 더 낫지 않나", "T-TIME, NOTE, LAME", "CPU"],
   ["6", "보고 형식: 창 균형 정확도 병기, 체크포인트·하이퍼파라미터를 시험 피험자로 고르지 않았음 명시, 클립 순서별 정확도 (SEED 감정 순서 고정)",
    "비교 가능성 · 누출", "EEG-FM-Bench, Suo 2026", "CPU"],
   ["7", "미지 영상 평가 (시험 영상을 학습에서 제외)", "반복 자극 — 영상을 알아본 것 아닌가", "CLISA 미지 자극, Kilgallen", "GPU 재학습"],
   ["8", "다른 FM (CBraMod · REVE) + FACED (123명, 같은 28클립 — SLA 에 이상적)", "LaBraM · SEED 계열에만 맞는 것 아닌가", "Compass, CBraMod 분할",
    "GPU"],
   ["9", "모집단 규모 변화 (학습 피험자 수)", "보정 이득이 약한 모집단 모델의 착시인가", "Tao &amp; Chen 2026", "GPU"],
   ["10", "양 끝 기준선: 같은 보정량으로 같은 사람 안 미세조정, PPDA 재현", "무거운 개인화 · 가장 가까운 선행과 비교", "Compass, PPDA", "GPU (소)"],
   ["✓", f"오라클 상한: 테스트 세션 전체로 중심화 — 이미 있음 (클립 +{gap_s:.3f} / +{gap_v:.3f})", "캘리브레이션만 써서 얼마를 놓치나",
    "T-TIME", "완료"]],
  [7, 62, 46, 32, 19], hl=(1, 2), split=True)

# ── 8. 함의 ──
P("8. 논문 방향에 주는 함의", H2)
B(["<b>자리는 확실하다.</b>  세 갈래가 독립적으로 'FM + 감정 + 평가 데이터 미사용 + 짧은 블록' 연구를 찾지 못했다.  Compass 가 이 문제를 "
   "'미해결 핵심 과제' 로 꼽았다 — 서론의 동기 문장으로 쓸 수 있다.",
   "<b>축을 '어떤 대응으로 회전을 추정하나' 로 세우면 문헌과 맞물린다.</b>  분야 합의 (재중심화 → 회전은 대응이 필요) 를 그대로 받아, 대응이 "
   "없을 때 (중심화까지), 자극 시점이 있을 때 (SLA), 클래스 라벨이 있을 때 (라벨 회전 · prototype 혼합) 를 같은 예산에서 비교하는 구성이다.  "
   "우리 프레임워크의 단계와 정확히 대응한다.",
   "<b>관련 연구 절 구성 제안</b>: (1) BCI 캘리브레이션 단축 — 재중심화와 템플릿 전이  (2) 테스트 시점 적응 — 전달식 대 온라인, 실패 보고  "
   "(3) 교차 피험자 감정 인식 — 전달식 적응과 소수의 분리 보정 (PPDA)  (4) 뇌파 FM 과 사람 차이 문제 — 벤치마크  (5) 공유 자극 정렬 — "
   "하이퍼정렬 · SRM 부터 CLISA · mdJPT · SCORE · GRN 까지.",
   "<b>주장 문구 초안</b> (영문): \"We adapt hyperalignment — Procrustes alignment on time-locked responses to a shared stimulus — as a "
   "calibration-only step for an EEG foundation model: the shared film is used only during a short session-start calibration, and the "
   "transform is applied unchanged to subsequent data without any information about test stimuli.\"",
   "<b>시점 경고.</b>  SCORE (8월), EEG-Arena (9/26), Tao &amp; Chen (9/28) 처럼 최근 두 달 안에 핵심 논문이 여럿 나왔다.  투고 직전에 "
   "'Procrustes EEG emotion', 'hyperalignment EEG foundation model', 'calibration EEG foundation model' 로 다시 찾아야 한다."])

# ── 9. 참고문헌 ──
P("9. 꼭 인용할 논문 (범주별, 링크)", H2)
REFS = [
    ("고전 BCI · 정렬", [
        ("Lotte 2015, 캘리브레이션 단축 리뷰 (Proc. IEEE)", "https://doi.org/10.1109/JPROC.2015.2404941"),
        ("Wu, Xu, Lu 2022, BCI 전이학습 리뷰 (IEEE TCDS)", "https://arxiv.org/abs/2004.06286"),
        ("Vidaurre et al. 2011, LDA 비지도 적응 (IEEE TBME)", "https://doi.org/10.1109/TBME.2010.2093133"),
        ("Zanini et al. 2018, 리만 재중심화 (IEEE TBME)", "https://doi.org/10.1109/TBME.2017.2742541"),
        ("Rodrigues et al. 2019, RPA (IEEE TBME)", "https://doi.org/10.1109/TBME.2018.2889705"),
        ("He &amp; Wu 2020, 유클리드 정렬 (IEEE TBME)", "https://doi.org/10.1109/TBME.2019.2913914"),
        ("He &amp; Wu 2020, 라벨 정렬 (IEEE TNSRE)", "https://doi.org/10.1109/TNSRE.2020.2980299"),
        ("Mellot et al. 2023, 공분산 정렬 (Imaging Neurosci.)", "https://doi.org/10.1162/imag_a_00040"),
        ("Junqueira et al. 2024, EA + 딥러닝 (J. Neural Eng.)", "https://doi.org/10.1088/1741-2552/ad4f18"),
        ("Wu 2025, EA 재검토 (J. Neural Eng.)", "https://arxiv.org/abs/2502.09203"),
        ("Kobler et al. 2022, SPD 도메인별 BN (NeurIPS)", "https://arxiv.org/abs/2206.01323"),
        ("Bleuzé et al. 2022, 접공간 정렬 (Front. Hum. Neurosci.)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9755175/"),
        ("Lopes et al. 2026, EA = 재중심화", "https://arxiv.org/abs/2606.16462")]),
    ("SSVEP · c-VEP 템플릿 전이", [
        ("Yuan et al. 2015, tt-CCA (J. Neural Eng.)", "https://doi.org/10.1088/1741-2560/12/4/046006"),
        ("Wong et al. 2020, stCCA (IEEE TNSRE)", "https://ieeexplore.ieee.org/document/9177172"),
        ("Chiang et al. 2021, LST (J. Neural Eng.)", "https://doi.org/10.1088/1741-2552/abcb6e"),
        ("Bian et al. 2023, sd-LST (IEEE TNSRE)", "https://doi.org/10.1109/TNSRE.2022.3225878"),
        ("Liu et al. 2022, ALPHA (IEEE TBME)", "https://doi.org/10.1109/TBME.2021.3105331"),
        ("Chen et al. 2024, SSVEP-DAN (IEEE TNSRE)", "https://doi.org/10.1109/TNSRE.2024.3404432"),
        ("Deng et al. 2024, OS-SSVEP (Neural Networks)", "https://doi.org/10.1016/j.neunet.2024.106734"),
        ("Behboodi et al. 2026, c-VEP FM 보정", "https://arxiv.org/abs/2601.06028")]),
    ("테스트 시점 적응 · 소스 없는 적응", [
        ("Wimpff et al. 2024, 온라인 TTA (BCI Winter Conf.)", "https://arxiv.org/abs/2311.18520"),
        ("Li et al. 2024, T-TIME (IEEE TBME)", "https://doi.org/10.1109/TBME.2023.3303289"),
        ("Bakas et al. 2025, 잠재 정렬 (J. Neural Eng.)", "https://doi.org/10.1088/1741-2552/adb336"),
        ("Li, Kawanabe, Kobler 2025, SPDIM (ICLR)", "https://arxiv.org/abs/2411.07249"),
        ("Chu et al. 2026, OSPDIM (EUSIPCO)", "https://arxiv.org/abs/2608.05315"),
        ("Duan et al. 2025, 정렬 지배 소거", "https://arxiv.org/abs/2509.19403"),
        ("Wimpff et al. 2025, RAP (보정 블록 대표성)", "https://arxiv.org/abs/2507.06779"),
        ("Lee, Pradeepkumar, Sun 2026, NeuroAdapt-Bench (MLHC)", "https://arxiv.org/abs/2604.16926"),
        ("Wang et al. 2025, NeuroTTT", "https://arxiv.org/abs/2509.26301"),
        ("Li et al. 2026, NeuroOnline", "https://arxiv.org/abs/2607.03925"),
        ("Apicella et al. 2023, 정규화 ≥ 도메인 적응 (EAAI)", "https://doi.org/10.1016/j.engappai.2023.106205"),
        ("Imtiaz &amp; Khan 2026, 감정 SFDA (IEEE TAFFC)", "https://doi.org/10.1109/TAFFC.2026.3713402"),
        ("Boudiaf et al. 2022, LAME (CVPR)",
         "https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html"),
        ("Gong et al. 2022, NOTE (NeurIPS)",
         "https://papers.neurips.cc/paper_files/paper/2022/file/ae6c7dbd9429b3a75c41b5fb47e57c9e-Paper-Conference.pdf"),
        ("Zhao et al. 2023, TTAB (ICML)", "https://arxiv.org/abs/2306.03536"),
        ("Press et al. 2023, RDumb (NeurIPS)",
         "https://proceedings.neurips.cc/paper_files/paper/2023/file/7d640f377893fc5f22b5610e175ef7c3-Paper-Conference.pdf")]),
    ("감정 인식 — 캘리브레이션 · 정렬 · 평가", [
        ("Zhao, Yan, Lu 2021, PPDA (AAAI)", "https://doi.org/10.1609/aaai.v35i1.16169"),
        ("Li et al. 2020, 다중 소스 전이 + STM (IEEE TCYB)", "https://doi.org/10.1109/TCYB.2019.2904052"),
        ("Lin &amp; Jung 2017, 조건부 전이 (Front. Hum. Neurosci.)", "https://doi.org/10.3389/fnhum.2017.00334"),
        ("Lin 2020, 날짜 간 개인화 (IEEE JBHI)", "https://doi.org/10.1109/JBHI.2019.2934172"),
        ("Jin et al. 2024, EvoFA", "https://arxiv.org/abs/2409.15733"),
        ("Liu et al. 2025, FACE (분할 비교용)", "https://arxiv.org/abs/2503.18998"),
        ("Zheng &amp; Lu 2016, 개인화 감정 모델 (IJCAI)", "https://www.ijcai.org/Proceedings/16/Papers/388.pdf"),
        ("Chen et al. 2021, MS-MDA (Front. Neurosci.)", "https://doi.org/10.3389/fnins.2021.778488"),
        ("Zhou et al. 2024, PR-PL (IEEE TAFFC)", "https://arxiv.org/abs/2202.06509"),
        ("Fdez et al. 2021, 계층 정규화 (Front. Neurosci.)", "https://doi.org/10.3389/fnins.2021.626277"),
        ("Chen et al., Personal-Zscore (IEEE TAFFC)", "https://ieeexplore.ieee.org/document/9662246/"),
        ("Shen et al. 2022, CLISA (IEEE TAFFC)", "https://doi.org/10.1109/TAFFC.2022.3164516"),
        ("Shen et al. 2024, CL-SSTER (NeuroImage)", "https://arxiv.org/abs/2402.14213"),
        ("Xie et al. 2026, TA2CL", "https://arxiv.org/abs/2605.22379"),
        ("Li et al. 2021, 블록 설계 함정 (IEEE TPAMI)", "https://doi.org/10.1109/TPAMI.2020.2973153"),
        ("Kilgallen et al. 2025, 반복 자극 혼입", "https://arxiv.org/abs/2508.00531"),
        ("Lei et al. 2025, 시행 내 누출 (CEUR)", "https://ceur-ws.org/Vol-4115/paper7.pdf"),
        ("Suo et al. 2026, 체크포인트 선택", "https://arxiv.org/abs/2607.27655"),
        ("Kukhilava et al. 2025, 감정 인식 평가 리뷰", "https://arxiv.org/abs/2505.18175"),
        ("Apicella et al. 2024, 교차 피험자·세션 리뷰 (Neurocomputing)", "https://doi.org/10.1016/j.neucom.2024.128354"),
        ("Li et al. 2026, 교차 피험자 일반화 리뷰 (Front. Comput. Neurosci.)", "https://doi.org/10.3389/fncom.2026.1865513")]),
    ("파운데이션 모델 · 벤치마크", [
        ("Jiang et al. 2024, LaBraM (ICLR)", "https://arxiv.org/abs/2405.18765"),
        ("Wang et al. 2025, CBraMod (ICLR)", "https://arxiv.org/abs/2412.07236"),
        ("El Ouahidi et al. 2025, REVE (NeurIPS)", "https://arxiv.org/abs/2510.21585"),
        ("Zhou et al. 2025, CSBrain (NeurIPS)", "https://arxiv.org/abs/2506.23075"),
        ("Ma et al. 2026, CodeBrain (ICLR)", "https://arxiv.org/abs/2506.09110"),
        ("Zhang et al. 2025, mdJPT (NeurIPS)", "https://arxiv.org/abs/2510.22197"),
        ("Xiong et al. 2026, EEG-FM-Bench (ICML)", "https://arxiv.org/abs/2508.17742"),
        ("Liu et al. 2026, EEG-FM-Compass (Natl. Sci. Rev.)", "https://arxiv.org/abs/2601.17883"),
        ("Wu et al. 2025, AdaBrain-Bench", "https://arxiv.org/abs/2507.09882"),
        ("Chen et al. 2026, EEG-Arena", "https://arxiv.org/abs/2609.32743"),
        ("Lin, Wu, Jung 2026, The Identity Trap", "https://arxiv.org/abs/2606.06647"),
        ("Tao &amp; Chen 2026, 개인 대 모집단 이득", "https://arxiv.org/abs/2609.34801"),
        ("Sarhane et al. 2026, Stacked LoRA (WCCI)", "https://arxiv.org/abs/2607.03094"),
        ("Lee et al. 2025, Are Large Brainwave FMs Capable Yet? (ICML)", "https://arxiv.org/abs/2507.01196"),
        ("Kuruppu et al. 2026, FM 리뷰 (J. Neural Eng.)", "https://arxiv.org/abs/2507.11783"),
        ("Chen et al. 2023, FACED (Sci. Data)", "https://doi.org/10.1038/s41597-023-02650-w")]),
    ("같은 자극 정렬 (SLA 계보)", [
        ("Haxby et al. 2011, 하이퍼정렬 (Neuron)", "https://doi.org/10.1016/j.neuron.2011.08.026"),
        ("Chen et al. 2015, SRM (NeurIPS)", "https://proceedings.neurips.cc/paper/2015/hash/b3967a0e938dc2a6340e258630febd5a-Abstract.html"),
        ("Andreella et al. 2023, ProMises (Hum. Brain Mapp.)", "https://doi.org/10.1002/hbm.26170"),
        ("Zhang et al. 2017, MEG M-CCA (Hum. Brain Mapp.)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6866831/"),
        ("de Cheveigné et al. 2019, MCCA (NeuroImage)", "https://doi.org/10.1016/j.neuroimage.2018.11.026"),
        ("Bhattacharjee et al. 2026, ECoG 공유 공간 (Nat. Comput. Sci.)", "https://www.nature.com/articles/s43588-025-00900-y"),
        ("Tang &amp; Huth 2025, 참가자 간 의미 디코딩 (Curr. Biol.)", "https://doi.org/10.1016/j.cub.2025.01.024"),
        ("Thual et al. 2023, 기능 정렬로 새 피험자 디코딩", "https://arxiv.org/abs/2312.06467"),
        ("Cui et al. 2026, SCORE", "https://arxiv.org/abs/2608.19134"),
        ("Meng 2026, GRN (ICANN)", "https://arxiv.org/abs/2603.11119"),
        ("Hajlaoui et al. 2018, 감정과 피험자 간 상관", "https://arxiv.org/abs/1809.08273"),
        ("Zhang &amp; Liu 2013, 스타일 전이 사상 STM (IEEE TPAMI)", "https://doi.org/10.1109/TPAMI.2012.239"),
        ("Michalke &amp; Rieger 2025, BCI 보정 웜스타트 제안 (CCN)",
         "https://2025.ccneuro.org/abstract_pdf/Michalke_2025_Functional_Inter-Subject_Alignment_Outperforms_Anatomical_Alignment.pdf")]),
]
for cat, items in REFS:
    P(cat, H3)
    for i in range(0, len(items), 2):
        P(" &nbsp;·&nbsp; ".join(L(t, u) for t, u in items[i:i + 2]), REF)
P("이미 REVIEW.pdf 에 있는 것 (Jiahui 2020, RSRM-EEG 2021, Parra 연구실 ISC, AdaBN, Wimpff 2024 등) 은 그쪽 링크를 쓴다.", SMALL)

# ── 10. 한계 ──
P("10. 조사 한계와 확인 상태", H2)
B(["<b>직접 다시 확인한 것</b> (arXiv 원문): SCORE (직교 변환, 라벨·인코더 갱신 없음) · GRN (본문: \"같은 자극 시간축에 맞춘 샘플 사이에서만 "
   "공명 계산\", 자극을 어긋나게 맞추면 87.90 → 84.83) · mdJPT (본문: 같은 자극의 두 사람을 양성 쌍으로 하는 정렬 손실, 시각 맞춤 65.02 대 "
   "비정렬 55.47; 소수샷은 소수 피험자의 라벨로 분류기 학습 — 개인 보정이 아님) · Tao &amp; Chen (수치 일치) · EEG-Arena (존재와 초록; 같은 사람 "
   "안 대 분리 수치는 본문이 너무 커서 재확인 못 함).",
   "<b>웹 검색 한도</b>: 세션 공용 200회가 조사 중 소진돼 뒤쪽은 arXiv API · Crossref 로 검증했다.  2026 하반기 저널 · 중국어권 학술지가 "
   "빠졌을 수 있다.  더 넓게 확인하려면 한도를 올려야 한다 (Claude Code 설정 — 바꿀지는 사용자 판단).",
   "<b>확인 못 한 세부</b> (원 보고서에 UNVERIFIED 표시): Gram 의 감정 분할 · FUSED/ECHO 수치 · STEM/Stacked LoRA 보정량 · AdaBrain 게재처 · "
   "Personal-Zscore 의 시험 통계 출처 · Fdez 2021 의 시험 정규화 방식 · RPA 회전의 라벨 필요 (pyRiemann 문서로만) · Dmochowski 2012 DOI · "
   "준지도 SRM (Turek 2017) · 연결성 하이퍼정렬 (Guntupalli 2018).",
   "<b>인용 전 원문 확인</b>: 이 문서의 문헌 수치는 조사 에이전트가 원문·초록에서 옮긴 것이다.  논문에 쓰기 전에 해당 원문을 한 번 더 본다."])


# ── 글리프 검사와 출력 ────────────────────────────────────────────────────
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
    c.drawString(22 * mm, 12 * mm, "캘리브레이션 연구 지형 조사 (2026-10-05)")
    c.drawRightString(A4[0] - 22 * mm, 12 * mm, str(d.page))
    c.setStrokeColor(LINE); c.setLineWidth(0.4); c.line(22 * mm, 15 * mm, A4[0] - 22 * mm, 15 * mm); c.restoreState()


check_glyphs()
doc = BaseDocTemplate("reports/CALIBRATION_SURVEY.pdf", pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                      topMargin=18 * mm, bottomMargin=20 * mm, title="캘리브레이션 연구 지형 조사")
doc.addPageTemplates([PageTemplate(id="a", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)],
                                   onPage=footer)])
doc.build(E)
print("[saved] reports/CALIBRATION_SURVEY.pdf")
