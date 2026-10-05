"""CAFT 관련 연구 조사 — reports/CAFT_RELATED_WORK.pdf (2026-10-05 사용자: "지금 돌리는 것의 related work 를 조사해서 survey 를").

다섯 갈래 병렬 조사 (reports/survey_raw_caft/B1~B5) 를 종합한다.  문헌 내용 · 수치는 원 보고서에서 옮긴다 (원 보고서가 1차 출처에서
확인한 것, 확인 못 한 것은 각 보고서 끝).  우리 수치는 중간점검 (5명) 만 맥락으로 쓴다.  약어는 처음 나올 때 풀어 쓴다.
끝에 폰트에 없는 글자를 검사한다.

    python make_caft_survey_pdf.py
"""
from __future__ import annotations

import datetime
import re

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
LINE, BG, NOTE, OURS, WARN = (colors.HexColor("#d4d9dd"), colors.HexColor("#f5f7f9"), colors.HexColor("#eef4ef"),
                              colors.HexColor("#fbeae4"), colors.HexColor("#fdf3e1"))
LINKC = "#2c5f8a"


def st(name, size=9.3, lead=14.2, color=INK, font=F, space=4, left=0):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead, textColor=color, spaceAfter=space,
                          leftIndent=left, alignment=TA_LEFT)


BODY = st("b"); BUL = st("bul", left=10); BUL2 = st("bul2", 8.9, 13.4, left=20)
TITLE = st("t", 18, 25, INK, FB, 5); SUBT = st("st", 11.5, 16, INK, F, 6)
H2 = st("h2", 12.5, 17, ACC, FB, 6); H2.keepWithNext = 1
H3 = st("h3", 10.2, 14.5, INK, FB, 3); H3.keepWithNext = 1
H2S, H3S = ParagraphStyle("h2s", parent=H2), ParagraphStyle("h3s", parent=H3)
H2S.keepWithNext = H3S.keepWithNext = 0
SMALL = st("s", 8.0, 11.6, GREY, space=3); REF = st("r", 7.8, 11.2, INK, space=1.6, left=8)
CELL = st("c", 7.6, 10.5, INK, space=0); CELLB = st("cb", 7.6, 10.5, INK, FB, space=0)
BOX = ParagraphStyle("box", fontName=F, fontSize=9.2, leading=14.2, textColor=INK, backColor=NOTE, borderPadding=8,
                     spaceAfter=9, leftIndent=4, rightIndent=4)
WBOX = ParagraphStyle("wbox", parent=BOX, backColor=WARN)
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
    s = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.6),
         ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6), ("LEFTPADDING", (0, 0), (-1, -1), 3.2),
         ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
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
NOW = f"{datetime.date.today():%Y-%m-%d}"

# ════ 머리 ════════════════════════════════════════════════════════════════
P("CAFT 관련 연구 조사", TITLE)
P("파인튜닝 안에서 캘리브레이션을 흉내 내는 방법들 — 무엇이 이미 있고, 무엇이 남는가", SUBT)
P(f"{NOW} · 다섯 갈래 병렬 조사 (갈래당 33~35편, 중복 포함 180건) · 원 보고서: reports/survey_raw_caft/B1~B5 · 지난 캘리브레이션 "
  "조사 (CALIBRATION_SURVEY.pdf, A1~A5) 의 확인된 인용을 재사용했다", SMALL)
gap(3)
P("<b>결론.</b>  캘리브레이션 인지 파인튜닝 (CAFT) 과 '거의 같은' <b>출판</b> 논문은 찾지 못했다.  그러나 <b>부품은 모두 선례가 있다</b>.  "
  "원리 (학습 조건 = 배포 조건) 는 Matching Networks · 적응형 위험 최소화 (ARM-BN) · MetaBN · 음성의 화자 적응 학습 (SAT) 에, "
  "① 학습 중 피험자별 정규화는 층화 정규화 · CLISA · 잠재 정렬 · Kwak 2023 에, 배치 설계 (여러 피험자 × 같은 자극 시점) 와 "
  "② 같은 자극 사람 간 정렬은 CLISA → CL-SSTER → mdJPT → MSHCL 계보에 있다.  ICLR 2027 에 심사 중인 동시 투고 두 편 (PACE, SCC) "
  "이 특히 가깝다.  <b>남는 새로움은 조합이다</b> — 파운데이션 모델 (FM) 전체 지도 파인튜닝 안에서 배포와 똑같은 무파라미터 평균 "
  "빼기를, 캘리브레이션 블록 구조의 배치 (같은 세션 · 감정 균형 · 같은 클립 · 초) 로 학습하고, 평가 데이터와 분리된 블록으로 "
  "배포하는 비전이적 프로토콜에서 감정 피험자 하나 빼기 교차검증 (LOSO) 으로 검증한 것.  따라서 '새 방법' 보다 <b>'배포 일관 "
  "파인튜닝 원칙 + 현실적 캘리브레이션 프로토콜 + 체계적 분석'</b> 으로 위치를 잡아야 한다.  이름 'Calibration-Aware Fine-Tuning' · "
  "'CAFT' 는 다른 분야에서 이미 쓰이므로 바꾸는 것이 좋다 (제목 'Fine-Tune Like You Calibrate' 는 충돌 없음).", BOX)

# ── 1. 조사 방법 ──
P("1. 조사 방법", H2)
T([["갈래", "범위", "편수", "원 보고서"],
   ["B1 적응을 흉내 낸 학습", "그룹 단위 배치 학습 (ARM, 도메인별 배치 정규화 DSBN, MetaBN), 음성 화자 적응 학습 · 켑스트럼 평균 · 분산 "
    "정규화 (CMVN), 생체신호 사용자별 정규화, EEG 학습 중 정규화, 이론", "34", "B1_adaptation_aware_training.md"],
   ["B2 시험처럼 학습", "에피소드 · 메타학습, 사전학습처럼 파인튜닝 (FLYP), 시험 시점 학습 (TTT), BCI 캘리브레이션 메타학습, 이름 충돌",
    "35 (+5)", "B2_train_like_you_test.md"],
   ["B3 같은 자극 사람 간 정렬", "CLISA 계열, 딥 하이퍼정렬 · 다중 피험자 정렬, 정렬 손실의 위험과 처방, 자극 지름길", "33",
    "B3_shared_stimulus_alignment.md"],
   ["B4 FM 파인튜닝", "EEG FM 파인튜닝 · 매개변수 효율 파인튜닝 (PEFT) · 피험자 적응, 벤치마크 프로토콜, 다른 모달리티, 동시 투고",
    "35", "B4_fm_finetuning.md"],
   ["B5 교차 피험자 감정 인식", "학습 중 정규화 · 피험자 불변 학습, 적은 캘리브레이션 계열, SEED-V 수치와 공정 비교", "34",
    "B5_emotion_cross_subject.md"]], [34, 84, 14, 34])
B(["모든 논문의 제목 · 저자 · 연도 · 게재처는 1차 출처 (arXiv, DOI · Crossref, OpenAlex, 출판사, OpenReview, 공개 코드) 에서 확인했다.  "
   "본문을 읽은 것, 공개 코드로 확인한 것, 초록만 본 것을 원 보고서에 구분해 적었고, 확인 못 한 항목은 각 보고서 끝에 모았다.",
   "일부 사실은 2026-10-05 기준 공개 코드로 확인했다 (mdJPT 의 배치 샘플러 · 정규화, DMMR · MS-MDA · DAEST 의 평가 코드).  각 논문의 "
   "보고 수치가 그 코드 경로로 나왔는지는 확인하지 못했다.",
   "ICLR 2027 투고작 (PACE, SCC, SEAM, NeuroContext-TTA) 은 OpenReview 본문이 막혀 초록 기준이고, PACE 만 익명 공개 코드로 세부를 보강했다."])

# ── 2. 지형 ──
P("2. 부품별로 누가 이미 했나 — 지형 표", H2)
T([["연구", "모델", "학습 중 피험자별 정규화", "학습 연산 = 배포 연산", "배포 통계 출처", "배치 = 여러 피험자 × 같은 자극 시점",
    "사람 간 같은 자극 정렬", "새 파라미터"],
   ["층화 정규화 (Fdez 2021)", "작은 MLP", "○ 평균 · 분산, 세 층", "○ (전이적)", "평가 세션 전체", "△", "×", "0"],
   ["CLISA (Shen 2023)", "작은 CNN, 대조 사전학습", "○ z-점수 (입력 · 풀링 · 투영기)", "×", "평가 스트림 온라인", "○ 2명", "○ InfoNCE",
    "투영기"],
   ["mdJPT (Zhang 2025)", "감정 전용 사전학습", "○ z-점수 (코드)", "×", "평가 스트림 온라인", "○ 2명, 같은 세션", "○", "모델 전체"],
   ["잠재 정렬 (Bakas 2025)", "EEGNet 등", "○ 평균 · 분산, 여러 층", "○", "평가 시행 전체", "× (4명 × 12시행)", "×", "공유 아핀"],
   ["Kwak 2023 (운동 상상)", "작은 DNN", "○ 기준 신호 빼기 (학습 모듈)", "○", "1분 휴지기 (분리)", "×", "△ 같은 클래스", "모듈"],
   ["Du 2017 (근전도)", "ConvNet", "○ 세션별 BN", "○", "무라벨 캘리브레이션 (분리)", "×", "×", "0"],
   ["ARM-BN (Zhang 2021)", "이미지 모델", "○ 도메인 배치 BN", "○", "평가 배치 (전이적)", "×", "×", "0"],
   ["MetaBN (Bronskill 2020)", "메타학습 모델", "○ 문맥 집합 BN", "○", "문맥 집합 (비전이적)", "—", "×", "0"],
   ["PACE (ICLR 2027 투고)", "FM 4종, 사후학습", "○ 블록 중심화 + 백색화", "○", "평가 분할 안 32창 (전이적)", "○ 쌍",
    "○ InfoNCE + 같은 감정", "투영기"],
   ["SCC (ICLR 2027 투고)", "FM 5종", "○ 학습형 오프셋 차감", "○", "분리된 무라벨 문맥", "미확인", "미확인", "추정기"],
   ["Tao & Chen 2026", "CBraMod (동결)", "문맥 조건화 (FiLM · LoRA)", "○", "휴지기 · 앞쪽 시행", "×", "×", "생성기 — 결과 무효과"],
   [O("CAFT (우리)"), O("LaBraM 전체 지도 FT"), O("○ 평균만, 분류 토큰 한 곳"), O("○ 동일"), O("분리된 감정 균형 블록"),
    O("○ 같은 세션 4명 × 15곳"), O("○ 코사인 (양성만)"), O("0")]],
  [27, 20, 23, 15, 22, 21, 19, 19], hl=(12,), split=True)
P("○ 있음, △ 부분, × 없음.  '전이적' 은 평가 데이터 자체로 정규화 통계를 만든다는 뜻이다.  원 보고서 B1 · B2 · B4 · B5 의 비교표를 합쳤다.",
  SMALL)
P("<b>읽는 법.</b>  CAFT 의 각 칸에는 선례가 있다.  CAFT 에만 있는 것은 <b>줄의 조합</b>이다: FM 전체 지도 파인튜닝 + 배포 연산과 "
  "정확히 같은 무파라미터 평균 빼기 + 평가 데이터와 분리된 감정 균형 캘리브레이션 블록 (비전이적) + 그 블록 구조를 복제한 배치 + "
  "감정 LOSO 실증.", BODY)

# ── 3. 갈래별 ──
P("3. 갈래별 요약", H2)
P("B1 — 배포 때의 정규화를 학습 때도 똑같이 하는 학습법", H3)
B(["<b>① 의 원리는 여러 분야에서 다른 이름으로 반복해 나왔다.</b>  음성의 SAT (1996) 와 CMVN ('정규화된 특징으로 모델을 학습'), "
   "근전도의 다중 스트림 AdaBN (Du 2017: 학습은 세션별 통계, 배포는 무라벨 캘리브레이션), 행동 인식의 사용자별 BN (2020), 일반 "
   "기계학습의 ARM-BN · MetaBN, EEG 의 층화 정규화 · SPD 도메인별 BN · 잠재 정렬 ('학습과 추론의 행동 차이를 없앤다').",
   "<b>문헌상 기대 효과는 작고 들쭉날쭉하다.</b>  '배포 때만 정규화' 에 '학습 때도 같은 정규화' 를 더한 추가 이득: 잠재 정렬 대 "
   "AdaBN +0.3~1.8%p, 다중 스트림 대 AdaBN +0.2~1.2%p, ARM-BN 대 BN 적응 FEMNIST +3.2%p (최악 사용자 −1.2%p), WILDS FMoW "
   "−9.6%p.  우리 중간점검 (5명, 중심화 뒤 클립 +0.06) 은 이 범위보다 크다 — 16명 · 여러 데이터셋에서 유지되는지가 관건이다.",
   "<b>짧은 명제로 쓸 수 있다.</b>  피험자 특징을 z = φ(x) + b<sub>s</sub> 로 두면, 같은 내용 · 감정 구성의 블록 평균을 빼는 순간 "
   "b<sub>s</sub> 가 정확히 사라진다 (구성이 다르면 그 차이만큼 편향, 추정 분산은 창 수 n 에 대해 1/n).  감정 균형과 자극 위치 "
   "일치가 바로 이 조건을 학습 · 배포에서 맞추는 장치다.  '새 이론' 이 아니라 정규형 · 집단 평균 중심화 문헌의 특수 경우로 쓴다."])
P("B2 — '시험처럼 학습' 원리", H3)
B(["원리는 Matching Networks (2016) 의 'test and train conditions must match' 이고, ProtoNets · MAML · 에피소드 도메인 일반화 · "
   "TTT · FLYP 가 같은 계열이다.  원전으로 인용한다.",
   "ARM 저자들은 자기 방법들을 'straightforward extensions ... and this is intentional' 이라고 썼다 — 기존 적응 연산에 학습 단계를 "
   "맞춰 주는 것 자체가 정당한 기여 유형이라는 선례다.  '이전 연구의 사후 중심화 → CAFT' 는 ARM 의 'BN 적응 → ARM-BN' 과 같은 관계다.",
   "<b>Tao & Chen (arXiv 2026-09)</b> 은 EEG FM (CBraMod) 에 캘리브레이션 문맥을 쓰도록 에피소드로 학습했지만 진짜 문맥과 섞은 "
   "문맥의 차이가 중앙값 0.00%p 였다.  CAFT 가 같은 대조군 (섞은 · 교환 캘리브레이션, 학습량 맞춤) 아래 양의 효과를 보이면 "
   "'학습형 문맥 조건화는 효과가 없었는데 고정 연산의 학습 일치는 효과가 있다' 는 대비가 그 자체로 기여가 된다.",
   "<b>TaskNorm 이 짚을 약점:</b> 학습 때 분류 대상 15창이 자기 평균에 들어간다 (배포는 분리된 블록).  평균용 창과 손실용 창을 "
   "나누는 변형을 대조로 넣어야 '같은 연산' 주장이 단단해진다."])
P("B3 — 학습 단계의 같은 자극 사람 간 정렬 (② 의 선례)", H3)
B(["② 의 핵심 아이디어는 확립된 계열이다: CLISA (TAFFC 2023) → CL-SSTER (NeuroImage 2024) → mdJPT (NeurIPS 2025) → MSHCL "
   "(TAFFC 2025) → TA2CL (2026).  mdJPT 는 시각을 어긋나게 한 양성쌍이 성능을 크게 떨어뜨리고, 같은 시각 양성쌍이 지도 대조학습 "
   "(SupCon) 보다 낫다고 보였다.  MSHCL 공개 코드는 같은 위치 InfoNCE 와 감정 교차 엔트로피를 한 단계에서 함께 학습한다.",
   "찾지 못한 것: (a) 사전학습 FM 을 전체 파인튜닝하며 보조 손실로 쓰기, (b) 음성쌍 없는 4명 다중 뷰 코사인, (c) 학습 중심화 공간과 "
   "배포 공간의 일치 — (a)(b) 는 설계 변형 수준, (c) 는 ① 쪽 기여다.",
   "<b>우리 중간 결과 (② 가 방향 일치도 c 를 0.38 → 0.57 로 올리지만 정확도 이득은 작다) 는 문헌으로 설명된다</b>: 불변성을 세게 "
   "걸면 고정 특징의 전이성 · 판별성이 떨어질 수 있고 (Kornblith 2021, BSP 2019), 음성쌍 없는 정렬은 차원 붕괴를 부를 수 있으며 "
   "(Jing 2022), CLISA · mdJPT 는 손실을 버리는 사영 헤드에 걸지만 CAFT 는 배포 표현 z 에 직접 건다 (Guillotine 2023).",
   "<b>가장 큰 타당성 위험은 자극 지름길이다.</b>  Gerster 2026 (bioRxiv, 동료 심사 전) 은 FACED 에서 교차 피험자 분류가 감정보다 "
   "영상 정체를 따른다고 보였다 (감정당 영상을 1개로 줄이면 정확도가 오르고, 자기보고 라벨로 바꾸면 떨어진다).  CLISA 도 처음 보는 "
   "자극에서는 이득이 SEED +6.5%p → +3.0%p 로 줄었다.  SEED-V 는 세션마다 영상이 완전히 다르므로 교차 세션 평가가 곧 처음 보는 영상 평가다."])
P("B4 — EEG FM 파인튜닝과 빈틈", H3)
B(["출판된 동일 연구는 없다.  다만 ICLR 2027 동시 투고 <b>PACE</b> 가 FM 4종에서 피험자별 무라벨 32창 블록으로 토큰을 중심화 · "
   "백색화 (학습과 하류 평가 모두) 하고, 같은 세션 · 같은 자극 · 같은 시각 쌍을 양성으로 쓴다.  <b>SCC</b> 는 FM 5종에서 문맥 오프셋을 "
   "학습해 빼며, 피험자 차이 ≈ 평행이동이라는 우리 배포 가설을 독립적으로 지지한다.  SEAM · NeuroContext-TTA 도 배포 보정을 "
   "원천 피험자로 학습하는 같은 방향이다.",
   "<b>벤치마크 공백:</b> FM 감정 벤치마크 가운데 짧은 캘리브레이션 블록을 경사하강 없이 쓰는 프로토콜은 없다.  새 피험자 데이터를 "
   "쓰는 프로토콜 (Compass, 'Are EEG FMs Worth It?') 은 모두 그 피험자 라벨로 미세조정한다.",
   "FM 표현은 피험자 정체가 지배하고 미세조정이 이를 키운다 (Identity Trap 2026) — FM 파인튜닝에서 학습 시점 중심화가 필요한 동기로 쓸 수 있다."])
P("B5 — 교차 피험자 감정 인식", H3)
B(["CLISA · mdJPT · DAEST 가 '같은 세션 여러 피험자 × 클립마다 같은 (클립, 시각)' 배치, 배치 안 피험자별 z-점수, 같은 시각 피험자 간 "
   "손실을 함께 쓴다 (mdJPT 는 공개 코드로 확인).  ① 의 논리 ('배포 때의 피험자별 정규화를 학습에도') 는 층화 정규화 (Fdez 2021) 와 "
   "Kwak 2023 에 있다.",
   "SEED-V 교차 피험자 문헌 수치 (약 58~82%) 는 우리와 직접 비교할 수 없다 — 시행 전체에 거는 비인과 평활, 평가 피험자 데이터로 "
   "하는 정규화, 평가 정확도로 하는 모델 선택 (DMMR · MS-MDA · DAEST 공개 코드), 첫 세션만 쓰는 평가 때문이다 (8절).",
   "이 분야에서는 같은 부품의 변형도 TAFFC 급에 실린다 — 분야 관례상 조합 기여로 인정되지만, CLISA · mdJPT · 층화 정규화를 부품의 "
   "원전으로 반드시 인용해야 한다."])

# ── 4. 정밀 비교 ──
P("4. 가장 가까운 선행 — 정밀 비교", H2)
T([["선행", "같은 점", "다른 점", "우리 대응"],
   ["CLISA (Shen 2023, TAFFC) — 출판 선행 중 가장 가까움", "여러 피험자 × 같은 시간 구간 배치, 배치 안 피험자별 정규화, 같은 구간 "
    "정렬이 한 논문에 모두 있다", "작은 CNN 의 무라벨 대조 사전학습, 분류는 그 뒤 별도 단계.  예측 때 정규화는 평가 스트림으로 갱신 "
    "(전이적), z-점수 · 여러 층, 음성쌍 있는 대조", "학습 연산 = 비전이적 배포 연산으로 설계한 점을 앞세우고, CLISA식 설정 (z-점수, "
    "여러 층, 평가 스트림 통계) 과 직접 비교"],
   ["잠재 정렬 (Bakas 2025, JNE) — ① 단독의 가장 가까운 선행", "4명 × n시행 배치, 피험자별 정규화를 학습 · 추론에 똑같이",
    "작은 모델, 평균 · 분산 · 여러 층, 평가 시행 전체 (전이적), 감정 · 자극 정렬 없음.  불균형 문맥에서 붕괴 보고",
    "FM 에 잠재 정렬을 그대로 적용한 기준선 (분류 토큰 평균 + 분산, 여러 층) 으로 '평균만 · 최종층만' 선택을 정당화, 감정 불균형 블록 실험"],
   ["ARM-BN (2021) · MetaBN (2020) · Du 2017 — 원리 · 연산의 원형", "그룹 배치로 정규화하며 학습 → 배포 때 같은 연산, 새 파라미터 · "
    "배포 경사하강 없음.  MetaBN 은 비전이적, Du 2017 은 분리된 캘리브레이션으로 배포", "이미지 · 소수샷 · 근전도, 모든 BN 층 평균 · 분산, "
    "배치가 캘리브레이션 블록 구성 아님.  ARM-BN 은 FMoW 에서 악화", "원리를 숨기지 말고 인용 · 정식화 ('학습 목적을 배포 위험의 대리로 "
    "맞춘다').  'BN 적응 대 ARM-BN' 형식의 핵심 대조 (사후 중심화 대 CAFT)"],
   ["PACE · SCC (ICLR 2027 투고, 심사 중) — 가장 가까운 동시 연구", "FM, 학습 · 배포가 일관된 피험자 블록 정규화 (PACE), 같은 세션 · "
    "같은 자극 · 같은 시각 양성 (PACE), 학습형 중심화 (SCC)", "PACE 는 사후학습 뒤 일반 파인튜닝, 무작위 32창 블록 (평가 분할 안, "
    "전이적), 백색화 포함, InfoNCE.  SCC 는 오프셋 추정기를 학습", "동시 연구로 인용 (ICLR 심사 지침상 비교 의무는 없지만 안전하게).  "
    "무작위 대 캘리브레이션 구조 블록, 백색화 유무, 학습형 오프셋 대 평균 대조"],
   ["Kwak 2023 (JBHI, 운동 상상) · Tao & Chen 2026", "짧은 별도 기록으로 피험자 성분을 빼는 연산을 학습 · 배포에 똑같이 (Kwak), EEG FM 에서 "
    "캘리브레이션 문맥을 쓰도록 에피소드 학습 (Tao & Chen)", "휴지기 기준 · 학습형 모듈 (Kwak), 동결 백본 · 학습형 생성기 · 결과 무효과 "
    "(Tao & Chen)", "'휴지기 대신 감정 균형 자극 블록' (중립 클립만 대조), 섞은 · 교환 캘리브레이션과 학습량 맞춤 대조"]],
  [34, 42, 46, 44], split=True)

# ── 5. 새로움 판정 ──
P("5. 새로움 판정 — 쓸 수 있는 주장과 쓰면 안 되는 주장", H2)
T([["쓰면 안 되는 주장 (선례)", "쓸 수 있는 주장 (검색 범위 안, 'to our knowledge' 로 한정)"],
   ["'같은 자극 · 같은 시점 배치를 처음 썼다' (CLISA, mdJPT, DAEST)", "EEG FM 전체 지도 파인튜닝 안에서, 분류 토큰 평균 빼기 하나를 "
    "배포의 비전이적 캘리브레이션 블록 연산과 똑같이 학습에 넣은 설계"],
   ["'배치 안 피험자별 정규화를 처음 썼다' (CLISA, mdJPT, 층화 정규화)", "배치를 캘리브레이션 블록 구성 (같은 세션, 감정 균형, 피험자 간 "
    "같은 (클립, 초)) 으로 짠 것"],
   ["'학습과 배포의 정규화를 일치시킨 첫 연구' (층화 정규화, Kwak, 잠재 정렬, ARM-BN)", "평가 데이터와 분리된 캘리브레이션 블록, "
    "캘리브레이션 클립을 평가에서 뺀 비전이적 프로토콜에서의 감정 LOSO 실증 (클립 · 창 단위)"],
   ["'피험자 간 같은 자극 시점 손실이 새롭다' (CLISA, CL-SSTER, mdJPT, TA2CL)", "학습형 문맥 조건화의 무효과 (Tao & Chen) 와 대비되는, "
    "고정 연산의 학습 일치 효과 (대조군을 갖춘 경우)"],
   ["'EEG FM 을 캘리브레이션 에피소드로 학습한 첫 사례' (Tao & Chen, STEM)", "FM 벤치마크에 없는 '짧은 블록 · 경사하강 없음' 캘리브레이션 "
    "프로토콜 자체"]], [80, 86])
P("<b>권장 영문 문단 (원 보고서 B2 · B5 초안을 합침).</b>  <i>CAFT follows the principle that training conditions should match "
  "deployment conditions (Vinyals et al., 2016), and in particular meta-training a model under the very normalization it will use "
  "at test time (ARM-BN; MetaBN; camera-based BN).  Subject-wise normalization during training has been used for EEG (stratified "
  "normalization; CLISA; Kobler et al.), and the batch sampler and stimulus-locked cross-subject objective follow CLISA and mdJPT.  "
  "Unlike them, we fine-tune an EEG foundation model on exactly the deployment-time centered representation — the mean of a short, "
  "emotion-balanced calibration block, held out from testing — with no added parameters and no test-time optimization.</i>  "
  "'first' 는 쓰지 않고, 꼭 써야 하면 범위를 좁힌다.", BODY)

# ── 6. 리뷰어 대비 ──
P("6. 리뷰어가 요구할 기준선 · 대조 실험", H2)
T([["", "실험", "근거 (선례)"],
   ["★1", "사후 중심화 대 CAFT (같은 모델 · 같은 학습량) — 지금 시험이 이것", "ARM 의 'BN 적응 대 ARM-BN'"],
   ["★2", "2×2: 표집기 {무작위, CAFT 배치} × 연산 {중심화 없음, 배치 안 중심화} — 배치 구조만의 효과 분리", "ARM, CLISA · mdJPT"],
   ["★3", "이득 분해: 피험자 배치 + 무작위 창 (ARM-BN형) / 감정 균형만 / 균형 + 자극 일치", "B1 C-2"],
   ["★4", "정규화 형태 · 위치: 평균만 대 평균 + 분산 (z-점수) 대 여러 층 대 입력 유클리드 정렬 (EA) — 배포도 같은 연산으로",
    "층화 정규화, CLISA, 잠재 정렬"],
   ["★5", "전이적 상한 (평가 세션 평균) 과 하한 (무보정) 을 함께", "선행 수치와의 다리"],
   ["★6", "교환 · 섞은 캘리브레이션 (남의 블록 평균) — 개인 특이성", "Tao & Chen"],
   ["★7", "처음 보는 영상 평가 (SEED-V 교차 세션) + 시간 셔플 짝 대조 — 자극 지름길 배제", "Gerster 2026, CLISA"],
   ["★8", "규모: 16명 전원 · 시드 3개 · SEED · SEED-IV · DEAP · FACED · FM 3종 (LaBraM, CBraMod, REVE)", "B4 · B5"],
   ["9", "자기 포함 평균 제거: 평균용 · 손실용 창 분리, 자기 제외 평균, 평균 기울기 차단", "TaskNorm · MetaBN"],
   ["10", "추정 잡음 · 블록 모양: M ∈ {5, 15, 30, 세션} × 배포 길이 {20 s, 40 s, 끝까지}, 감정 불균형 블록, 수축 평균", "TaskNorm, Wu & Johnson"],
   ["11", "학습형 대안: 문맥 조건화 (ARM-CML, FiLM) · 학습형 오프셋 (SCC식) 대 평균 — '왜 고정 연산인가'", "ARM-CML, SCC"],
   ["12", "FM 기준선: 선형 탐침, LoRA ± CAFT 배치, 피험자 임베딩, 캘리브레이션 라벨로 소수샷 미세조정 (같은 데이터량 · 비용 함께)",
    "Compass, 'Worth It'"],
   ["13", "PACE식 블록 정규화 (중심화 + 백색화, 무작위 32창) 대 캘리브레이션 구조 블록", "PACE"],
   ["14", "학습 손실을 배포 분류기와 맞추기: 다른 3명으로 감정 prototype, 나머지 1명으로 질의하는 코사인 손실", "ProtoNets, Meta-Baseline"],
   ["15", "② 대조: 양성쌍 정의 (같은 클립 다른 초 · 같은 감정 다른 클립 · 무작위), InfoNCE 대 코사인, 중심화 전 대 후", "mdJPT 부록"],
   ["16", "학습 중 도메인 적응 · 감정 분야 기준선: CLISA · mdJPT 식 학습을 LaBraM 에, 층화 정규화를 블록 통계로, 영역 적대 학습 (DANN) · "
    "MS-MDA (블록 = 대상), DMMR 또는 MAT", "B5 D"]],
  [10, 120, 36], hl=(1,), split=True)
P("모든 조건은 같은 배포 (캘리브레이션 블록 평균 빼기 → prototype) 로 평가하고, 피험자 단위 짝 비교 · 평균과 최악 피험자 · 무효과와 "
  "음의 결과까지 모두 보고한다.  ★ = 핵심 주장을 지키는 데 필요 (★1 은 지금 돌고 있는 시험).", SMALL)

# ── 7. ② 개선 단서 ──
P("7. ② 개선 단서 (B3 의 문헌 근거)", H2)
P("중간점검 (5명): ② 를 더하면 c 가 크게 오르지만 정확도 이득은 ① 만보다 작고, 평균을 빼지 않은 경로는 −0.045.  아래는 16명 결과를 "
  "본 뒤 GPU 사용 승인을 받아 시험할 후보다 (우선순위 순).", SMALL)
B(["<b>가중:</b> λ 0.5 → 0.05~0.2, 점증 일정 또는 교차 엔트로피와 기울기가 충돌할 때 끄는 게이트.",
   "<b>손실 위치:</b> z 대신 버리는 작은 사영 헤드, z 의 일부 차원, 또는 감정 판별 부분공간에만 (CLISA · mdJPT 는 사영 헤드에 건다).",
   "<b>붕괴 방지:</b> 분산 · 불변 · 공분산 정규화 (VICReg) 의 분산 · 공분산 항, 또는 사람 간 Barlow Twins (딥 CCA형 상관 정렬).",
   "<b>음성쌍:</b> 다른 위치를 음성으로 쓰는 InfoNCE (같은 감정 위치는 음성에서 제외).",
   "<b>템플릿 정렬:</b> 나머지 3명 평균 (기울기 차단) 에 맞춘다 — 배포 템플릿 (자극 시점 정렬) 과 일관.",
   "<b>가중 · 지연:</b> 위치별 사람 간 상관 (ISC) 신뢰도로 가중하고 ±1~2초 지연을 허용 (TA2CL).",
   "<b>진단:</b> c 와 함께 새 사용자의 클래스 간 / 내 분산비, 유효 차원, 처음 보는 영상에서의 c · 정확도를 본다."])

# ── 8. 공정 비교 ──
P("8. SEED-V 문헌 수치와 공정 비교 (B5)", H2)
T([["출처", "SEED-V (5감정, 우연 20%)", "프로토콜 주의", "우리와 비교"],
   ["MST (Qing & Li, arXiv 2026)", "LaBraM 32.5 ± 4.9, CBraMod 31.5 ± 5.1, MST 36.1 ± 4.5 (창)", "LOSO 16명 3세션, 원신호, 평활 없음, "
    "평가 데이터 통계 안 씀", "가장 공정 — 우리 '적응 없음' 창 단위와 바로"],
   ["LibEER (TAFFC 2025)", "20.8~44.3 (MS-MDA 44.3 최고)", "피험자 60/20/20 분할 (LOSO 아님), 미분 엔트로피 + 시행 전체 평활", "범위 비교 (분할 · 평활 명시)"],
   ["MAT (arXiv 2025)", "단일 세션 53.1~63.5, 교차 세션 41.2~58.4", "첫 세션만 / 교차 세션, 시행 전체 평활", "조건부"],
   ["CLISA · DAEST · TA2CL", "59.3~73.6", "평가 스트림 온라인 정규화, 평활, DAEST 코드는 평가 피험자 정확도 최대 epoch", "직접 비교 불가 — 우리 프로토콜로 재실행"],
   ["mdJPT (NeurIPS 2025)", "39.7~65.0", "데이터셋 하나 빼기, 대상 피험자 1/4 로 분류기 학습", "비교 불가"],
   [O("우리 (LaBraM 전체 FT)"), O("클립: 적응 없음 0.30~0.36, 중심화 0.37~0.46, 자극 시점 정렬까지 0.47~0.54"),
    O("LOSO + 학습 피험자 2명 검증, 비전이적 캘리브레이션 (클립은 평가에서 제외), 평활 없음"), O("—")]],
  [30, 44, 56, 36], hl=(6,), split=True)
B(["시행 전체에 거는 선형 동적 시스템 평활 (LDS, 비인과) 이 들어간 '창 정확도' 는 순수 창과 클립의 중간쯤이다 — 우리 창 단위는 MST · "
   "LibEER 의 무평활 수치와, 우리 클립 단위는 우리 프로토콜로 다시 돌린 기준선과 비교한다.",
   "평가 피험자 통계로 정규화하는 방법 (CLISA 계열 · DMMR · MS-MDA · 층화 정규화) 은 '캘리브레이션 블록 통계' 변형으로 다시 돌리거나, 우리 "
   "전이적 상한과 나란히 놓는다.",
   "평가 정확도로 고른 epoch (공개 코드에서 확인) 은 성능을 크게 부풀린다 — 우리는 학습 피험자 2명으로 검증함을 본문에 명시한다.  "
   "세션 수 · 피험자 수 (공개판 16명) 를 표에 항상 함께 적는다."])

# ── 9. 이름 · 학회 ──
P("9. 이름과 학회 함의", H2)
B(["<b>이름 충돌.</b>  'Calibration-Aware Fine-Tuning' 은 대규모 언어 모델의 확률 보정 논문 (Xiao 외, ICML 2025) 이 이미 쓰고, 약칭 "
   "'CAFT' 는 Concept-Aware Fine-Tuning · Concept Ablation Fine-Tuning (2025) 과 겹친다.  기계학습에서 'calibration' 은 보통 확률 보정을 "
   "뜻하므로, 방법 이름 · 약칭을 바꾸거나 초록 첫 문장에서 '새 사용자 캘리브레이션 데이터' 의 뜻을 분명히 한다.  제목 'Fine-Tune Like "
   "You Calibrate' 는 충돌이 없다 (비슷한 구조의 'Fine-Tuning is Fine, if Calibrated', NeurIPS 2024 는 확률 보정).  새 이름 후보 (예: "
   "calibration-matched fine-tuning) 는 충돌 검사가 필요하다.",
   "<b>학회.</b>  '새 방법' 으로만 내면 ICLR · NeurIPS · ICML 에서 새로움 지적 가능성이 높다 — 부품이 CLISA · 잠재 정렬 · ARM-BN 과 그대로 "
   "대응한다.  받아들여질 틀은 '배포 일관 파인튜닝 원칙 + 어떤 FM 벤치마크에도 없는 비전이적 캘리브레이션 프로토콜 + FM 3종 · 데이터셋 "
   "4~5개 (FACED 포함) 의 체계적 분석과 강한 기준선' 이다.  IJCAI 는 상대적으로 유리하나 같은 비교가 필요하고, 응용 저널 (TAFFC · JNE · "
   "TNSRE) 관례로는 이 수준의 조합도 기여로 인정된다.",
   "<b>동시 투고.</b>  ICLR 심사 지침은 마감 4개월 안에 출판된 논문 · arXiv 만 있는 논문과의 비교를 요구하지 않는다 — PACE · SCC 는 동시 "
   "연구로 처리할 수 있으나, 같은 분야 리뷰어가 볼 가능성이 높으므로 인용하고 차이를 실험으로 보이는 편이 안전하다.",
   "<b>기대 효과 크기.</b>  문헌의 추가 이득은 +0.2~3%p 로 작다.  중간점검의 +0.06 이 16명 · 여러 데이터셋에서 유지되면 그 자체로 보고 "
   "가치가 있고, 유지되지 않아도 그대로 보고한다 (분야 관례)."])

# ── 10. 참고문헌 ──
P("10. 참고문헌 (주제별, 링크) — 전체 목록은 원 보고서", H2)
REFS = [
    ("원리 · 일반 기계학습", [
        ("Vinyals et al. 2016, Matching Networks (NeurIPS)", "https://proceedings.neurips.cc/paper/2016/hash/90e1357833654983612fb05e3ec9148c-Abstract.html"),
        ("Snell et al. 2017, Prototypical Networks (NeurIPS)", "https://proceedings.neurips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html"),
        ("Finn et al. 2017, MAML (ICML)", "https://proceedings.mlr.press/v70/finn17a.html"),
        ("Li D. et al. 2018, MLDG (AAAI)", "https://doi.org/10.1609/aaai.v32i1.11596"),
        ("Sun et al. 2020, Test-Time Training (ICML)", "https://proceedings.mlr.press/v119/sun20b.html"),
        ("Goyal et al. 2023, FLYP: Finetune like you pretrain (CVPR)", "https://doi.org/10.1109/CVPR52729.2023.01853"),
        ("Zhang M. et al. 2021, Adaptive Risk Minimization, ARM-BN (NeurIPS)", "https://arxiv.org/abs/2007.02931"),
        ("Bronskill et al. 2020, TaskNorm · MetaBN (ICML)", "https://proceedings.mlr.press/v119/bronskill20a.html"),
        ("Chang et al. 2019, Domain-Specific Batch Normalization (CVPR)", "https://doi.org/10.1109/CVPR.2019.00753"),
        ("Zhuang et al. 2020, Camera-based Batch Normalization (ECCV)", "https://doi.org/10.1007/978-3-030-58610-2_9"),
        ("Li Y. et al. 2018, AdaBN (Pattern Recognition)", "https://doi.org/10.1016/j.patcog.2018.03.005"),
        ("Schneider et al. 2020, covariate shift adaptation by BN statistics (NeurIPS)", "https://arxiv.org/abs/2006.16971"),
        ("Ioffe 2017, Batch Renormalization (NeurIPS)", "https://proceedings.neurips.cc/paper/2017/hash/c54e7837e0cd0ced286cb5995327d1ab-Abstract.html"),
        ("Wu & Johnson 2021, Rethinking 'Batch' in BatchNorm (arXiv)", "https://arxiv.org/abs/2105.07576")]),
    ("음성 · 생체신호", [
        ("Anastasakos et al. 1996, Speaker-Adaptive Training (ICSLP)", "https://doi.org/10.21437/ICSLP.1996-253"),
        ("Viikki & Laurila 1998, segmental CMVN (Speech Communication)", "https://doi.org/10.1016/S0167-6393(98)00033-8"),
        ("Busso et al. 2013, Iterative Feature Normalization (IEEE TAFFC)", "https://doi.org/10.1109/T-AFFC.2013.26"),
        ("Du et al. 2017, multi-stream AdaBN for sEMG (Sensors)", "https://doi.org/10.3390/s17030458"),
        ("Mazankiewicz et al. 2020, real-time HAR personalization (Proc. ACM IMWUT)", "https://doi.org/10.1145/3432230")]),
    ("EEG 정규화 · 피험자 적응", [
        ("Fdez et al. 2021, stratified normalization (Front. Neurosci.)", "https://doi.org/10.3389/fnins.2021.626277"),
        ("Kobler et al. 2022, SPD domain-specific BN (NeurIPS)", "https://arxiv.org/abs/2206.01323"),
        ("Bakas et al. 2025, Latent alignment (J. Neural Eng.)", "https://doi.org/10.1088/1741-2552/adb336"),
        ("Kwak et al. 2023, baseline correction module (IEEE JBHI)", "https://doi.org/10.1109/JBHI.2023.3238421"),
        ("He & Wu 2020, Euclidean alignment (IEEE TBME)", "https://doi.org/10.1109/TBME.2019.2913914"),
        ("Chen H. et al. 2023, Personal-Zscore (IEEE TAFFC)", "https://doi.org/10.1109/TAFFC.2021.3137857"),
        ("Apicella et al. 2023, normalization for EEG domain adaptation (Eng. Appl. AI)", "https://doi.org/10.1016/j.engappai.2023.106205")]),
    ("같은 자극 사람 간 정렬", [
        ("Haxby et al. 2011, hyperalignment (Neuron)", "https://doi.org/10.1016/j.neuron.2011.08.026"),
        ("Turek et al. 2017, semi-supervised SRM (ICASSP)", "https://doi.org/10.1109/ICASSP.2017.7952326"),
        ("Shen X. et al. 2023, CLISA (IEEE TAFFC)", "https://doi.org/10.1109/TAFFC.2022.3164516"),
        ("Shen X. et al. 2024, CL-SSTER (NeuroImage)", "https://doi.org/10.1016/j.neuroimage.2024.120890"),
        ("Zhang Q. et al. 2025, mdJPT (NeurIPS)", "https://arxiv.org/abs/2510.22197"),
        ("Chang J. et al. 2025, MSHCL (IEEE TAFFC)", "https://doi.org/10.1109/TAFFC.2025.3535542"),
        ("Shen X. et al. 2025, DAEST (IEEE TAFFC)", "https://doi.org/10.1109/TAFFC.2025.3593630"),
        ("Xie Y. et al. 2026, TA2CL (arXiv)", "https://arxiv.org/abs/2605.22379"),
        ("Défossez et al. 2023, speech decoding from MEG/EEG (Nat. Mach. Intell.)", "https://doi.org/10.1038/s42256-023-00714-5"),
        ("Scotti et al. 2024, MindEye2 (ICML)", "https://arxiv.org/abs/2403.11207")]),
    ("EEG 파운데이션 모델 · 동시 투고", [
        ("Jiang et al. 2024, LaBraM (ICLR)", "https://arxiv.org/abs/2405.18765"),
        ("PACE (anonymous, ICLR 2027 submission)", "https://openreview.net/forum?id=9QdMLgXMOk"),
        ("SCC: Subject-Coordinate Canonicalization (anonymous, ICLR 2027 submission)", "https://openreview.net/forum?id=fwfHnohQbt"),
        ("SEAM (anonymous, ICLR 2027 submission)", "https://openreview.net/forum?id=0CFQcqmVpt"),
        ("NeuroContext-TTA (anonymous, ICLR 2027 submission)", "https://openreview.net/forum?id=KJ0mwI2BjI"),
        ("Tao & Chen 2026, personal vs population gains in EEG FM calibration (arXiv)", "https://arxiv.org/abs/2609.34801"),
        ("Lin et al. 2026, The Identity Trap (arXiv)", "https://arxiv.org/abs/2606.06647"),
        ("Yang et al. 2026, Are EEG Foundation Models Worth It? (ICLR)", "https://openreview.net/forum?id=5Xwm8e6vbh"),
        ("Lu et al. 2026, OmniEEG-Bench (arXiv)", "https://arxiv.org/abs/2606.00815")]),
    ("정렬 손실의 위험 · 처방", [
        ("Kornblith et al. 2021, why do better loss functions transfer worse (NeurIPS)", "https://papers.nips.cc/paper_files/paper/2021/hash/f0bf4a2da952528910047c31b6c2e951-Abstract.html"),
        ("Chen X. et al. 2019, Batch Spectral Penalization (ICML)", "https://proceedings.mlr.press/v97/chen19i.html"),
        ("Jing et al. 2022, dimensional collapse (ICLR)", "https://arxiv.org/abs/2110.09348"),
        ("Bardes et al. 2022, VICReg (ICLR)", "https://arxiv.org/abs/2105.04906"),
        ("Zbontar et al. 2021, Barlow Twins (ICML)", "https://proceedings.mlr.press/v139/zbontar21a.html"),
        ("Wang & Isola 2020, alignment and uniformity (ICML)", "https://proceedings.mlr.press/v119/wang20k.html"),
        ("Bordes et al. 2023, Guillotine regularization (TMLR)", "https://arxiv.org/abs/2206.13378")]),
    ("평가 · 타당성", [
        ("Gerster et al. 2026, stimulus identity drives EEG classification on FACED (bioRxiv, 동료 심사 전)", "https://doi.org/10.64898/2026.06.12.731889"),
        ("Liu H. et al. 2025, LibEER (IEEE TAFFC)", "https://doi.org/10.1109/TAFFC.2025.3605833"),
        ("Qing & Li 2026, MST (arXiv)", "https://arxiv.org/abs/2606.00884"),
        ("Zhao L.-M. et al. 2021, PPDA (AAAI)", "https://doi.org/10.1609/aaai.v35i1.16169")]),
    ("이름 충돌", [
        ("Xiao et al. 2025, Calibration-Aware Fine-Tuning for aligned LLMs (ICML)", "https://arxiv.org/abs/2505.01997"),
        ("Chen M.K. et al. 2025, Concept-Aware Fine-Tuning (arXiv)", "https://arxiv.org/abs/2506.07833"),
        ("Casademunt et al. 2025, Concept Ablation Fine-Tuning (arXiv)", "https://arxiv.org/abs/2507.16795"),
        ("Mai et al. 2024, Fine-Tuning is Fine, if Calibrated (NeurIPS)", "https://arxiv.org/abs/2409.16223")]),
]
for head, items in REFS:
    P(head, H3)
    for lab, url in items:
        P("• " + L(lab, url), REF)

# ── 11. 한계 ──
P("11. 한계", H2)
B(["ICLR 2027 투고작은 초록 (PACE 는 공개 코드 포함) 기준이고 심사 중이다 — 내용이 바뀌거나 공개되지 않을 수 있다.",
   "공개 코드로 확인한 사실 (샘플러 · 정규화 · 평가 방식) 이 각 논문의 보고 수치를 만든 경로인지는 확인하지 못했다.",
   "Kwak 2023 은 초록 기준 (본문 세부 미확인), MSHCL 은 본문 대신 공개 코드로 확인했다.  미확인 항목은 각 원 보고서 마지막 절에 있다.",
   "2026 하반기 저널 논문과 비영어권 문헌은 빠졌을 수 있다.  인용 전에 원문을 한 번 더 본다."])


# ── 글리프 검사와 출력 ────────────────────────────────────────────────────
def check_glyphs():
    cmaps = [set(FTFont(p).getBestCmap()) for p in (FONT_R, FONT_B)]
    miss = set()
    for t in TEXTS:
        plain = re.sub(r"<[^>]+>", "", t).replace("&amp;", "&").replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
        miss |= {ch for ch in plain if not ch.isspace() and all(ord(ch) not in c for c in cmaps)}
    print("폰트에 없는 글자:", sorted(miss) if miss else "없음")


def footer(c, d):
    c.saveState(); c.setFont(F, 7.5); c.setFillColor(GREY)
    c.drawString(22 * mm, 12 * mm, f"CAFT 관련 연구 조사 ({NOW})")
    c.drawRightString(A4[0] - 22 * mm, 12 * mm, str(d.page))
    c.setStrokeColor(LINE); c.setLineWidth(0.4); c.line(22 * mm, 15 * mm, A4[0] - 22 * mm, 15 * mm)
    c.restoreState()


doc = BaseDocTemplate("reports/CAFT_RELATED_WORK.pdf", pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                      topMargin=18 * mm, bottomMargin=20 * mm, title="CAFT 관련 연구 조사")
doc.addPageTemplates([PageTemplate(id="a", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)],
                                   onPage=footer)])
doc.build(E)
print("[saved] reports/CAFT_RELATED_WORK.pdf")
check_glyphs()
