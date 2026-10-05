"""SEED · SEED-V 총정리 (PDF, 2026-10-04).

조판은 make_review_pdf.py 와 같은 규칙 (Noto Sans KR TTF, A4, 22 mm 여백).
수치는 결과 파일에서 확인한 값이고, 각 절 끝에 근거 문서를 적는다.

    python make_summary_pdf.py  → reports/SUMMARY_SEED_SEEDV.pdf
"""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, Image, KeepTogether)
from doc_terms import Paragraph  # 약어를 풀어 쓴다 (찍히는 글자에만 적용)

pdfmetrics.registerFont(TTFont("NotoKR", ".fonts/NotoSansKR-regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoKR-B", ".fonts/NotoSansKR-bold.ttf"))
registerFontFamily("NotoKR", normal="NotoKR", bold="NotoKR-B", italic="NotoKR", boldItalic="NotoKR-B")
F, FB = "NotoKR", "NotoKR-B"

INK = colors.HexColor("#1a1d21")
GREY = colors.HexColor("#6f757b")
ACC = colors.HexColor("#c1553b")
LINE = colors.HexColor("#d4d9dd")
BG = colors.HexColor("#f5f7f9")
NOTE = colors.HexColor("#eef4ef")


def st(name, size=9.4, lead=14.4, color=INK, font=F, space=4, left=0):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead, textColor=color,
                          spaceAfter=space, leftIndent=left, alignment=TA_LEFT)


BODY = st("b")
BUL = st("bul", left=10)
H1 = st("h1", 16, 21, INK, FB, 4)
SUB = st("sub", 9.6, 14, GREY, space=10)
H2 = st("h2", 11.5, 16, ACC, FB, 5)
H2.keepWithNext = 1          # 제목이 쪽 끝에 홀로 남지 않게
SMALL = st("s", 8.0, 11.6, GREY, space=3)
CELL = st("c", 8.0, 11.0, INK, space=0)
CELLB = st("cb", 8.0, 11.0, INK, FB, space=0)
BOX = ParagraphStyle("box", fontName=F, fontSize=9.4, leading=14.4, textColor=INK, backColor=NOTE,
                     borderPadding=7, spaceAfter=8, leftIndent=4, rightIndent=4)

E = []


def P(t, s=BODY):
    E.append(Paragraph(t, s))


def B(items):
    for t in items:
        E.append(Paragraph("• " + t, BUL))


def T(rows, widths):
    data = [[Paragraph(c, CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, 0), BG),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK)]))
    group = [t]                   # 작은 표는 쪽을 넘기지 않게, 바로 앞 제목도 같이 묶는다
    if E and isinstance(E[-1], Paragraph) and E[-1].style is H2:
        group.insert(0, E.pop())
    E.append(KeepTogether(group))
    E.append(Spacer(1, 6))


# ── 표지 ──────────────────────────────────────────────────────────────────────
P("SEED · SEED-V 총정리", H1)
P("LaBraM 교차 피험자 감정 인식에서 캘리브레이션 보정이 무엇을 고치고 무엇을 남기는가 — 2026-10-04 기준", SUB)
P("<b>한 줄 요약.</b>  처음 보는 사람에게서 생기는 오차의 큰 부분은 세션 평균의 이동이고, 짧은 캘리브레이션의 "
  "평균만 빼도 라벨 없이 잡힌다 (두 데이터셋).  남는 것은 세션별 감정 방향의 회전이며 그 크기는 데이터셋마다 "
  "다르다.  회전이 큰 SEED-V 에서는 같은 영상의 시점 대응이라는 공짜 정보(SLA)로 감정 라벨보다 더 많이 "
  "회수하고(+0.061), 회전이 작은 SEED 에서는 회수할 것이 없다.", BOX)

# ── 0 ─────────────────────────────────────────────────────────────────────────
P("0. 설정", H2)
B(["<b>과제</b>: 처음 보는 사람의 EEG 로 감정 분류 (교차 피험자, LOSO).  LaBraM 을 fine-tuning, 4초 창 입력, 시드 3개.",
   "<b>데이터</b>: SEED — 15명, 3감정, 세 세션 모두 같은 15개 영상.  SEED-V — 16명, 5감정, 세션마다 다른 15개 영상 "
   "(LaBraM 사전학습에 없는 외부 검증용).",
   "<b>평가</b>: window (4초 창) 와 clip (영상 하나의 창을 모아 한 번 판정).  캘리브레이션 = 감정당 영상 1편의 "
   "앞 20초·40초·전체, 그 영상은 평가에서 빼고 나머지 영상으로 평가.  캘리브레이션 시간만 바뀌고 평가는 고정이다.",
   "<b>주 입력</b>: EA 없는 팔 — 테스트 데이터를 입력 정규화에 전혀 쓰지 않는 완전히 현실적인 파이프라인.",
   "<b>통계</b>: 피험자 단위 짝지은 Wilcoxon, 결과를 보기 전에 기준을 고정한 사전 등록, 다중 비교는 Holm."])

# ── 1 ─────────────────────────────────────────────────────────────────────────
P("1. 출발점 — 아무 적응도 하지 않으면", H2)
T([["", "window", "clip", "우연 (clip)", "참고: subject-dependent clip"],
   ["SEED", "0.551", "0.596", "0.333", "0.751"],
   ["SEED-V", "0.296", "0.311", "0.200", "0.522"]], [24, 22, 22, 24, 56])
P("subject-dependent 는 평가 클립 구성이 달라 (SEED 세션별 10~15번 클립, SEED-V 클립 그룹 3폴드) 직접 비교표로 "
  "쓰려면 구성을 맞춰야 한다.  SEED-V 의 window 우연 수준은 클립 길이 불균형 때문에 0.261 이다.", SMALL)

# ── 2 ─────────────────────────────────────────────────────────────────────────
P("2. 주 결과 — 도메인 중심화 (라벨 없음)", H2)
P("캘리브레이션 창들의 평균 특징을 빼기만 한다.  clip 향상 (괄호는 향상된 피험자 수):")
T([["", "감정당 20초", "40초", "클립 전체"],
   ["SEED", "+0.071 (12/15)", "+0.081 (13/15)", "<b>+0.100 (15/15)</b>"],
   ["SEED-V", "+0.057 (14/16)", "+0.058 (15/16)", "+0.056 (13/16)"]], [24, 38, 38, 48])
B(["모든 칸 p ≤ 0.006.  <b>사전학습 밖인 SEED-V 에서도 재현</b> — 현실적 EA 위 정식 판정 "
   "+0.066 / +0.074 / +0.079 (16명 중 13·14·16명).",
   "논문의 SEED 주 수치 (현실적 EA + 중심화, S13): <b>0.679 / 0.695 / 0.726</b> (총 1분 / 2분 / 약 12분)."])

# ── 3 ─────────────────────────────────────────────────────────────────────────
P("3. SEED-V 재현 — 사전 등록 기준 9개 중 8개 재현", H2)
B(["<b>재현</b>: 중심화 이득 (정식 포함), 결정 규칙만으로는 효과 없음, 단일 감정 캘리브레이션의 열위, 영상 안 "
   "시간 곡선 증가, 학습 ≫ 검증 ≈ 테스트, few-shot 2종, EA 효과 감지 불가.",
   "<b>실패</b>: 시간 가중 — SEED 에서도 가장 작던 효과라 이득으로 주장하지 않는다."])

# ── 4 ─────────────────────────────────────────────────────────────────────────
P("4. 기전 — 왜 중심화가 되고, 무엇이 남는가", H2)
T([["오차 성분", "결과", "근거"],
   ["오프셋 (평균 이동)", "중심화로 라벨 없이 해결", "2절"],
   ["입력 단계 EA", "현실적 조건에서 두 데이터셋 모두 감지 한계 아래", "기준 8"],
   ["분산", "기여 없음 (stratified normalization, Holm 0/16)", "V2"],
   ["<b>남은 오차</b>", "사용자·세션마다 다른 <b>감정 방향의 회전</b>.  학습 피험자는 세션 간 일치 "
    "(0.965 / 0.860), 처음 보는 피험자는 어긋남 (0.833 / 0.414)", "V3"],
   ["라벨 없는 적응", "pseudo-label·adaBN·T3A 모두 0.  공유된 회전을 지우면 오히려 손해 → <b>중심화가 천장</b>",
    "S10, V3"]], [32, 110, 22])

# ── 5 ─────────────────────────────────────────────────────────────────────────
P("5. 감정 라벨을 쓰면 — 캘리브레이션 영상의 자극 라벨, 추가 시청 없음", H2)
B(["<b>클립 전체를 볼 때만</b> 효과가 난다.  짧은 예산에서는 0 이거나 음수 — 영상 앞부분에는 감정이 아직 "
   "특징에 덜 실려 있다.",
   "SEED-V +0.03~+0.04 (CR 회전 +0.032, L3 prototype 혼합 +0.042), SEED +0.005~+0.015 (구현에 따라).",
   "가장 단순한 L3 가 회전(CR) 이상이고, 라벨 방법 셋이 거의 같은 정확도에 닿는다 → <b>병목은 방법이 아니라 "
   "라벨의 양</b> (감정당 1편) 으로 보인다."])

# ── 6 ─────────────────────────────────────────────────────────────────────────
P("6. SLA — 같은 영상의 시점 대응으로 회전 추정 (라벨 없음)", H2)
P("새 사용자가 캘리브레이션으로 본 영상은 학습 피험자들도 본 같은 영상이다.  같은 영상·같은 시점에서 학습 "
  "피험자들이 보인 평균 응답을 대응점으로 삼아 세션마다 직교 회전을 추정한다 (fMRI 의 hyperalignment 를 "
  "배포 시점 캘리브레이션으로).  근거: 처음 보는 피험자도 같은 시점 신호를 공유한다 (15/15, 16/16).")
T([["clip, 중심화 대비", "20초", "40초", "클립 전체"],
   ["SEED", "+0.000", "−0.002", "+0.007 (n.s.)"],
   ["SEED-V", "<b>+0.017</b> (14/16)", "<b>+0.031</b> (14/16)", "<b>+0.061 (15/16)</b>, Holm p = 0.0004"]],
  [34, 32, 32, 66])
B(["SEED-V 에서 <b>감정 라벨 없이 라벨 방법(L3)보다 +0.019 높고, 20초 예산에서도 작동</b>한다 — 중심화 다음으로 "
   "큰 이득.",
   "사전 등록 판정: <b>실패 (강등 전 부분)</b> — SEED 무효과, 그리고 SEED 20초에서 SLA+L3 가 L3 보다 유의하게 "
   "나빠 한 단계 강등.",
   "특징을 fp32 로 다시 뽑아 점검해도 결론 불변 (SEED 의 무효과는 정밀도 잡음 탓이 아니다)."])

E.append(KeepTogether([Image("figs/fig_summary.png", width=166 * mm, height=166 * mm * 881 / 2144),
                       Paragraph("<b>그림.</b>  (a) 적응 없음 → 중심화 (클립 전체 캘리브레이션).  (b, c) 중심화 위의 "
                                 "추가 이득, 캘리브레이션 시간별, 95% 부트스트랩 신뢰구간.  현실적 파이프라인 (EA 없음), "
                                 "15 / 16명 × 시드 3 × 캘리브레이션 추출 10회.  *** p &lt; 0.001, ** p &lt; 0.01 "
                                 "(Wilcoxon, SLA vs 중심화).", SMALL)]))
E.append(Spacer(1, 6))

# ── 7 ─────────────────────────────────────────────────────────────────────────
P("7. 종합", H2)
for i, t in enumerate([
        "처음 보는 사람에게서 생기는 오차의 큰 부분은 <b>세션 평균의 이동</b>이고, 짧은 캘리브레이션 평균만 빼도 "
        "라벨 없이 잡힌다 (두 데이터셋).",
        "남는 것은 <b>세션별 감정 방향의 회전</b>이며 크기가 데이터셋마다 다르다 — 테스트 세션과 학습 prototype 의 "
        "코사인 SEED 0.75, SEED-V 0.32.",
        "회전이 큰 SEED-V 에서는 <b>같은 영상의 시점 대응</b>이라는 공짜 정보로 라벨보다 더 많이 회수하고, 회전이 "
        "작은 SEED 에서는 회수할 것이 없다."], 1):
    E.append(Paragraph(f"{i}. {t}", BUL))

# ── 8 ─────────────────────────────────────────────────────────────────────────
P("8. 한계와 열린 문제", H2)
B(["표본이 15·16명이라 감지 한계가 약 +0.04~0.06 — 이보다 작은 효과는 주장할 수 없다.",
   "SLA 는 데이터셋 하나에서만 효과가 있었다 → 세 번째 데이터셋 SEED-IV 에서 재현과 '어긋남 → 이득' 예측을 "
   "사전 등록해 검증 중 (V6).",
   "subject-dependent 와 LOSO 는 평가 구성이 달라, 표로 나란히 놓으려면 구성을 맞춰야 한다."])

# ── 9 ─────────────────────────────────────────────────────────────────────────
P("9. 진행 중 — V6 (SEED-IV)", H2)
P("SEED-IV (4감정, 15명, 세션마다 다른 24개 영상, 같은 세션 안에서는 전원 같은 영상) 의 LOSO 학습을 GPU 두 장에서 "
  "병렬로 돌리고 있다 (피험자 1~8 GPU3, 9~15 GPU0).  끝나면 특징 캐시 → 어긋남 c 측정과 이득 예측 기록 → SLA "
  "판정 순서.  판정 기준과 예측 규칙은 SEED-IV 특징이 생기기 전 (14:18) 에 고정했다.")

# ── 문서 지도 ─────────────────────────────────────────────────────────────────
P("근거 문서", H2)
T([["주제", "문서 (reports/)"],
   ["SEED 최종 사다리 (현실적 EA + 중심화)", "S13_FINAL_LADDER.md, README.md"],
   ["SEED-V 재현 9개 기준", "V1_SEEDV_REPLICATION.md, REPLICATION_PLAN.md"],
   ["분산 정규화 (stratified norm)", "V2_STRATNORM.md"],
   ["세션 기하 · 남은 오차의 정체", "V3_SESSION_GEOMETRY_PREREG.md"],
   ["감정 라벨 회전 (CR) · 라벨 방법 비교", "V4_CALIB_ROTATION_PREREG.md (부록 A 포함)"],
   ["SLA 제안 · 판정 · 정밀도 점검", "V5_METHOD_PROPOSAL.md, V5_SLA_GATE1_PREREG.md"],
   ["SEED-IV 재현 (진행 중)", "V6_SEEDIV_SLA_PREREG.md"],
   ["작업 일지 · 실행 명령", "NEXT.md"]], [70, 94])


def footer(c, d):
    c.saveState(); c.setFont(F, 7.5); c.setFillColor(GREY)
    c.drawString(22 * mm, 12 * mm, "SEED · SEED-V 총정리 (2026-10-04)")
    c.drawRightString(A4[0] - 22 * mm, 12 * mm, str(d.page))
    c.setStrokeColor(LINE); c.setLineWidth(0.4)
    c.line(22 * mm, 15 * mm, A4[0] - 22 * mm, 15 * mm); c.restoreState()


doc = BaseDocTemplate("reports/SUMMARY_SEED_SEEDV.pdf", pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                      topMargin=20 * mm, bottomMargin=20 * mm, title="SEED · SEED-V 총정리")
doc.addPageTemplates([PageTemplate(id="a", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)],
                                   onPage=footer)])
doc.build(E)
print("[saved] reports/SUMMARY_SEED_SEEDV.pdf")
