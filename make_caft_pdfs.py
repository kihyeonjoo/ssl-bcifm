"""캘리브레이션 인지 파인튜닝 (CAFT) 문서 두 개 — reports/CAFT_METHOD.pdf (기술) · reports/CAFT_OVERVIEW.pdf (누구나).

2026-10-05 사용자 요청: "CAFT 에 대해서 자세히 설명해주고 이것도 method.pdf, Overview.pdf 를 만들어줘".  학습이 진행 중이라
결과 절은 '대기' 로 두고, 결과가 나오면 RESULTS 에 채워 다시 만든다.  기존 결과 수치는 results/ 에서 읽는다.
끝에 폰트에 없는 글자를 검사한다 (조합 문자 · 원문자 등).

    python make_caft_pdfs.py
"""
from __future__ import annotations

import datetime
import os
import re
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
from scipy import stats
from fontTools.ttLib import TTFont as FTFont
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
                                KeepTogether, CondPageBreak, Image)

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
SMALL = st("s", 8.0, 11.6, GREY, space=3); EQ = st("eq", 9.6, 15.0, INK, F, 5, left=14)
CELL = st("c", 7.9, 11.0, INK, space=0); CELLB = st("cb", 7.9, 11.0, INK, FB, space=0)
BOX = ParagraphStyle("box", fontName=F, fontSize=9.3, leading=14.4, textColor=INK, backColor=NOTE, borderPadding=8,
                     spaceAfter=9, leftIndent=4, rightIndent=4)
TEXTS: list = []


class Doc:
    def __init__(self):
        self.E = []

    def _p(self, t, s):
        TEXTS.append(t)
        return Paragraph(t, s)

    def P(self, t, s=BODY):
        self.E.append(self._p(t, s))

    def B(self, items, s=BUL):
        for t in items:
            self.E.append(self._p("• " + t, s))

    def gap(self, h=6):
        self.E.append(Spacer(1, h))

    def _heads(self):
        hs = []
        while self.E and isinstance(self.E[-1], Paragraph) and self.E[-1].style in (H2, H3):
            hs.insert(0, self.E.pop())
        return hs

    def T(self, rows, widths, hl=()):
        data = [[self._p(str(c), CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[w * mm for w in widths], hAlign="LEFT", repeatRows=1)
        s = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.8),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8), ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 3.5), ("BACKGROUND", (0, 0), (-1, 0), BG),
             ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
             ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK)]
        for r in hl:
            s.append(("BACKGROUND", (0, r), (-1, r), HL))
        t.setStyle(TableStyle(s))
        self.E.append(KeepTogether(self._heads() + [t])); self.gap(5)

    def fig(self, path, cap, width=166):
        w, h = PILImage.open(path).size
        self.E.append(KeepTogether(self._heads() + [Image(path, width=width * mm, height=width * mm * h / w),
                                                    self._p(cap, SMALL)])); self.gap(6)

    def build(self, out, title, foot):
        def footer(c, d):
            c.saveState(); c.setFont(F, 7.5); c.setFillColor(GREY)
            c.drawString(22 * mm, 12 * mm, foot)
            c.drawRightString(A4[0] - 22 * mm, 12 * mm, str(d.page))
            c.setStrokeColor(LINE); c.setLineWidth(0.4); c.line(22 * mm, 15 * mm, A4[0] - 22 * mm, 15 * mm)
            c.restoreState()
        doc = BaseDocTemplate(out, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=18 * mm,
                              bottomMargin=20 * mm, title=title)
        doc.addPageTemplates([PageTemplate(id="a", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width,
                                                                 doc.height)], onPage=footer)])
        doc.build(self.E)
        print(f"[saved] {out}")


# ── 기존 결과 (results/) ─────────────────────────────────────────────────
R = lambda f: np.load(f"results/{f}")
m = lambda a: float(np.asarray(a, float).mean())
DS = (("SEED", ""), ("SEED-V", "seedv_"), ("SEED-IV", "seediv_"))
CEN = {d: m(R(f"{p}sla_noea.npz")["Tfull__center__clip"]) - m(R(f"{p}calib_protocol_noea_r10.npz")["none__proto_clip"])
       for d, p in DS}
SLA = {d: m(R(f"{p}sla_noea.npz")["Tfull__sla__clip"] - R(f"{p}sla_noea.npz")["Tfull__center__clip"]) for d, p in DS}
MIS = {d: m(R(f"{p}misalignment_noea.npz")["c"]) for d, p in DS}
SG = {d: R(f"{p}session_geometry.npz") for d, p in DS[:2]}
geo = {d: (m(SG[d]["cw_train"]), m(SG[d]["cw_test"])) for d in SG}
NOW = f"{datetime.date.today():%Y-%m-%d}"
# CAFT 비교 — 세 팔 모두 caft_analyze.sh (같은 인자) 의 출력.  파일이 있는 팔만 채우고 없으면 '학습 중'.
# '적응 없음' 은 캘리브레이션 분석, 나머지는 자극 시점 정렬 분석 파일 (위 CEN · SLA 와 같은 출처).  두 분석 모두 피험자 번호 순으로 저장한다.
ARMS = (("ⓐ 일반 파인튜닝 (시드 0)", "seedv_noea_seed0"), ("ⓑ CAFT ①", "caftb_seedv_noea"), ("ⓒ CAFT ①+②", "caftc_seedv_noea"))


def load_arm(tag):
    fs = [f"results/{tag}_{x}.npz" for x in ("calib_protocol_r10", "sla", "misalignment")]
    if not all(os.path.exists(f) for f in fs):
        return None
    cp, sl, mi = (np.load(f) for f in fs)
    return {"none": cp["none__proto_clip"], "T20": sl["T20__center__clip"], "T40": sl["T40__center__clip"],
            "Tfull": sl["Tfull__center__clip"], "Tfull_sla": sl["Tfull__sla__clip"], "win": sl["Tfull__center__win"],
            "sla": sl["Tfull__sla__clip"] - sl["Tfull__center__clip"], "c": mi["c"]}


ARM = {t: load_arm(t) for _, t in ARMS}
B0 = ARM["seedv_noea_seed0"]
assert B0 is not None and len(B0["none"]) == 16
DONE = ARM["caftb_seedv_noea"] is not None and ARM["caftc_seedv_noea"] is not None


def cmp(tag, key):
    """ⓐ(시드0) 대비 (평균, 차이, 향상 수, Wilcoxon p).  tag 결과가 없으면 None."""
    A = ARM[tag]
    if A is None:
        return None
    v, a = np.asarray(A[key], float), np.asarray(B0[key], float)
    d = v - a
    try:
        p = float(stats.wilcoxon(v, a).pvalue)
    except ValueError:
        p = float("nan")
    return float(v.mean()), float(d.mean()), int((d > 0).sum()), p


def ptxt(p):
    return "p&lt;0.001" if p < 0.001 else f"p={p:.3f}"


STATUS = ("SEED-V 16 fold · 시드 0 완료 (2026-10-05 14:04~21:57, ⓑ GPU3 · ⓒ GPU0, 분석 22:02).  회귀 검사 6/6 일치." if DONE else
          "SEED-V 시험 학습 중 — 2026-10-05 14:04 시작.  결과는 나오는 대로 이 문서에 추가한다.")

# ════════════════════════════════════════════════════════════════════════
#  1) CAFT_METHOD.pdf  (기술 문서)
# ════════════════════════════════════════════════════════════════════════
D = Doc()
D.P("캘리브레이션 인지 파인튜닝 (CAFT) — 방법", TITLE)
D.P("Calibration-Aware Fine-Tuning of EEG Foundation Models: 정의 · 설정 · 평가 설계", SUBT)
D.P(f"{NOW} · 상태: {STATUS} · 약어는 처음 나올 때 풀어 쓴다", SMALL)
D.gap(4)
D.P("<b>요약.</b>  배포 때 새 사용자에게 하는 캘리브레이션 — 캘리브레이션 블록의 평균 특징을 빼는 도메인 중심화 — 을 파인튜닝 "
    "안에서 미리 흉내 낸다.  배치 하나를 '같은 세션의 학습 피험자 K = 4 명 × 같은 (클립, 시점) M = 15 곳 (감정별로 고르게)' 으로 "
    "짜서, ① 피험자마다 자기 배치 평균을 빼고 (배치 안 즉석 중심화) 감정을 분류하고, ② 같은 영상의 같은 순간을 본 사람들의 "
    "중심화된 특징이 같은 방향을 가리키도록 한다 (사람 간 자극 시점 일관성 손실).  배포는 바뀌지 않는다.  비교: ⓐ 일반 파인튜닝 "
    "(기존 시드 0 모델) · ⓑ CAFT ① · ⓒ CAFT ①+②, 같은 현실적 캘리브레이션 평가로.", BOX)

D.P("1. 동기 — 분석 결과에서 설계로", H2)
D.T([["지금까지의 분석 결과", "CAFT 의 설계 결정"],
     [f"캘리브레이션 블록 평균을 빼는 중심화가 이득의 대부분 (영상 끝까지 클립 SEED {CEN['SEED']:+.3f}, SEED-V "
      f"{CEN['SEED-V']:+.3f}, SEED-IV {CEN['SEED-IV']:+.3f})", "① 학습에서도 head 가 중심화된 특징을 보게 한다 — 학습과 배포의 표현을 같게"],
     [f"남는 오차는 처음 보는 사람의 감정 방향 회전 (같은 사람 다른 날 방향 일치도: 학습한 사람 SEED {geo['SEED'][0]:.2f} · "
      f"SEED-V {geo['SEED-V'][0]:.2f} 대 처음 보는 사람 {geo['SEED'][1]:.2f} · {geo['SEED-V'][1]:.2f}) — 라벨이나 같은 자극 같은 "
      f"대응 정보 없이는 추정하기 어렵다",
      "② 학습 단계에서 사람 간 방향을 미리 맞춘다 — 배포 때 남는 회전을 줄이는 것이 목표"],
     [f"같은 영상의 같은 순간이 회전 추정의 대응점이 된다 (자극 시점 정렬: SEED-V {SLA['SEED-V']:+.3f}, SEED-IV "
      f"{SLA['SEED-IV']:+.3f}, SEED {SLA['SEED']:+.3f}; 어긋남이 클수록 큼)", "② 의 대응 단위 = (클립, 시점) — 감정 라벨이 아니라 같은 순간"],
     ["9월에 검토한 '학습 중 중심화' 는 epoch 시작에 계산한 도메인 평균표를 썼고, 모델을 고르는 epoch 6~10 에서 그 평균이 "
      "41~55% 낡았다 → 전체 학습은 보류 (6절)", "① 의 평균은 배치 안에서 현재 모델로 즉석 계산 — 낡음 0"],
     ["한 감정만으로 캘리브레이션하면 크게 나쁘다 (감정 구성 > 길이)", "배치 위치를 감정별로 고르게 (SEED-V 5감정 × 3) — 실제 캘리브레이션 블록과 같은 구성"]],
    [82, 84])
D.fig("figs/fig_caft_concept.png", "그림 1.  CAFT 가 다루는 두 문제 (설명용 합성 점, 실제 특징 아님).  (a) 사람 · 날짜마다 특징 덩어리 "
      "전체가 밀린다 (평행 이동).  (b) ① 사람마다 자기 평균을 빼면 밀림은 사라지지만, 같은 감정의 방향이 사람마다 돌아가 있다 — 어긋남 "
      "c 가 낮다 (실선 화살표 = 사람 1 의 감정 평균, 점선 = 사람 2).  검은 화살표: ② 가 같은 클립 · 같은 초의 두 반응을 서로 당긴다.  "
      "(c) ② 뒤에는 감정 축이 사람과 무관해진다 (c 가 1 에 가깝다).")

D.P("2. 정의", H2)
D.P("2-1. 표기", H3)
D.P("f: LaBraM 인코더, z = f(x) ∈ R<super>200</super> 는 4초 창의 분류 토큰 (class token).  g: 감정 head.  s: 피험자, a: 세션 (녹화한 날), "
    "p = (k, t): 클립 k 의 t 번째 4초 창 (시점).  K: 배치당 피험자 수, M: 피험자당 위치 수.", BODY)
D.P("2-2. 배치 구성 (캘리브레이션 블록 모양)", H3)
D.B(["세션 a 를 고른다 (학습 피험자 K 명 이상이 그 세션을 가진 것 중 균등).",
     "그 세션의 학습 피험자 K 명을 비복원 추출한다.",
     "감정마다 M / C 곳 (C = 감정 수, SEED-V 는 15 / 5 = 3): 그 감정의 클립을 복원 추출하고, 시점 t 를 K 명의 그 클립 창 수 중 "
     "가장 작은 값 안에서 균등 추출한다.  클립의 감정은 SEED 계열에서 모든 피험자가 같다 (다르면 다수결).  M 이 C 로 나뉘지 않으면 "
     "남는 자리를 감정에 무작위로 준다.",
     "배치 = 피험자 우선 순서의 K × M 창 (SEED-V: 60 창).  위치 순서는 모든 피험자가 같다.",
     "epoch 당 배치 수 = 학습 창 수 / (K·M) 의 정수 몫 (SEED-V 학습 13명: 394) — 한 epoch 에 보는 창 수는 기준선과 같은 수준이지만, "
     "복원 추출이라 모든 창을 정확히 한 번씩 보지는 않는다.  난수 = 시드 × 100003 + epoch — epoch 마다 다른 배치, 재현 가능."])
D.P("2-3. ① 배치 안 즉석 중심화", H3)
D.P("u<sub>s,p</sub> = z<sub>s,p</sub> − m<sub>s</sub>,   m<sub>s</sub> = (1/M) Σ<sub>p</sub> z<sub>s,p</sub>", EQ)
D.P("m<sub>s</sub> 는 이 배치 안의 피험자 s 의 M 창 평균이다.  감정별로 고르게 뽑은 M 창이므로 배포 때의 캘리브레이션 블록 평균 "
    "μ<sub>d</sub> 와 같은 구성 · 같은 연산이다.  현재 모델로 계산하므로 낡지 않는다.  기울기는 평균을 통해서도 흐른다 (배치 "
    "정규화의 학습 모드와 같은 방식).  head 입력은 u 다: 로짓 = g(u).", BODY)
D.P("2-4. ② 사람 간 자극 시점 일관성 (조건 ⓒ)", H3)
D.P("L<sub>align</sub> = (1/M) Σ<sub>p</sub> [ 1 − (K·||c<sub>p</sub>||<super>2</super> − 1) / (K − 1) ],   "
    "c<sub>p</sub> = (1/K) Σ<sub>s</sub> u<sub>s,p</sub> / ||u<sub>s,p</sub>||", EQ)
D.P("(K·||c<sub>p</sub>||<super>2</super> − 1)/(K − 1) 은 위치 p 에 있는 K 개 단위벡터의 평균 쌍별 코사인과 정확히 같다.  즉 L<sub>align</sub> 은 "
    "'같은 영상의 같은 순간을 본 사람들의 중심화된 특징이 이루는 평균 쌍별 (1 − 코사인)' 이다.  같은 특징이면 0, 무작위면 약 1.  "
    "대조학습과 달리 음성 쌍이 없고, 붕괴는 분류 손실이 막는다.", BODY)
D.P("<b>왜 회전을 학습 단계에서 줄이나.</b>  배포 때 새 사용자는 라벨이 없어, 회전을 추정하려면 대응 정보 (라벨, 또는 같은 자극의 "
    "템플릿) 가 따로 필요하다 — 자극 시점 정렬이 이 방식이다.  학습 데이터에는 여러 사람이 같은 영상의 같은 순간을 봤다는 대응이 이미 "
    "들어 있으므로, 회전은 학습 단계에서 줄이는 편이 쉽다.  그렇게 되면 배포에는 평균 빼기만 남고 같은 자극 대응은 학습 쪽에서만 "
    "필요해진다 — 새 사용자가 학습에 쓰지 않은 영상으로 캘리브레이션해도 될 가능성이 생긴다 (학습에 없던 영상 평가로 확인할 것).", BODY)
D.P("2-5. 전체 손실", H3)
D.P("L = CE(g(u), y) + λ · L<sub>align</sub>   (CE 는 라벨 스무딩 0.1;  ⓑ: λ = 0,  ⓒ: λ = 0.5)", EQ)
D.P("2-6. 모델 구조에서 바뀌는 것 — 새 파라미터 0개", H3)
D.T([["부분", "기준선 ⓐ", "CAFT ⓑ · ⓒ"],
     ["인코더 (LaBraM-base: 시간 패치 임베딩 → 채널 · 시간 위치 임베딩 + 분류 토큰 → 트랜스포머 12층, 5,819,936 파라미터)",
      "전부 학습", "같음"],
     ["감정 head (LayerNorm → Linear 200 → GELU → Linear 5, 41,605 파라미터)", "입력 z", "입력 u — 구조 같음"],
     ["인코더와 head 사이", "없음", "① 빼기 연산 (학습: 배치 평균 m<sub>s</sub>, 배포: 캘리브레이션 평균 μ<sub>d</sub>) — 파라미터 0"],
     ["손실", "CE", "CE + λ L<sub>align</sub> (②) — 파라미터 0"],
     ["배치", "무작위 64 창", "구조 배치 60 창 (샘플러)"],
     ["배포", "캘리브레이션 평균 빼기 → prototype", "같음"]], [64, 38, 64])
D.P("<b>인코더가 바뀌는 이유 (기울기).</b>  u<sub>s,p</sub> = z<sub>s,p</sub> − m<sub>s</sub> 이므로 "
    "∂L/∂z<sub>s,p</sub> = ∂L/∂u<sub>s,p</sub> − (1/M) Σ<sub>q</sub> ∂L/∂u<sub>s,q</sub> 이고, 한 사람의 M 창에 대한 기울기의 합은 0 이다.  "
    "즉 손실은 그 사람의 모든 특징을 같은 벡터만큼 옮기는 것 (밀림, 그림 1a) 에 전혀 반응하지 않는다.  인코더는 사람 공통 성분을 감정 "
    "판단에 쓸 수 없고, 감정 정보를 그 사람 안의 상대적 차이로 표현하도록 학습된다.  ② 도 중심화된 공간의 코사인이라 같은 불변성을 "
    "가지면서, 남는 회전 (그림 1b → 1c) 을 줄이는 쪽으로 인코더를 민다.  두 장치 모두 파라미터가 없어 다른 파운데이션 모델에도 그대로 붙는다.",
    BODY)

D.fig("figs/fig_caft.png", "그림 2.  CAFT 한눈에.  위 (파인튜닝): 배치 = 같은 세션의 학습 피험자 4명 × 같은 순간 15곳 "
      "(행 = 사람, 열 = 같은 클립 · 같은 초, 색 = 감정, 감정마다 3곳).  LaBraM 의 분류 토큰 z 에서 ① 사람마다 자기 배치 평균 μ 를 "
      "빼고 (u = z − μ) head 로 분류 (CE), ② 같은 열을 본 사람들의 u 방향을 맞춘다 (L<sub>align</sub>, 조건 ⓒ).  아래 (배포, 바뀌지 "
      "않음): 새 사용자의 캘리브레이션 블록에서 μ 를 한 번 계산해 이후 모든 창에서 빼고 prototype 으로 분류한다 (선택: 그 위에 자극 "
      "시점 정렬).  점선: 학습과 배포의 ① 은 같은 연산이다.")

D.P("3. 학습 설정 — 기준선과 같은 것, 다른 것", H2)
D.T([["항목", "기준선 ⓐ (일반 파인튜닝)", "CAFT ⓑ · ⓒ"],
     ["모델", "LaBraM-base 인코더 전체 (5,819,936 파라미터) + 감정 head (41,605), 모두 학습", "같음"],
     ["최적화", "AdamW (가중치 감쇠 0.05), 층별 학습률 감쇠 0.65 (최고 5e-4), step 단위 코사인 (warmup 5 epoch), 50 epoch, "
      "기울기 자르기 3.0, drop path 0.1, bf16", "같음"],
     ["입력", "유클리드 정렬 없음, µV/100 (scale100), 4초 창", "같음"],
     ["교차검증", "피험자 하나 빼기, 검증 2명 (순환), 검증 macro-F1 로 epoch 선택", "같음"],
     ["배치", "무작위 64 창", "<b>구조 배치 60 창</b> (같은 세션 · 4명 × 같은 15곳, 감정 균형)"],
     ["head 입력", "분류 토큰 z", "<b>배치 안 중심화된 u</b>"],
     ["손실", "CE (라벨 스무딩 0.1)", "CE + λ L<sub>align</sub> (<b>ⓒ: λ = 0.5</b>, ⓑ: 0)"],
     ["검증 채점", "중심화 없음", "검증 세션 평균을 채점 직전 현재 모델로 계산해 빼고 채점 (epoch 선택에만 쓴다)"],
     ["시드 · 데이터", "시드 0, 1, 2 (비교에는 시드 0)", "시드 0, SEED-V 16 fold (시험)"]], [24, 70, 72], hl=(5, 6, 7))

D.P("4. 배포와 평가", H2)
D.B(["<b>배포는 사후 파이프라인과 같다</b>: 새 사용자의 캘리브레이션 블록 (감정당 영상 1편, 앞 20초 · 40초 · 끝까지) 평균을 빼고, "
     "학습 피험자들의 중심화된 특징으로 만든 감정 prototype 과 코사인으로 분류한다 (선택: 그 위에 자극 시점 정렬).  CAFT 의 head 는 "
     "중심화된 입력을 배웠으므로, 평가는 기존 주 지표인 prototype 경로를 쓴다.",
     "<b>절차</b>: 특징 캐시 → 현실적 캘리브레이션 분석 (예산 3개 × 반복 10, 적응 없음 · 평가 데이터를 미리 쓴 상한 · 한 감정만 포함) → "
     "중심화 위 자극 시점 정렬 · prototype 혼합 → 어긋남 c.  기존 분석 스크립트를 그대로 쓴다.",
     "<b>비교</b>: ⓑ · ⓒ 대 ⓐ (기존 일반 파인튜닝의 시드 0) — 같은 fold · 같은 시드라 피험자 16명을 짝지어 비교한다.",
     "<b>지표</b>: 주 = 중심화 뒤 클립 정확도 (영상 끝까지 · 감정당 20초).  보조 = 창 정확도, 균형 정확도, 어긋남 c, 그 위 자극 시점 "
     "정렬 이득, 적응 없음 정확도.  판단은 분야 관례대로 피험자 단위 Wilcoxon · 향상 피험자 수 · 평균 향상폭, 모든 예산 · 지표를 보고.",
     "<b>가설</b>: H1 — ⓑ 는 학습과 배포가 같아져 ⓐ 이상.  H2 — ⓒ 는 사람 간 방향이 맞춰져 c (테스트 감정 방향과 학습 prototype 의 코사인, 1 = 일치) 가 오르고, 중심화 뒤 "
     "정확도가 오르며, 그 위 자극 시점 정렬의 추가 이득은 줄어든다 (이미 맞춰져 있으므로)."])

D.P("5. 구현과 점검", H2)
D.T([["무엇", "내용"],
     ["코드", "caft.py (CAFTBatchSampler · in_batch_center · stim_align_loss), finetune_labram_hemi_aux.py 의 caft 경로 — 설정에서 "
      "꺼져 있으면 기존 로더 · 손실과 같다"],
     ["설정", "configs/caftb_seedv_noea.yaml (ⓑ), caftc_seedv_noea.yaml (ⓒ) — 기존 SEED-V 설정에 caft 블록 · 중심화 채점 · 시드 0 · 출력 경로만 더함"],
     ["구성기 점검", "SEED-V 5명으로 배치 50개: 같은 세션 · 서로 다른 4명 · 같은 (클립, 시점) · 같은 라벨 · 감정별 정확히 3개 — 50/50 통과, "
      "epoch 마다 다른 배치"],
     ["손실 점검", "중심화 뒤 피험자별 평균 ≈ 0, 정렬 손실은 같은 특징 (평균만 다름) 이면 ≈ 0 (10<super>−8</super>) · 무작위면 ≈ 1 "
      "(1.006), 기울기 유한.  2-4 의 항등식 (평균 쌍별 코사인) 은 직접 계산과 수치 오차 3×10<super>−8</super>"],
     ["스모크", "GPU1 에서 1 epoch (피험자 7명, 학습 4명) — 학습 · 체크포인트 · 결과 기록 정상 (33초)"],
     ["실행", "ⓑ GPU3, ⓒ GPU0 (2026-10-05 14:04).  fold 당 약 28분.  학습 → 각자 특징 캐시 → DEAP 남은 fold 순서로 이어진다.  "
      "분석은 CPU (caft_analyze.sh — 세 팔에 같은 인자; 기준선 ⓐ 시드 0 은 14:50 완료, 회귀 일치)"]],
    [26, 140])

D.P("6. 9월 '학습 중 중심화' 와의 차이", H2)
D.T([["", "9월 검토 (보류)", "CAFT ①"],
     ["평균 계산", "epoch 시작에 학습셋 전체를 한 번 더 통과시켜 도메인 (피험자 × 세션) 평균표", "배치 안에서 현재 모델로 즉석"],
     ["평균의 낡음", "모델을 고르는 epoch 6~10 에서 41~55% (학습 후반에는 사라짐)", "0"],
     ["평균에 쓰는 창", "그 도메인의 모든 창", "감정별로 고른 15창 — 배포의 캘리브레이션 블록과 같은 구성"],
     ["배치", "무작위", "같은 세션 · 같은 위치의 구조 배치"],
     ["추가 손실", "없음", "② 사람 간 일관성 (ⓒ)"],
     ["결과", "낡음 0 인 대리 실험 (실험 D: 백본 고정, 중심화된 특징으로 head 재학습, SEED): 기존 head + 중심화보다 클립 "
      "+0.024 (12/15) 이지만 prototype + 중심화보다 −0.017 (3/15) → end-to-end 학습 (37~47시간) 은 보류", "대기"]],
    [28, 72, 66])
D.P("실험 D 는 당시 평가 세션 전체의 평균으로 중심화한 참고 수치이고, 백본이 중심화에 맞춰 함께 움직이는 효과를 구조적으로 잘라 낸 "
    "실험이라 end-to-end 학습의 효과를 배제하지 못했다 (당시 기록의 한계).  CAFT 는 그 경로 — 백본까지 함께 학습 — 를 낡음 없이 시험한다.",
    SMALL)

D.P("7. 관련 연구와의 관계 — 정직한 위치", H2)
D.T([["선행", "공통점", "차이"],
     ["도메인별 배치 정규화 (AdaBN, 도메인별 BN, SPD 도메인별 BN)", "학습 중 도메인별 통계로 정규화", "CAFT 는 평균만, 감정 균형 · 같은 위치의 "
      "캘리브레이션 블록 구성, 배포 연산과 정확히 같다"],
     ["조건 일치형 학습: 에피소드 학습 (prototypical network, '시험처럼 학습'), 사전학습처럼 파인튜닝 (FLYP, CVPR 2023)",
      "학습 조건을 다른 단계 (시험 · 사전학습) 의 조건과 맞춘다", "맞추는 대상이 새 사용자의 캘리브레이션 연산이다 — "
      "'Fine-Tune Like You Calibrate'"],
     ["같은 자극 정렬 학습 (CLISA, CL-SSTER, mdJPT)", "같은 자극 · 같은 시각의 사람 간 쌍", "FM 파인튜닝 단계, 중심화된 공간, 음성 쌍 없는 "
      "일관성 손실, 캘리브레이션 구성과 결합"],
     ["하이퍼정렬 · 자극 시점 정렬 (배포 시점 회전)", "같은 순간의 대응으로 사람 간 방향 맞추기", "② 는 그 학습판 — 회전을 배포 전에 줄인다"]],
    [46, 52, 68])
D.P("부품 하나하나에는 선례가 있다.  기여로 주장할 수 있는 것은 '배포 캘리브레이션과 같은 구성 · 같은 연산으로 FM 파인튜닝을 바꾸고, "
    "평가 데이터를 쓰지 않는 현실적 프로토콜로 검증한 것' 이다 (효과가 확인될 경우).", SMALL)

D.P("8. 위험과 한계", H2)
D.B(["<b>배치 구성 변화 자체의 효과</b>: ⓑ 대 ⓐ 의 차이에는 '구조 배치' 와 '중심화' 가 섞인다.  필요하면 '구조 배치 + 중심화 없음' 대조를 더한다.",
     "<b>영상 외우기 (반복 자극)</b>: ② 는 같은 순간의 반응을 맞추므로 모델이 감정보다 영상을 알아보게 될 수 있다 — 학습에 없던 영상으로 "
     "평가해 확인해야 한다.",
     "<b>미조정 초매개변수</b>: λ = 0.5, K = 4, M = 15 는 검증으로 고르지 않고 조정 없이 정한 값이다.  시드 1개, 데이터셋 1개 (SEED-V) 의 시험이다.",
     "<b>배포 때 중심화 필수</b>: CAFT 의 head 는 중심화된 입력을 배운다 — 중심화 없이 쓰면 안 된다 (prototype 경로는 원래 중심화를 쓴다).",
     "<b>다른 데이터셋</b>: M 은 감정 수에 맞춰 정한다 (예: SEED 3감정 → 15, SEED-IV 4감정 → 16).  DEAP 처럼 라벨이 사람마다 다르면 감정 균형을 "
     "설계 라벨로 맞춰야 한다."])

D.P("9. 결과 — SEED-V 16 fold, 시드 0", H2)
if not DONE:
    D.T([["", "적응 없음", "중심화 20초", "중심화 끝까지", "그 위 자극 시점 정렬", "어긋남 c"],
         ["ⓐ 일반 파인튜닝 (시드 0)", "분석 중", "분석 중", "분석 중", "분석 중", "분석 중"],
         ["ⓑ CAFT ①", "학습 중", "학습 중", "학습 중", "학습 중", "학습 중"],
         ["ⓒ CAFT ①+②", "학습 중", "학습 중", "학습 중", "학습 중", "학습 중"]], [38, 24, 26, 27, 31, 20])
    D.P("학습 완료 → 특징 캐시 → 위 분석.  결과가 나오면 이 절을 채워 다시 만든다.", SMALL)
else:
    def cell(tag, key, fmt="{:.3f}", signed=False):
        if tag == "seedv_noea_seed0":
            return (("{:+.3f}" if signed else fmt)).format(m(B0[key]))
        r = cmp(tag, key)
        s = (("{:+.3f}" if signed else fmt)).format(r[0])
        return f"{s} ({r[1]:+.3f}, {r[2]}/16, {ptxt(r[3])})"
    D.P("<b>주 지표 — 중심화 뒤 클립 정확도 (prototype 경로).</b>  ⓐ 대비 (차이, 향상 피험자 수, 피험자 단위 Wilcoxon, 보정 전).  감정 "
        "5개라 우연 수준 0.200.", BODY)
    for name, tag, key, lab in [("중심화 끝까지", None, "Tfull", None), ("중심화 20초", None, "T20", None)]:
        pass
    D.T([["클립 정확도 (16명)", "ⓐ 일반 파인튜닝 (시드 0)", "ⓑ CAFT ① (대 ⓐ)", "ⓒ CAFT ①+② (대 ⓐ)"],
         ["적응 없음 (중심화 안 함)", cell("seedv_noea_seed0", "none"), cell("caftb_seedv_noea", "none"),
          cell("caftc_seedv_noea", "none")],
         ["중심화 20초", cell("seedv_noea_seed0", "T20"), cell("caftb_seedv_noea", "T20"), cell("caftc_seedv_noea", "T20")],
         ["중심화 끝까지", cell("seedv_noea_seed0", "Tfull"), cell("caftb_seedv_noea", "Tfull"),
          cell("caftc_seedv_noea", "Tfull")],
         ["그 위 자극 시점 정렬 (끝까지)", cell("seedv_noea_seed0", "Tfull_sla"), cell("caftb_seedv_noea", "Tfull_sla"),
          cell("caftc_seedv_noea", "Tfull_sla")]], [38, 30, 39, 39], hl=(3,))
    sla_a, sla_b, sla_c = m(B0["sla"]), m(ARM["caftb_seedv_noea"]["sla"]), m(ARM["caftc_seedv_noea"]["sla"])
    D.T([["보조 지표", "ⓐ (시드 0)", "ⓑ CAFT ① (대 ⓐ)", "ⓒ CAFT ①+② (대 ⓐ)"],
         ["창 정확도 (중심화 끝까지)", cell("seedv_noea_seed0", "win"), cell("caftb_seedv_noea", "win"),
          cell("caftc_seedv_noea", "win")],
         ["어긋남 c (1 = 일치)", cell("seedv_noea_seed0", "c"), cell("caftb_seedv_noea", "c"),
          cell("caftc_seedv_noea", "c")],
         ["자극 시점 정렬의 추가 이득 (끝까지)", f"{sla_a:+.3f}", f"{sla_b:+.3f}", f"{sla_c:+.3f}"]], [38, 30, 39, 39],
        hl=(2,))
    rb, rc = cmp("caftb_seedv_noea", "Tfull"), cmp("caftc_seedv_noea", "Tfull")
    cb, cc = cmp("caftb_seedv_noea", "c"), cmp("caftc_seedv_noea", "c")
    D.P(f"<b>H1 (ⓑ ≥ ⓐ) — 성립.</b>  ① 만으로 중심화 뒤 클립 정확도가 끝까지 {rb[1]:+.3f} ({rb[2]}/16, {ptxt(rb[3])}), 20초 "
        f"{cmp('caftb_seedv_noea', 'T20')[1]:+.3f} ({cmp('caftb_seedv_noea', 'T20')[2]}/16, {ptxt(cmp('caftb_seedv_noea', 'T20')[3])}) "
        f"올랐다.  창 정확도 · 균형 정확도 · 그 위 자극 시점 정렬도 같은 방향으로 유의하다.  배치 안 평균 빼기 하나를 학습에서 흉내 낸 "
        f"것만으로 기준선을 넘는다.", BODY)
    D.P(f"<b>H2 (ⓒ 는 회전을 줄인다) — 방향은 뚜렷, 정확도 이득은 ⓑ 이하.</b>  ⓒ 는 어긋남 c 를 {cc[1]:+.3f} 올려 16명 전원 "
        f"({cc[2]}/16, {ptxt(cc[3])}) 에서 커졌다 — ② 가 사람 간 방향을 맞춘다는 설계가 확인된다.  예측대로 그 위 자극 시점 정렬의 추가 "
        f"이득은 ⓐ {sla_a:+.3f} · ⓑ {sla_b:+.3f} 에서 ⓒ {sla_c:+.3f} 로 줄었다 (회전 보정 몫을 학습이 가져감).  그러나 중심화 뒤 클립 "
        f"정확도는 ⓒ {rc[1]:+.3f} ({rc[2]}/16) 로 ⓑ ({rb[1]:+.3f}) 를 넘지 못했고, 중심화를 안 한 경로는 오히려 {cmp('caftc_seedv_noea', 'none')[1]:+.3f} "
        f"다.  방향은 맞췄지만 감정 판별력이 조금 깎인 것으로 보인다 (λ = 0.5 가 강했을 가능성 — 7절 · 관련 연구 조사의 ② 개선 단서).", BODY)
    D.P("ⓐ 는 CAFT 와 같은 스크립트 (caft_analyze.sh) 로 기존 일반 파인튜닝의 시드 0 모델만 다시 분석한 값이다 (다른 문서의 시드 3개 "
        "평균과 조금 다름).  회귀 검사: 중심화 값이 같은 캐시 기준과 피험자별로 같다 (최대 차이 0).  세 팔 모두 피험자 번호 순 저장이라 "
        "짝짓기가 유효하다.  시드 1개 · SEED-V 하나의 시험이다 — 시드 3개 · 다른 데이터셋 · 다른 FM · 처음 보는 영상 평가는 다음 단계.", SMALL)
D.build("reports/CAFT_METHOD.pdf", "캘리브레이션 인지 파인튜닝 (CAFT) — 방법", f"CAFT 방법 ({NOW})")

# ════════════════════════════════════════════════════════════════════════
#  2) CAFT_OVERVIEW.pdf  (누구나)
# ════════════════════════════════════════════════════════════════════════
O = Doc()
O.P("캘리브레이션 인지 파인튜닝 (CAFT)", TITLE)
O.P("실전처럼 연습시키기 — 새 사용자에게 맞추는 단계를 모델 학습 안에 넣는다", SUBT)
O.P(f"{NOW} · 쉽게 읽는 설명 · 상태: {STATUS}", SMALL)
O.gap(4)
O.P("<b>한 줄 요약.</b>  지금까지는 모델을 다 만든 뒤 새 사용자에게 맞추는 방법 (캘리브레이션) 을 연구했다.  CAFT 는 그 맞추는 "
    "과정을 <b>모델을 학습할 때부터 미리 연습</b>시킨다.  학습 데이터를 '새 사용자의 캘리브레이션' 과 똑같은 모양으로 묶어서, "
    "① 그 묶음 안에서 사람마다 평균을 빼고 감정을 맞히게 하고, ② 같은 영상의 같은 장면을 본 사람들의 반응 방향을 서로 맞춘다.", BOX)

O.P("1. 지금까지 알게 된 것", H2)
O.B([f"<b>평균 하나를 빼는 것이 핵심이다.</b>  새 사용자가 처음에 잠깐 본 영상의 평균 뇌파 특징을 빼기만 해도 정확도가 크게 오른다 "
     f"(SEED {CEN['SEED']:+.3f}, SEED-V {CEN['SEED-V']:+.3f}, SEED-IV {CEN['SEED-IV']:+.3f}).",
     "<b>남는 문제는 '방향' 이다.</b>  평균을 빼면 원점은 맞지만, 감정을 가리키는 방향이 사람마다 조금씩 돌아가 있다.  새 사용자의 뇌파만 "
     "보고는 이것을 고칠 수 없다.",
     f"<b>같은 장면이 실마리다.</b>  새 사용자와 학습한 사람들이 같은 영상의 같은 순간을 봤다는 정보로 방향을 맞추면 일부 되찾는다 "
     f"(SEED-V {SLA['SEED-V']:+.3f}, SEED-IV {SLA['SEED-IV']:+.3f}).  방향이 많이 어긋난 데이터셋일수록 효과가 컸다."])

O.P("2. 무엇이 문제인가", H2)
O.P("지금의 학습 (일반 파인튜닝) 은 '나중에 새 사용자에게 평균을 빼서 맞출 것' 을 전혀 모른 채 진행된다.  시험장에서만 쓰는 도구를 "
    "연습 때는 한 번도 안 써 본 셈이다.  그래서 학습한 표현이 그 도구와 잘 맞는다는 보장이 없고, 사람마다 다른 방향도 학습 중에는 "
    "줄여 주지 않는다.", BODY)
O.P("사람마다 뇌파가 다른 방식은 크게 두 가지다 (그림 1).  <b>통째로 밀림</b> — 저울마다 영점이 다른 것처럼 특징 전체가 한쪽으로 "
    "밀린다.  평균을 빼면 대부분 사라진다.  <b>방향이 돌아감</b> — 영점을 맞춰도 같은 감정을 가리키는 방향이 사람마다 기울어 있다.  "
    "지금까지 남는 오차의 핵심이 이것이었고, 새 사용자의 뇌파만 보고는 고치기 어렵다.", BODY)
O.fig("figs/fig_caft_concept.png", "그림 1.  사람마다 다른 두 가지 (설명용 점).  (a) 사람 · 날짜마다 특징 전체가 통째로 밀린다.  "
      "(b) 사람마다 평균을 빼면 밀림은 사라지지만, 같은 감정의 방향이 사람마다 기울어 있다 (실선 = 사람 1, 점선 = 사람 2).  "
      "(c) CAFT 의 ② 는 같은 장면을 본 사람들의 반응을 서로 당겨 방향을 맞춘다.")

O.P("3. 아이디어 — 실전처럼 연습하기", H2)
O.fig("figs/fig_caft.png", "그림 2.  CAFT 한눈에.  위: 학습 — 4명 × 같은 순간 15곳의 묶음 (색 = 감정) 에서 ① 사람마다 평균을 빼고 "
      "감정을 맞히고, ② 같은 순간 (초록 칸) 을 본 사람들의 방향을 맞춘다.  아래: 새 사용자에게 쓸 때 — 캘리브레이션 블록의 평균을 빼고 "
      "분류한다.  점선: 두 줄의 ① 은 같은 계산이다.")
O.B(["<b>연습 문제를 실전 형식으로.</b>  학습 데이터를 한 번에 '같은 날 녹화한 학습 피험자 4명이, 같은 영상의 같은 순간 15곳 (감정마다 "
     "3곳) 을 본 조각' 으로 묶는다.  이 묶음은 새 사용자의 캘리브레이션 블록과 모양이 같다.",
     "<b>① 묶음 안에서 사람마다 평균을 빼고 감정을 맞힌다.</b>  배포 때 하는 것과 똑같은 계산을 학습에서도 한다.  그러면 모델은 "
     "사람마다 통째로 밀리는 부분을 감정 판단에 쓸 수 없게 되어, '그 사람 안에서의 차이' 로 감정을 표현하도록 바뀐다 (그림 1 의 a → b).  "
     "평균은 매번 지금의 "
     "모델로 바로 계산한다 — 9월에 비슷한 시도를 보류한 이유 중 하나 (미리 계산해 둔 평균이 학습 도중 낡음) 를 없앴다.",
     "<b>② 같은 장면을 본 사람들의 반응 방향을 맞춘다.</b>  같은 영상의 같은 순간에 대한 4명의 (평균을 뺀) 반응이 같은 방향을 "
     "가리키도록 학습한다.  새 사용자에게 남던 '방향 어긋남' 을 학습 단계에서 미리 줄이려는 것이다 (그림 1 의 b → c).  "
     "왜 학습 때 하나 — 새 사용자에게 쓸 때는 정답 (감정 라벨) 이 없어 방향을 고치려면 '같은 영상의 같은 순간' 같은 짝 정보가 따로 "
     "필요하다.  학습 데이터에는 여러 사람이 같은 장면을 봤다는 짝 정보가 이미 들어 있으니 그때 맞춰 두는 편이 쉽다.  그러면 쓸 때는 "
     "평균 빼기 하나만 남고, 학습에 쓰지 않은 영상으로 캘리브레이션해도 될 가능성이 생긴다 (확인 필요).",
     "<b>새 사용자에게 쓸 때는 바뀌는 것이 없다.</b>  지금처럼 캘리브레이션 블록의 평균을 빼고 분류한다 — 다만 모델이 그 상황을 연습해 두었다.",
     "<b>모델의 뼈대는 그대로다.</b>  모델 구조나 크기는 바꾸지 않는다.  바꾸는 것은 학습 데이터를 묶는 방식, 학습 때 인코더와 분류기 "
     "사이에 넣는 '평균 빼기', 보조 목표 하나뿐이라 새로 배울 부품이 없다 — 그래서 다른 파운데이션 모델에도 그대로 붙일 수 있다."])

O.P("4. 무엇을 비교하나", H2)
O.T([["", "학습 방법", "역할"],
     ["ⓐ", "일반 파인튜닝 (이미 있는 모델)", "기준"],
     ["ⓑ", "CAFT ① (묶음 안 평균 빼기)", "실전 형식으로 연습한 효과"],
     ["ⓒ", "CAFT ① + ② (+ 같은 장면 방향 맞추기)", "방향 맞추기의 추가 효과"]], [12, 80, 74])
O.P("데이터는 SEED-V (16명, 감정 5개 — 방향 어긋남이 가장 커서 효과가 있다면 가장 잘 보일 곳), 모델 학습 설정은 셋 다 같다 (배치 구성과 "
    "손실만 다름).  평가는 지금까지와 똑같이 평가 데이터를 전혀 미리 쓰지 않는 현실적 캘리브레이션으로 한다.", BODY)

O.P("5. 무엇을 보면 성공인가", H2)
O.B(["<b>정확도</b>: 새 사용자에게 평균을 뺀 뒤의 클립 정확도가 ⓑ · ⓒ 에서 ⓐ 보다 높다 (16명을 짝지어 비교, 몇 명이 오르는지).",
     "<b>방향</b>: 새 사용자의 감정 방향이 학습 기준과 더 잘 맞는다 (c 가 1 에 가까워진다 — 1 = 완전 일치).",
     "<b>남는 일</b>: 이미 방향이 맞춰졌다면, 그 위에 '같은 장면 짝짓기' 를 더했을 때의 추가 이득은 줄어야 한다."])
O.P(f"<b>넘어야 할 기준</b> (ⓐ, 같은 시드): 평균을 뺀 뒤 클립 정확도 {m(B0['Tfull']):.3f} (영상 끝까지) · {m(B0['T20']):.3f} "
    f"(감정당 20초), 평균을 빼기 전 {m(B0['none']):.3f}, 방향 일치 c {m(B0['c']):.3f}.  감정이 5개라 아무렇게나 찍으면 0.200 이다.", BODY)
if DONE:
    rb, rc = cmp("caftb_seedv_noea", "Tfull"), cmp("caftc_seedv_noea", "Tfull")
    cc = cmp("caftc_seedv_noea", "c")
    O.P(f"<b>결과 (SEED-V 16명, 시드 0).</b>  <b>① 만 (ⓑ) 이 성공이다</b>: 평균을 뺀 뒤 클립 정확도가 끝까지 {m(ARM['caftb_seedv_noea']['Tfull']):.3f} "
        f"({rb[1]:+.3f}, 16명 중 {rb[2]}명 향상), 20초 {m(ARM['caftb_seedv_noea']['T20']):.3f} "
        f"({cmp('caftb_seedv_noea', 'T20')[1]:+.3f}, {cmp('caftb_seedv_noea', 'T20')[2]}명) 로 기준선보다 높다.  <b>방향 맞추기 (ⓒ) 는 "
        f"방향 (c {m(ARM['caftc_seedv_noea']['c']):.2f}, {cc[2]}명 모두 상승) 은 크게 좋아지고, 예측대로 '같은 장면 짝짓기' 의 추가 이득은 "
        f"줄었지만</b> ({m(B0['sla']):+.3f} → {m(ARM['caftc_seedv_noea']['sla']):+.3f}), 정확도 ({rc[1]:+.3f}) 는 ① 만보다 높지 않다.  "
        f"즉 지금은 ① 이 주인공이고, ② 는 '원리는 맞지만 더 다듬어야 할 부분' 이다.", BOX)

O.P("6. 위험 — 미리 알아 둘 것", H2)
O.B(["학습 데이터를 묶는 방식이 바뀐 것만으로도 결과가 달라질 수 있다 — 필요하면 그 몫만 따로 떼어 보는 실험을 더한다.",
     "② 는 모델이 감정이 아니라 '영상 자체' 를 외우게 할 수 있다 — 학습에 없던 영상으로 평가해 확인해야 한다.",
     "이번 시험은 데이터셋 1개, 시드 1개, 조정 없이 정한 설정값 (λ = 0.5 등) 이다 — 효과가 보이면 넓혀서 확인한다."])

# 2026-10-05 사용자 요청 ("이걸 overview 에 작성해도 될까?"): 방법론 기여 판단과 투고 목표를 넣는다.  결과 전이므로 '계획' 으로
# 명시하고, BK21 같은 내부 행정 사항은 누구나 읽는 문서라 뺀다.
O.P("7. 무엇이 새로운가", H2)
O.P("① 만 (ⓑ) 의 결과 (5절) 가 긍정적으로 나왔으므로, 아래 포지셔닝이 뒷받침된다 (16명 · 시드 1개 · SEED-V 하나의 1차 결과).", BODY)
O.B(["<b>실전과 같은 조건으로 연습시키는 파인튜닝.</b>  학습 묶음을 새 사용자의 캘리브레이션과 같은 모양으로 짜고, 같은 계산 "
     "(평균 빼기) 을 학습 안에서 한다.  새 사용자에게 쓸 때 드는 비용은 지금과 같다.",
     "<b>사람마다 다른 '방향' 을 학습 단계에서 미리 맞춘다.</b>  지금은 쓸 때 따로 하던 '같은 장면 짝짓기' 를 학습으로 옮긴다 — "
     "성공하면 새 사용자에게는 평균 빼기 하나로 충분해진다.",
     "<b>분석에서 나온 설계.</b>  '남는 오차는 사람마다 다른 방향' 이라는 진단에서 방법이 나왔고, 결과에서도 그 방향이 실제로 "
     "맞춰졌는지 (c) 까지 확인한다 — 왜 되는지를 보일 수 있다."])
O.P("<b>솔직한 위치.</b>  부품 하나하나에는 다른 분야의 선례가 있다 — 음성 인식에서 화자마다 정규화한 공간으로 학습하는 방법, "
    "같은 자극을 본 사람들의 반응을 맞추는 방법 등.  새로운 것은 이를 뇌파 파운데이션 모델의 파인튜닝에 맞게 묶은 방식과 그 근거 · "
    "검증이다.  그래서 기여의 크기는 '중간' 으로 본다.  그렇게 인정받으려면 다음을 보여야 한다.", BODY)
O.B(["평균 빼기를 이미 한 기존 모델보다도 낫다 — <b>① 만 (ⓑ) 에서 확인됨</b> (5절).",
     "부품마다 몫을 따로 보인다 (묶는 방식만 · ① 만 · ① + ②).",
     "학습에 없던 영상에서도 된다 — 영상을 외운 것이 아님을 보인다 (아직 안 함).",
     "다른 파운데이션 모델 · 다른 데이터셋 · 여러 시드에서도 된다 (아직 안 함)."])

O.P("8. 일정과 다음 단계", H2)
O.T([["언제", "무엇"],
     ["완료 (10-05)", "두 조건 (ⓑ · ⓒ) SEED-V 16 fold 학습 · 분석 — ① 만 (ⓑ) 이 기준선을 넘음 (5절)"],
     ["다음", "처음 보는 영상 평가 (SEED-V 교차 세션), 시드 3개, SEED-IV · DEAP · SEED, 다른 FM (CBraMod · REVE), ② 다듬기 (λ 낮추기 등)"],
     ["논문 (계획)", "가제 후보 'Fine-Tune Like You Calibrate: Calibration-Aware Fine-Tuning of EEG Foundation Models' — '사전학습처럼 "
      "파인튜닝하라' 는 기존 방법 이름 (FLYP) 에 빗댄 것 (또는 'Calibration-Aware Fine-Tuning of EEG Foundation Models for New "
      "Users').  지금까지의 분석은 동기와 진단, 기존 결과는 비교 기준이 된다"],
     ["투고 목표 (계획)", "2027년 1월 IJCAI 또는 ICML (인공지능 · 기계학습 최상위 학회) → 범위가 넓고 결과가 강하면 5월 NeurIPS.  "
      "저널은 IEEE TAFFC (감성 컴퓨팅 분야 대표 저널).  결과가 약하면 지금까지의 분석을 중심으로 저널에 낸다.  마감은 예년 일정 기준"]],
    [34, 132])

O.P("9. 용어 풀이", H2)
O.T([["용어", "뜻"],
     ["파운데이션 모델", "아주 많은 사람의 뇌파로 미리 학습해 둔 큰 모델 (여기서는 LaBraM) — 감정 인식에 맞게 파인튜닝해서 쓴다"],
     ["인코더 · 분류기", "인코더는 뇌파를 특징으로 바꾸는 모델의 본체, 분류기 (head) 는 그 특징을 보고 감정을 고르는 작은 마지막 층"],
     ["파인튜닝", "미리 학습된 큰 모델을 감정 인식에 맞게 다른 사람들의 데이터로 추가 학습하는 것"],
     ["캘리브레이션 (블록)", "새 사용자가 처음에 감정마다 영상 한 편씩 짧게 보며 녹화한 것.  그 평균으로 모델을 그 사람에게 맞춘다"],
     ["중심화 (평균 빼기)", "캘리브레이션 블록의 평균 특징을 모든 특징에서 빼서 사람 · 날짜에 따른 통째 밀림을 없애는 것"],
     ["CAFT", "캘리브레이션 인지 파인튜닝 — 캘리브레이션 상황을 학습 중에 미리 흉내 내는 파인튜닝"],
     ["묶음 안 평균 빼기 (①)", "학습 묶음 안에서 사람마다 자기 조각들의 평균을 빼는 것 — 배포 때의 중심화와 같은 계산"],
     ["방향 맞추기 (②)", "같은 영상의 같은 순간을 본 사람들의 (평균을 뺀) 반응이 같은 방향을 가리키게 하는 학습 목표"],
     ["어긋남 c", "새 사용자의 감정 방향이 학습 기준과 얼마나 일치하는지 (1 = 완전 일치)"],
     ["자극 시점 정렬", "배포 때 같은 순간의 반응을 짝지어 방향을 되돌리는 기존 방법 — ② 는 이것의 학습판"]], [40, 126])
O.build("reports/CAFT_OVERVIEW.pdf", "캘리브레이션 인지 파인튜닝 (CAFT) — 쉽게 읽는 설명", f"CAFT 쉽게 읽는 설명 ({NOW})")


# ── 글리프 검사 (두 문서 모두) ─────────────────────────────────────────────
cmaps = [set(FTFont(p).getBestCmap()) for p in (FONT_R, FONT_B)]
miss = set()
for t in TEXTS:
    plain = re.sub(r"<[^>]+>", "", t).replace("&amp;", "&").replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    miss |= {ch for ch in plain if not ch.isspace() and all(ord(ch) not in c for c in cmaps)}
print("폰트에 없는 글자:", sorted(miss) if miss else "없음")
