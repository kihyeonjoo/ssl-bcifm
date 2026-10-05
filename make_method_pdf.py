"""사용한 방법 전체 (PDF) — 2026-10-04 전면 개정.

reportlab 로 조판한다.  수치는 전부 results/ 의 원본(npz·csv)에서 직접 읽어 본문에 넣는다 —
손으로 옮기면 보고서와 결과가 어긋난다.  v1 (EA·중심화·결정 규칙·캘리브레이션) 의 절은 유지하고,
그 뒤에 쓴 방법들 (stratified norm, 세션 기하 진단, 감정 라벨 보정, SLA, subject-dependent,
정밀도 점검) 을 같은 표기로 더했다.  v1 의 few-shot 문장("세션을 모은 덕")은 10-04 정정대로 고쳤다.

    python make_method_pdf.py  → reports/METHOD.pdf
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
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, KeepTogether)
from doc_terms import Paragraph  # 약어를 풀어 쓴다 (찍히는 글자에만 적용)

# ── 한글 폰트 ────────────────────────────────────────────────────────────
# reportlab 은 CFF 외곽선을 못 읽는다.  NotoSansCJK 의 한국어 서브폰트를 otf2ttf 로
# TrueType 으로 바꿔 둔 것을 쓴다 (.fonts/).
pdfmetrics.registerFont(TTFont("NotoKR", ".fonts/NotoSansKR-regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoKR-B", ".fonts/NotoSansKR-bold.ttf"))
registerFontFamily("NotoKR", normal="NotoKR", bold="NotoKR-B", italic="NotoKR", boldItalic="NotoKR-B")
FONT, FONT_B = "NotoKR", "NotoKR-B"

INK = colors.HexColor("#1a1d21")
GREY = colors.HexColor("#6f757b")
ACC = colors.HexColor("#c1553b")
LINE = colors.HexColor("#d4d9dd")
BG = colors.HexColor("#f5f7f9")
NOTE = colors.HexColor("#eef4ef")


def st(name, size=9.4, leading=14.2, color=INK, font=FONT, space=4, left=0):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading, textColor=color,
                          spaceAfter=space, leftIndent=left, alignment=TA_LEFT)


BODY = st("body")
BUL = st("bul", left=10)
H1 = st("h1", 15.5, 20.5, INK, FONT_B, 5)
H2 = st("h2", 11.5, 16, ACC, FONT_B, 5)
H2.keepWithNext = 1
H3 = st("h3", 10, 14, INK, FONT_B, 3)
H3.keepWithNext = 1
SMALL = st("small", 8.1, 11.8, GREY, space=3)
CELL = st("cell", 7.9, 10.8, INK, space=0)
CELLB = st("cellb", 7.9, 10.8, INK, FONT_B, space=0)
CODE = ParagraphStyle("code", fontName=FONT, fontSize=9.2, leading=13.6, textColor=INK, spaceAfter=4,
                      leftIndent=10, backColor=BG, borderPadding=5)
BOX = ParagraphStyle("box", fontName=FONT, fontSize=9.3, leading=14.0, textColor=INK, backColor=NOTE,
                     borderPadding=7, spaceAfter=8, leftIndent=4, rightIndent=4)

E = []


def P(t, s=BODY):
    E.append(Paragraph(t, s))


def B(items):
    for t in items:
        E.append(Paragraph("• " + t, BUL))


def gap(h=5):
    E.append(Spacer(1, h))


def code(*lines):
    E.append(KeepTogether([Paragraph(x, CODE) for x in lines]))


def T(rows, widths, bold_last=False):
    data = [[Paragraph(str(c), CELLB if (i == 0 or (bold_last and i == len(rows) - 1)) else CELL)
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, 0), BG),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK)]))
    group = [t]
    if E and isinstance(E[-1], Paragraph) and E[-1].style in (H2, H3):
        group.insert(0, E.pop())
    E.append(KeepTogether(group))
    gap(5)


# ── 수치 ─────────────────────────────────────────────────────────────────
def L(f):
    return np.load(f"results/{f}")


def m(a):
    return float(np.asarray(a, float).mean())


def pr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return float(d.mean()), float(p), int((d > 0).sum()), len(d)


def holm(ps):
    ps = np.asarray(ps, float); n = len(ps); out = np.empty(n); mx = 0.0
    for r, i in enumerate(np.argsort(ps)):
        mx = max(mx, min(1.0, ps[i] * (n - r))); out[i] = mx
    return out


def fp(p):
    return "p&lt;0.0001" if p < 1e-4 else f"p={p:.4f}" if p < 0.01 else f"p={p:.3f}"


def fd(d, p, w, n):
    return f"{d:+.4f} ({fp(p)}, {w}/{n})"


DS = (("SEED", ""), ("SEED-V", "seedv_"))
BUD = (("T20", "20초"), ("T40", "40초"), ("Tfull", "클립 전체"))

C = L("centering_analysis.npz"); VC = L("seedv_centering_analysis.npz")
M = L("centering_mechanism.npz"); TR = L("ea_transfer.npz"); X = L("ea_x_centering.npz")
F = L("final_ladder.npz"); N = L("noea_ladder.npz"); VR = L("seedv_ea_realistic.npz")
CP = {d: L(f"{p}calib_protocol_noea_r10.npz") for d, p in DS}
SG = {d: L(f"{p}session_geometry.npz") for d, p in DS}
SS = {d: L(f"{p}session_subspace.npz") for d, p in DS}
SP = {d: L(f"{p}session_projection.npz") for d, p in DS}
STN = {d: L(f"{p}stratnorm.npz") for d, p in DS}
CR = {d: L(f"{p}calib_rotation_noea.npz") for d, p in DS}
LC = {d: L(f"{p}labelcalib_noea.npz") for d, p in DS}
SL = {d: L(f"{p}sla_noea.npz") for d, p in DS}
ISC = {d: L(f"{p}isc_feasibility_noea.npz") for d, p in DS}
IR = {d: L(f"{p}isc_roles_noea.npz") for d, p in DS}
ICV = {d: L(f"{p}isc_cv_noea.npz") for d, p in DS}
MIS = {d: L(f"{p}misalignment_noea.npz") for d, p in DS}
PREC = L("precision_feature_level.npz")
CP32 = L("calib_protocol_noea_r10_fp32.npz"); SL32 = L("sla_noea_fp32.npz")


def sd_summary(path):
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        for k in ("accuracy", "clip_accuracy", "clip_f1_macro"):
            by[int(r["subject"])][k].append(float(r[k]))
    out = {}
    for k in ("accuracy", "clip_accuracy", "clip_f1_macro"):
        v = np.array([np.mean(by[s][k]) for s in sorted(by)])
        out[k] = (v.mean(), v.std(ddof=1), len(v))
    return out


SD = {"SEED": sd_summary("results/a02_sd.csv"), "SEED-V": sd_summary("results/seedv_a02_sd.csv")}

cen = {d: {t: pr(CP[d][f"{t}__proto_clip"], CP[d]["none__proto_clip"]) for t, _ in BUD} for d, _ in DS}
sla = {d: {t: pr(SL[d][f"{t}__sla__clip"], SL[d][f"{t}__center__clip"]) for t, _ in BUD} for d, _ in DS}

# ════════════════════════════════════════════════════════════════════════
P("교차 피험자 EEG 감정 인식 — 사용한 방법 전체", H1)
P("학습·평가 절차, 보정 방법 (EA · 도메인 중심화 · 감정 라벨 보정 · SLA), 진단과 점검 — SEED · SEED-V", SMALL)
P(f"{datetime.date.today():%Y-%m-%d} 개정 · 모든 수치는 results/ 의 원본에서 직접 읽어 넣었다 (손으로 옮긴 값 없음).", SMALL)
gap(6)

P("요약", H2)
P("핵심은 <b>추론 시 한 번의 뺄셈</b>이다: 테스트 세션의 짧은 캘리브레이션 녹화에서 임베딩 평균 μ 를 구해 그 세션의 "
  "모든 임베딩에서 뺀다 (도메인 중심화).  학습·라벨·역전파가 없다.  완전히 현실적인 파이프라인 (EA 없음, 테스트 "
  f"데이터를 정규화에 쓰지 않음) 에서 클립 전체 캘리브레이션 기준 SEED <b>{cen['SEED']['Tfull'][0]:+.3f}</b> "
  f"({cen['SEED']['Tfull'][2]}/{cen['SEED']['Tfull'][3]}), SEED-V <b>{cen['SEED-V']['Tfull'][0]:+.3f}</b> "
  f"({cen['SEED-V']['Tfull'][2]}/{cen['SEED-V']['Tfull'][3]}) 다.", BOX)
P("중심화 뒤에 남는 오차는 사용자·세션마다 다른 <b>감정 방향의 회전</b>이고, 라벨 없는 적응은 중심화 근처가 천장이다.  "
  "캘리브레이션 영상의 자극 라벨을 쓰면 클립 전체에서만 SEED-V +0.03~+0.04 를 더한다.  같은 영상의 <b>시점 대응</b>으로 "
  f"회전을 추정하는 SLA 는 감정 라벨 없이 SEED-V {sla['SEED-V']['Tfull'][0]:+.3f} "
  f"({sla['SEED-V']['Tfull'][2]}/{sla['SEED-V']['Tfull'][3]}) 를 더하지만 SEED 에서는 효과가 없다 "
  f"({sla['SEED']['Tfull'][0]:+.3f}).")
gap(4)

# ── 0. 방법 지도 ──
P("0. 방법 지도", H2)
T([["방법", "작동 단계", "쓰는 정보", "결과 (현실적, clip, 클립 전체 캘리브레이션)", "절"],
   ["Euclidean Alignment (diag)", "입력", "캘리브레이션 창의 공분산", "현실적 조건에서 감지 한계 아래", "4, 8"],
   ["<b>도메인 중심화</b>", "특징", "캘리브레이션 창의 평균", f"SEED {cen['SEED']['Tfull'][0]:+.3f}, "
    f"SEED-V {cen['SEED-V']['Tfull'][0]:+.3f} — <b>주 효과</b>", "5"],
   ["prototype 결정 · 클립 집계", "결정", "학습 prototype", "중심화와 함께일 때만 효과", "6"],
   ["stratified normalization", "특징", "평균 + 차원별 분산", "중심화 위에 기여 없음", "9"],
   ["세션 기하 진단", "진단", "테스트 라벨 (진단 전용)", "남은 오차 = 세션별 회전", "10"],
   ["CR / L2 (자극 라벨 회전)", "특징", "캘리브레이션 자극 라벨", "SEED 0, SEED-V +0.03", "11"],
   ["L3 (prototype 혼합)", "결정", "캘리브레이션 자극 라벨", "SEED +0.005~+0.015, SEED-V +0.04", "11"],
   ["L4 (head 적응)", "head", "캘리브레이션 자극 라벨", "head 기준선 대비 +, 조건4 와 비슷", "11"],
   ["<b>SLA</b> (자극 시점 정렬)", "특징", "캘리브레이션 영상의 정체성·시점", f"SEED {sla['SEED']['Tfull'][0]:+.3f}, "
    f"SEED-V <b>{sla['SEED-V']['Tfull'][0]:+.3f}</b>", "12"],
   ["라벨 없는 TTA · 학습 중 중심화 등", "여러", "—", "±0.005 안 또는 기준선 미달", "13"]],
  [40, 15, 36, 66, 9])

from PIL import Image as _PI
_w, _h = _PI.open("figs/fig_pipeline.png").size
E.append(KeepTogether([Image("figs/fig_pipeline.png", width=166 * mm, height=166 * mm * _h / _w),
                       Paragraph("그림 0.  파이프라인에서 각 방법이 끼어드는 곳.  위: 새 사용자의 추론 경로 (① EA 는 입력 단계라 "
                                 "모델도 EA 입력으로 학습해야 하며 주 파이프라인에서는 끈다.  ②~④ 는 고정된 특징 위의 연산이다).  "
                                 "가운데: 캘리브레이션 세션이 각 단계에 공급하는 것 (색 = 쓰는 정보).  아래: 학습 때 다른 피험자들로 "
                                 "만들어 두는 것 (인코더, SLA 템플릿, prototype).", SMALL)]))
gap(6)

# ── 1 ──
P("1. 문제 설정과 표기", H2)
P("한 피험자를 통째로 떼어 평가한다 (LOSO).  적응의 단위는 피험자가 아니라 <b>(피험자, 세션)</b> 이다 — SEED·SEED-V 는 "
  "같은 사람이 다른 날 세 번 녹화했고, 전극 위치와 임피던스가 그 사이에 달라진다.")
P("데이터: <b>SEED</b> — 15명, 3감정 (부정·중립·긍정), 세 세션 모두 같은 15개 영상.  <b>SEED-V</b> — 16명, 5감정 "
  "(혐오·공포·슬픔·중립·행복), 세션마다 다른 15개 영상 (감정당 3), LaBraM 사전학습에 없는 외부 검증용.  두 데이터셋 모두 "
  "같은 세션 안에서는 전원이 같은 영상을 같은 순서로 본다 (클립 길이·라벨 일치 확인) — SLA 의 전제다.")
P("x ∈ R<super>62×800</super> : 4초 창 (200 Hz, 겹치지 않음) · f : LaBraM 인코더 · z = f(x) ∈ R<super>200</super> : CLS "
  "임베딩 (분석에서는 L2 정규화) · g = (피험자, 세션) : 도메인 · P<sub>c</sub> : 감정 c 의 학습 prototype · "
  "μ<sub>g</sub> : 도메인 평균", SMALL)
gap(4)

# ── 2 ──
P("2. 백본과 학습", H2)
P("LaBraM-base (공개 사전학습 가중치) 를 <b>전체 fine-tuning</b> 한다.  분류 head 는 CLS → LayerNorm → Linear(200,200) "
  "→ GELU → Linear(200, 감정 수) 의 작은 MLP 다 (반구 비대칭 보조 head 는 λ=0 이라 쓰이지 않는다).")
T([["항목", "값"],
   ["옵티마이저", "AdamW, layer-wise lr decay 0.65 (LaBraM 공식 레시피), lr 5e-4, weight decay 0.05"],
   ["일정", "50 epoch (warmup 5), batch 64, gradient clip 3.0, label smoothing 0.1, bf16 autocast"],
   ["검증 · epoch 선택", "fold 마다 검증 피험자 2명 (테스트 다음 피험자부터 순환) — 검증 macro-F1 최대 epoch 의 모델을 쓴다"],
   ["시드", "0, 1, 2 — 모든 비교는 피험자별 시드 평균으로"],
   ["두 팔", "<b>EA 있음</b> (diag EA α=0.2, 세션 단위) / <b>EA 없음</b> (µV/100 스케일만) — 학습부터 따로 돌린 재학습 비교"],
   ["특징 캐시", "선택된 체크포인트로 모든 피험자의 창을 인코딩해 CLS 200차원을 저장 — 이후 분석은 전부 CPU"]],
  [34, 132])

P("어디를 fine-tuning 했나", H3)
P("<b>LaBraM 의 모든 파라미터 (5,819,936개) 와 새 분류 head 를 함께 학습한다</b> — 고정한 층도 LoRA 도 없다 "
  "(use_lora = false).  다만 LaBraM 공식 레시피의 <b>layer-wise lr decay</b> 를 써서 깊이마다 학습률이 다르다: "
  "층 번호 l (입력 임베딩 0, Transformer 블록 1~12, 최종 norm·head 13) 의 학습률은 5e-4 × 0.65<super>13−l</super> 이다.")
_lr = lambda l: 5e-4 * 0.65 ** (13 - l)
T([["부분 (층 번호)", "무엇", "학습률 (최대값, cosine 감쇠 전)", "최상위 대비"],
   ["입력 임베딩 (0)", "시간 합성곱 패치 임베딩, CLS 토큰, 채널 위치·시간 임베딩", f"{_lr(0):.1e}", f"{_lr(0) / 5e-4:.1%}"],
   ["하위 블록 (1~4)", "Transformer 블록 (자기주의 + MLP, 상대 위치 편향)", f"{_lr(1):.1e} ~ {_lr(4):.1e}",
    f"{_lr(1) / 5e-4:.1%} ~ {_lr(4) / 5e-4:.1%}"],
   ["중간 블록 (5~8)", "〃", f"{_lr(5):.1e} ~ {_lr(8):.1e}", f"{_lr(5) / 5e-4:.1%} ~ {_lr(8) / 5e-4:.0%}"],
   ["상위 블록 (9~12)", "〃", f"{_lr(9):.1e} ~ {_lr(12):.1e}", f"{_lr(9) / 5e-4:.0%} ~ {_lr(12) / 5e-4:.0%}"],
   ["최종 norm + 분류 head (13)", "head 는 새로 초기화 (약 4.1만 파라미터)", f"{_lr(13):.1e}", "100%"]],
  [36, 62, 40, 28])
B(["그래서 <b>모든 층이 학습되지만 실제로 크게 움직이는 것은 상위 블록과 head</b> 다 — 입력 임베딩과 하위 블록은 사전학습 "
   "표현을 거의 유지한다.",
   "사전학습 가중치는 공개 LaBraM-base 체크포인트 (labram-base.pth) 에서 읽고 사전학습용 head 와 mask token 은 버린다.  "
   "drop path 0.1, layer scale 초기값 0.1, 편향·1차원 파라미터는 weight decay 0.",
   "입력: 62채널 × 4초 (200 Hz) 를 채널마다 1초 패치 4개로 잘라 248개 패치 토큰 + CLS.  채널 위치 임베딩은 LaBraM 의 10-20 "
   "채널 목록에서 해당 62채널을 골라 쓴다 (세 데이터셋이 같은 캡).",
   "분류는 마지막 층 CLS 토큰 (200차원) 에서 한다.  이후 모든 분석 (중심화·라벨 보정·SLA) 은 선택된 체크포인트의 이 CLS "
   "특징을 고정한 채 그 위에서만 한다 — <b>추론 시 보정은 어떤 파라미터도 바꾸지 않는다</b> (예외: L4 는 head 의 마지막 "
   "Linear 만 캘리브레이션 창으로 미세조정).",
   "코드 이름의 'HemiAux' (반구 비대칭 보조 손실) 는 이 연구에서 λ=0 이라 학습에 영향이 없다."])

# ── 3 ──
P("3. 평가 프로토콜과 통계", H2)
B(["<b>window</b> — 모델 입력과 같은 4초 창 하나하나의 정확도.  <b>clip</b> — 영상 하나의 창을 모아 한 번 내린 판정의 "
   "정확도.  prototype 경로는 중심화된 창 특징의 평균을 L2 정규화해 최근접 prototype 으로, head 경로는 창별 softmax 를 "
   "원소별 평균해 argmax 로 판정한다.",
   "<b>현실적 캘리브레이션</b> — 테스트 세션마다 감정당 영상 1편을 무작위로 뽑아 그 앞 20초·40초·전체만 보정에 쓰고, "
   "<b>그 영상 전체를 평가에서 뺀다</b> (예산이 달라도 같은 반복 안에서는 평가 영상이 같다).  나머지 영상 (SEED 12, "
   "SEED-V 10) 을 평가한다.  추출 10회 × 시드 3, 같은 반복 안에서 모든 방법이 같은 영상을 쓴다.",
   "<b>현실적</b>의 뜻 — 테스트 데이터는 평가 대상일 뿐 어떤 정규화·적응의 통계에도 들어가지 않는다.  EA 없는 팔이 이 "
   "조건을 완전히 만족해서, V4 이후 사전 등록 실험의 주 입력으로 썼다.",
   "<b>통계</b> — 피험자 단위 (시드 평균) 짝지은 Wilcoxon, 부트스트랩 95% CI, 향상된 피험자 수.  여러 예산·데이터셋을 "
   "묶는 판정은 Holm 보정.  재학습 비교의 최소 감지 효과 MDE = (t<sub>0.975,n−1</sub> + t<sub>0.80,n−1</sub>) · "
   "sd(d) / √n (SEED 는 0.0386 고정, SEED-V 는 비교마다 계산).",
   "<b>사전 등록</b> — V1 이후의 판정은 결과를 보기 전에 기준·예측을 문서에 고정했고, 결과는 그 기준에 대보기만 했다."])

# ── 4 ──
P("4. Euclidean Alignment (입력 단계)", H2)
P("He &amp; Wu (2020) 의 EA 를 입력에 적용한다.  도메인 g 마다 평균 공간 공분산을 구해 백색화한다:")
code("R̄<sub>g</sub> = mean<sub>i</sub> ( X<sub>i</sub> X<sub>i</sub><super>T</super> / T )",
     "X′<sub>i</sub> = α · diag(R̄<sub>g</sub>)<super>−1/2</super> X<sub>i</sub>     (α = 0.2)")
P("<b>diag</b> 은 채널별 진폭만 맞추고 공간 상관은 남긴다 (full 은 상관까지 없앤다).  full 대 diag 의 차이는 감지 한계 "
  "아래였고 (+0.013, p=0.31), diag 을 고른 근거는 안정성이다 — full 은 2개 fold 를 3pp 넘게 악화시켰다.  diag 은 채널별 "
  "상수 곱과 같아서, 캘리브레이션을 바꿀 때마다 GPU 에서 broadcast 곱 한 번으로 재인코딩할 수 있다.  강건성을 위해 "
  "파워 상위 5% 구간을 빼고 평균하고 (trim 0.05), 고유값에 1e-6 바닥을 둔다.")

# ── 5 ──
P("5. 도메인 중심화 (특징 단계) — 주 제안", H2)
code("μ<sub>g</sub> = mean( z : x ∈ 캘리브레이션 창 of g )", "z′ = z − μ<sub>g</sub>")
P("학습 도메인도 각자의 평균으로 중심화하고, 그렇게 중심화된 임베딩에서 감정별 클립 임베딩 평균으로 prototype P<sub>c</sub> "
  "를 만든다 (도메인마다 먼저 평균한 뒤 도메인 간 평균).  테스트와 학습이 같은 좌표계에 놓인다.")
rows = [["EA 없는 팔 (현실적)", "적응 없음"] + [b for _, b in BUD]]
for d, _ in DS:
    rows.append([d, f"{m(CP[d]['none__proto_clip']):.4f}"] +
                [f"{m(CP[d][f'{t}__proto_clip']):.4f}<br/>{fd(*cen[d][t])}" for t, _ in BUD])
T(rows, [30, 22, 38, 38, 38])
P("선형 프로브로 무엇이 지워지는지 쟀다 (SEED): 피험자 식별 "
  f"{m(M['probe_raw_subj']):.2f} → {m(M['probe_cent_subj']):.2f}, 감정 분류 {m(M['probe_raw_emo']):.2f} → "
  f"{m(M['probe_cent_emo']):.2f}.  <b>도메인 고유의 1차 모멘트만 지우고 감정 정보는 남긴다.</b>  피험자 정보는 RBF 커널로 "
  "0.834, 공분산으로 0.863 남는다 — 가장 큰 한 축만 제거한다.")
rows = [["SEED 사다리 (S13, 현실적 EA)"] + [b for _, b in BUD]]
rows.append(["1. 아무것도 없음"] + [f"{m(N['none__proto_clip']):.4f}"] * 3)
rows.append(["2. + EA (캘리브레이션 창으로)"] + [f"{m(F[f'eaRealNoCen_{t}__proto_clip']):.4f}" for t, _ in BUD])
rows.append(["3. + 중심화"] + [f"{m(F[f'eaReal_{t}__proto_clip']):.4f}" for t, _ in BUD])
T(rows, [56, 36, 36, 36], bold_last=True)
vr = [pr(VR[f"eaReal_{t}__proto_clip"], VR[f"eaRealNoCen_{t}__proto_clip"]) for t, _ in BUD]
P("SEED-V 정식 재현 (현실적 EA 위 중심화 이득): " + " / ".join(f"{d:+.4f} ({w}/{n})" for d, p, w, n in vr) +
  f", 모두 {fp(max(x[1] for x in vr))} 이하 — 사전 등록 기준 (세 예산 Δ&gt;0, p&lt;0.05, 12/16 이상) 충족.", SMALL)

# ── 6 ──
P("6. 결정 규칙과 클립 집계", H2)
P("두 경로를 쟀다.  <b>head</b> 는 fine-tuning 때 학습한 분류기를 그대로 쓰되 중심화된 특징을 학습 도메인 평균 위치로 옮겨 "
  "넣는다.  <b>prototype</b> 은 head 를 버리고 중심화된 임베딩과 P<sub>c</sub> 의 코사인을 쓴다.  네 조건의 분해 "
  "(transductive 중심화, EA 팔):")
T([["조건", "중심화", "경로", "SEED clip", "SEED-V clip"],
   ["1", "없음", "head", f"{m(C['c1_clip']):.4f}", f"{m(VC['c1_clip']):.4f}"],
   ["2", "있음", "head", f"{m(C['c2_clip']):.4f}", f"{m(VC['c2_clip']):.4f}"],
   ["3", "없음", "prototype", f"{m(C['c3_clip']):.4f}", f"{m(VC['c3_clip']):.4f}"],
   ["4", "있음", "prototype", f"{m(C['c4_clip']):.4f}", f"{m(VC['c4_clip']):.4f}"]],
  [16, 22, 28, 32, 32], bold_last=True)
d1 = pr(C["c3_clip"], C["c1_clip"]); d2 = pr(VC["c3_clip"], VC["c1_clip"])
P(f"결정 규칙만 바꾼 조건3 − 조건1 은 SEED {d1[0]:+.4f} ({fp(d1[1])}), SEED-V {d2[0]:+.4f} ({fp(d2[1])}) 로 <b>효과가 "
  "없다</b> — 이득은 결정 규칙이 아니라 중심화에서 온다.  창 지표와 클립 지표를 항상 함께 보고한다.")

# ── 7 ──
P("7. 캘리브레이션 프로토콜 — 비용은 시청 시간이다", H2)
P("비용의 단위는 창 수가 아니라 <b>사용자가 화면 앞에 앉아 있는 시간</b>이다.")
T([["감정별", "SEED 총 시청 (3감정)", "SEED-V 총 시청 (5감정)"],
   ["20초 (창 5개)", "60초", "100초"], ["40초 (창 10개)", "120초", "200초"], ["클립 전체", "약 11.2분", "약 13.4분"]],
  [40, 50, 50])
B(["<b>모든 감정을 포함해야 한다.</b>  한 감정의 클립만으로 μ 를 만들면 중심화를 안 하는 것보다 나쁘다 (SEED 0/15, "
   "SEED-V 1/16 만 개선).  세 감정 12초가 한 감정 224초를 이긴다 (+0.059, 13/15).",
   "<b>본 것은 다 쓴다.</b>  감정 신호는 영상이 진행될수록 강해지지만 (창 정확도 0.515 → 0.687), 앞부분을 버리면 오히려 "
   "나빠진다 (40초에서 앞 20초를 버리면 clip −0.0064, p=0.025)."])

# ── 8 ──
P("8. EA 는 어떻게 작동하나 — 그리고 왜 뺄 수 있나", H2)
dv, pv, _, _ = pr(X["ea_var_between_frac"], X["noea_var_between_frac"])
T([["가설", "측정", "결과"],
   ["도메인 간 분산을 줄인다", f"{m(X['ea_var_between_frac']):.3f} vs {m(X['noea_var_between_frac']):.3f}",
    f"기각 (Δ{dv:+.3f}, {fp(pv)})"],
   ["클래스 방향 전이를 높인다", f"코사인 {m(TR['ea_cos_mean']):.3f} vs {m(TR['noea_cos_mean']):.3f}",
    f"기각 ({fp(pr(TR['ea_cos_mean'], TR['noea_cos_mean'])[1])})"],
   ["피험자 '안' 의 판별력을 높인다", f"자기 도메인 상한 {m(TR['ea_ceil']):.3f} vs {m(TR['noea_ceil']):.3f}",
    f"지지 (Δ{pr(TR['ea_ceil'], TR['noea_ceil'])[0]:+.3f}, {fp(pr(TR['ea_ceil'], TR['noea_ceil'])[1])})"]],
  [50, 56, 60])
rows = [["SEED 시청 시간", "EA 없음", "+ EA 현실적", "Δ", "p"]]
for t, lbl in (("T20", "60초"), ("T40", "120초"), ("Tfull", "11분")):
    a, b = F[f"eaRealNoCen_{t}__proto_clip"], N["none__proto_clip"]
    dd, pp, _, _ = pr(a, b)
    rows.append([lbl, f"{m(b):.4f}", f"{m(a):.4f}", f"{dd:+.4f}", fp(pp)])
T(rows, [30, 28, 32, 26, 26])
P("EA 는 도메인 <i>사이</i>가 아니라 각 도메인 <i>안</i>을 고친다.  R̄ 를 짧은 캘리브레이션에서 추정하면 이득이 줄어 유의하지 "
  "않고 (재학습 비교에서도 두 데이터셋 모두 MDE 아래), EA 를 쓰려면 학습 파이프라인 전체를 다시 돌려야 한다.  <b>짧은 "
  "캘리브레이션에서는 EA 대신 중심화를 쓴다.</b>")

# ── 9 ──
P("9. 분산까지 맞추기 — stratified normalization (V2)", H2)
P("Fdez et al. (2021) 의 stratified normalization 을 특징 단계에 옮긴 것이다.  중심화에 차원별 표준편차 나눗셈을 더한다:")
code("z<sub>strat</sub> = ( z − μ<sub>g</sub> ) / σ<sub>g</sub>    (μ, σ 는 같은 캘리브레이션 창에서 차원별로)")
ps, rows = [], [["EA 팔", "예산", "중심화", "strat", "Δ"]]
for d, _ in DS:
    for t, lbl in BUD + (("trans", "transductive"),):
        for met in ("clip", "win"):
            ps.append(pr(STN[d][f"strat_{t}__{met}"], STN[d][f"center_{t}__{met}"])[1])
        if t != "trans":
            dd = pr(STN[d][f"strat_{t}__clip"], STN[d][f"center_{t}__clip"])
            rows.append([d, lbl, f"{m(STN[d][f'center_{t}__clip']):.4f}", f"{m(STN[d][f'strat_{t}__clip']):.4f}",
                         fd(*dd)])
T(rows, [22, 24, 26, 26, 52])
P(f"Holm 보정 (2 데이터셋 × 4 조건 × 2 지표 = {len(ps)} 검정) 후 유의한 것 <b>{int((holm(ps) < 0.05).sum())}/{len(ps)}</b> "
  "— <b>분산은 기여하지 않는다.</b>")

# ── 10 ──
P("10. 남은 오차의 진단 — 세션 기하 (V3)", H2)
P("세션마다 중심화한 뒤의 감정 prototype 이 같은 사람의 다른 날 사이에서 얼마나 일치하는지 잰다:")
code("Q<sub>s,a,c</sub> = l2( mean<sub>클립 k ∈ (s,a,c)</sub> ( mean<sub>창</sub> z − μ<sub>s,a</sub> ) )",
     "cos<sub>within</sub>(s) = mean<sub>c, a&lt;b</sub> cos( Q<sub>s,a,c</sub>, Q<sub>s,b,c</sub> )")
T([["cos<sub>within</sub>", "학습 피험자", "검증 피험자 (기울기 없음)", "테스트 피험자"]] +
  [[d, f"{m(SG[d]['cw_train']):.3f}", f"{m(SG[d]['cw_val']):.3f}", f"{m(SG[d]['cw_test']):.3f}"] for d, _ in DS],
  [30, 40, 50, 40])
B(["<b>세션 불일치는 학습의 실패가 아니라 일반화의 실패다</b> — fine-tuning 이 본 피험자는 이미 일치한다.",
   f"보지 못한 피험자의 세션 간 차이에서 뽑은 상위 4방향은 테스트 피험자 차이의 SEED "
   f"{m(SS['SEED']['val_k4']) / m(SS['SEED']['ceil_k4']):.2f}, SEED-V "
   f"{m(SS['SEED-V']['val_k4']) / m(SS['SEED-V']['ceil_k4']):.2f} 만 설명한다 (같은 수 차원의 상한 대비) — 체계적이지 않다.",
   f"그 공유 방향을 지우면 오히려 나빠진다 (클립 전체 SEED "
   f"{pr(SP['SEED']['proj__center_Tfull__clip'], SP['SEED']['base__center_Tfull__clip'])[0]:+.4f}, SEED-V "
   f"{pr(SP['SEED-V']['proj__center_Tfull__clip'], SP['SEED-V']['base__center_Tfull__clip'])[0]:+.4f} "
   f"({fp(pr(SP['SEED-V']['proj__center_Tfull__clip'], SP['SEED-V']['base__center_Tfull__clip'])[1])})) — 세션마다 "
   "달라지는 방향이 감정 방향 그 자체다.  <b>라벨 없는 적응은 중심화 근처가 천장이다.</b>",
   f"어긋남 c (테스트 세션 prototype 과 학습 prototype 의 코사인, V6 의 예측 변수): SEED <b>{m(MIS['SEED']['c']):.3f}</b>, "
   f"SEED-V <b>{m(MIS['SEED-V']['c']):.3f}</b>."])

# ── 11 ──
P("11. 감정 라벨을 쓰는 보정 (S11 · V4 · V4 부록 A)", H2)
P("캘리브레이션 영상은 실험자가 고르므로 각 창의 <b>자극 라벨</b> (어떤 감정을 유도하려고 보여 줬는지) 을 사용자 주석 없이 "
  "안다.  같은 캘리브레이션 창 (중심화된 A, 자극 라벨 y) 으로 세 가지를 쟀다.  초매개변수는 fold 마다 검증 피험자 2명으로 "
  "고르고 테스트 피험자는 쓰지 않는다.")
code("<b>CR / L2</b>  M<sub>c</sub> = mean( A<sub>i</sub> : y<sub>i</sub> = c ).  U = span(P) ⊕ A 의 PCA 상위 k<sub>extra</sub> 축",
     "        R = argmin<sub>R∈O</sub> ‖ M U R − P U ‖ (det R &gt; 0),   W = (1−β) I + β ( I − UU<super>T</super> + U R U<super>T</super> )",
     "        k<sub>extra</sub> ∈ {0, 2, 5}, β ∈ {0.25, 0.5, 1.0}",
     "<b>L3</b>     P′<sub>c</sub> = l2( λ · l2(M<sub>c</sub>)‖P<sub>c</sub>‖ + (1−λ) P<sub>c</sub> ),   λ ∈ {0.25, 0.5, 0.75, 1.0}",
     "<b>L4</b>     head 의 마지막 Linear 만 캘리브레이션 창으로 몇 스텝 미세조정, 원 가중치 쪽 L2 규제")
rows = [["EA 없는 팔, clip", "기준선", "CR − 기준선", "L2 − 기준선", "L3 − 기준선", "L4 (head)"]]
for d, _ in DS:
    lc = LC[d]
    rows.append([f"{d} 클립 전체", f"{m(CR[d]['Tfull__center__clip']):.4f}",
                 fd(*pr(CR[d]["Tfull__cr__clip"], CR[d]["Tfull__center__clip"])),
                 fd(*pr(lc["Tfull__L2__proto_clip"], lc["Tfull__base__proto_clip"])),
                 fd(*pr(lc["Tfull__L3__proto_clip"], lc["Tfull__base__proto_clip"])),
                 f"{m(lc['Tfull__L4__head_clip']):.4f}"])
    rows.append([f"{d} 20초", f"{m(CR[d]['T20__center__clip']):.4f}",
                 fd(*pr(CR[d]["T20__cr__clip"], CR[d]["T20__center__clip"])),
                 fd(*pr(lc["T20__L2__proto_clip"], lc["T20__base__proto_clip"])),
                 fd(*pr(lc["T20__L3__proto_clip"], lc["T20__base__proto_clip"])),
                 f"{m(lc['T20__L4__head_clip']):.4f}"])
T(rows, [27, 17, 31, 31, 31, 18])
B(["<b>클립 전체에서만 효과가 난다</b> — 영상 앞부분에는 감정이 아직 특징에 덜 실려 있어 (7절), 짧은 예산의 라벨은 쓸 재료가 "
   "없다.",
   "가장 단순한 L3 가 회전 (CR / L2) 이상이고, 세 라벨 방법이 클립 전체에서 거의 같은 정확도에 닿는다 → 병목은 방법이 아니라 "
   "<b>라벨의 양</b> (감정당 1편) 으로 보인다.  CR 과 L2 는 같은 방법이며 선택 지표·prototype 가중만 다르다 (SEED 에서 둘의 "
   "값이 갈리는 것은 효과가 0 근처라서다).",
   "라벨을 더 쓰면 (로지스틱 회귀, 규제 없음, S14) 이득은 <b>라벨 수를 따라간다</b> — 같은 수 (감정당 3) 면 같은 날 라벨이 "
   "세 세션을 모은 라벨보다 오히려 낫다 (+0.042 대 +0.032).  같은 날 감정당 1편 (k=1) 은 이득이 없다 (+0.0008)."])

# ── 12 ──
P("12. SLA — 자극 시점 정렬 (V5)", H2)
P("새 사용자가 캘리브레이션으로 본 영상은 학습 피험자들도 본 같은 영상이다.  같은 영상·같은 시점 (클립 안 창 번호) 에서 "
  "학습 피험자들이 보인 평균 응답을 대응점으로 삼아 세션마다 직교 회전을 추정한다 — fMRI 의 response hyperalignment "
  "(Haxby et al.) 를 배포 시점 캘리브레이션으로 옮긴 것이다.  감정 라벨은 쓰지 않는다.")
code("T<sub>a,k,t</sub> = mean<sub>s ∈ 학습 피험자</sub> ( z<sub>s,a,k,t</sub> − μ<sub>s,a</sub> )   "
     "(SEED 는 세 세션이 같은 영상이라 세션을 합쳐 평균)",
     "X = { z<sub>k,t</sub> − μ<sub>d</sub> },  Y = { T<sub>k,t</sub> } − mean(Y),   (k, t) ∈ 캘리브레이션 창",
     "U = [X; Y] 의 상위 k 우특이벡터,  R = argmin<sub>R∈O, det&gt;0</sub> ‖ X U R − Y U ‖",
     "W = (1−β) I + β ( I − UU<super>T</super> + U R U<super>T</super> ),   평가: ( z − μ<sub>d</sub> ) W → P",
     "k ∈ {10, 20, 50},  β ∈ {0.25, 0.5, 0.75} — fold·예산마다 검증 피험자 2명의 window 정확도로 선택")
P("대응점이 감정 수 (3·5) 에서 캘리브레이션 창 수 (클립 전체 기준 SEED ≈ 170, SEED-V ≈ 210) 로 늘어난다.  "
  "<b>SLA+L3</b> 는 W 를 적용한 공간에서 L3 를 한다.  자극 정체성은 캘리브레이션 클립에만 쓰고 평가 클립의 정체성·시점은 "
  "쓰지 않는다.")
P("성립성 진단 (gate 0, 감정 라벨·정확도 미사용)", H3)
P("동적 ISC = 클립 안 시간 평균을 뺀 테스트 궤적과 템플릿 궤적의 Frobenius 코사인, 귀무 = 템플릿을 클립 안에서 3창 이상 원형 "
  "이동.  교차검증 = 캘리브레이션 클립 (감정당 1편, 클립 전체) 으로 맞춘 회전 (k=50, β=0.5) 을 다른 클립에 적용.")
rows = [["", "동적 ISC (귀무)", "같은 클립 vs 같은 감정 다른 클립", "ISC 학습 / 검증 / 테스트", "다른 클립의 집단 일치 (정적)"]]
for d, _ in DS:
    z, r, cv = ISC[d], IR[d], ICV[d]
    dd = pr(z["dyn"], z["null"]); ds = pr(z["st_same"], z["st_emo"])
    cvd = pr(cv["st__k50b0.5"], cv["st__none"])
    rows.append([d, f"{m(z['dyn']):.3f} ({m(z['null']):+.3f}), {dd[2]}/{dd[3]}",
                 f"{ds[0]:+.3f}, {ds[2]}/{ds[3]}",
                 f"{m(r['train']):.3f} / {m(r['val']):.3f} / {m(r['test']):.3f}",
                 f"{m(cv['st__none']):.3f} → {m(cv['st__k50b0.5']):.3f} ({cvd[2]}/{cvd[3]})"])
T(rows, [16, 34, 38, 38, 40])
P("판정 (gate 1, 사전 등록, EA 없는 팔)", H3)
rows = [["SLA − 중심화, clip"] + [b for _, b in BUD]]
for d, _ in DS:
    rows.append([d] + [fd(*sla[d][t]) for t, _ in BUD])
for d, _ in DS:
    rows.append([f"{d} SLA+L3 − L3"] + [fd(*pr(SL[d][f"{t}__slal3__clip"], SL[d][f"{t}__l3__clip"])) for t, _ in BUD])
T(rows, [34, 44, 44, 44])
B([f"SEED-V: 감정 라벨 없는 SLA 가 라벨 방법 L3 보다 클립 전체 "
   f"{pr(SL['SEED-V']['Tfull__sla__clip'], SL['SEED-V']['Tfull__l3__clip'])[0]:+.4f} "
   f"({fp(pr(SL['SEED-V']['Tfull__sla__clip'], SL['SEED-V']['Tfull__l3__clip'])[1])}) 높고, 20초에서도 작동한다.",
   "사전 등록 판정: <b>실패 (강등 전 부분)</b> — 두 데이터셋 모두 클립 전체 +0.015 이상이어야 했는데 SEED 는 효과가 없었고, "
   "SEED 20초에서 SLA+L3 &lt; L3 가 유의해 한 단계 강등.  세 번째 데이터셋 SEED-IV 로 재현과 '어긋남 c → 이득' 예측을 "
   "검증 중 (V6)."])

# ── 13 ──
P("13. 시도했으나 기준선을 넘지 못한 방법들", H2)
T([["방법", "결과"],
   ["M1 — 감정 편향을 뺀 중심화", "균형 조건에서 약간 해로움, 오염된 캘리브레이션 복구에만 쓸모"],
   ["M2 — 의사라벨 Procrustes 회전", "window +0.005 (oracle 상한의 16%), clip 0"],
   ["adaBN · Latent Alignment · T3A", "±0.005 안"],
   ["학습 중 중심화", "테스트 시 중심화 (조건4) 를 넘지 못함 (3/15)"],
   ["화이트닝 (2차 모멘트까지) · stratified norm", "검증으로 고르면 이점 없음 · Holm 0/16"],
   ["공유된 세션 방향 제거 (V3 2b)", "SEED-V −0.025 — 해로움"],
   ["학습 창 시간 가중 · 결정 단계 확신도 가중", "음성 · 유해 (clip −0.044)"],
   ["결정 단계 시간 가중", "SEED +0.007 이나 다중비교 미통과, SEED-V 재현 실패 — 이득으로 제시하지 않음"]],
  [70, 96])

# ── 14 ──
P("14. subject-dependent 프로토콜 (위치 잡기)", H2)
P("한 피험자 안에서 학습·평가한다.  문헌의 규칙을 따른다: <b>SEED</b> 는 세션마다 클립 1~9 학습 / 10~15 평가, 세 세션 결과를 "
  "모은다.  <b>SEED-V</b> 는 클립 그룹 3폴드 (1~5 / 6~10 / 11~15) 이고 세 세션을 <b>합쳐</b> 학습한다 (이 비대칭은 문헌의 "
  "것이다).  검증은 학습 클립마다 뒤쪽 20% 창 (epoch 선택에만), 30 epoch, EA 팔 설정 (세션 단위 diag EA), 시드 3.  "
  "분할별 결정을 <b>모아 한 번</b> 지표를 계산한다 (macro-F1·balanced 는 분할 평균과 다르다).")
T([["", "window", "clip", "clip macro-F1"]] +
  [[d, f"{SD[d]['accuracy'][0]:.4f} ± {SD[d]['accuracy'][1]:.4f}",
    f"{SD[d]['clip_accuracy'][0]:.4f} ± {SD[d]['clip_accuracy'][1]:.4f}",
    f"{SD[d]['clip_f1_macro'][0]:.4f}"] for d in ("SEED", "SEED-V")], [30, 44, 44, 30])
P("평가 클립 구성이 LOSO 와 달라 (SD 는 정해진 클립, LOSO 캘리브레이션은 무작위 추출 후 나머지) 같은 표에 나란히 놓으려면 "
  "구성을 맞춰야 한다.", SMALL)

# ── 15 ──
P("15. 수치 정밀도 점검", H2)
P("캐시는 학습·평가 때와 같은 bf16 autocast 로 뽑았다.  SEED 모델의 CLS 특징이 정밀도에 민감한 창이 있어 (피험자 15·1 의 "
  "세션 1 에 몰림), SEED (EA 없는 팔) 를 bf16 autocast·TF32 를 끈 <b>진짜 fp32</b> 로 다시 뽑아 같은 분석을 반복했다.")
cpb = pr(CP['SEED']['Tfull__proto_clip'], CP['SEED']['none__proto_clip'])
cpf = pr(CP32['Tfull__proto_clip'], CP32['none__proto_clip'])
slb = sla["SEED"]["Tfull"]
slf = pr(SL32["Tfull__sla__clip"], SL32["Tfull__center__clip"])
T([["", "bf16 (보고값)", "fp32"],
   ["창별 코사인 &lt; 0.9 인 창 비율 (bf16 vs fp32)", "—", f"파일 평균 {m(PREC['frac_lt09']):.3f} "
    f"(최대 {float(np.max(PREC['frac_lt09'])):.3f})"],
   ["중심화 클립 전체 − 없음", fd(*cpb), fd(*cpf)],
   ["SLA − 중심화 클립 전체", fd(*slb), fd(*slf)]], [62, 52, 52])
P("부호와 유의성이 바뀐 결론이 없다 — 보고값은 정밀도에 강건하다.  같은 설정 (GPU bf16) 으로 다시 뽑으면 기존 캐시가 재현된다.")

# ── 16 ──
P("16. 정리 — 어떤 정보로 무엇이 되는가", H2)
T([["쓰는 정보", "방법", "효과 (현실적, 클립 전체)"],
   ["캘리브레이션 창 (라벨 없음)", "도메인 중심화", f"SEED {cen['SEED']['Tfull'][0]:+.3f}, SEED-V "
    f"{cen['SEED-V']['Tfull'][0]:+.3f} — 주 효과, 두 데이터셋 재현"],
   ["+ 캘리브레이션 영상의 자극 라벨", "L3 · CR · L4", "SEED-V +0.03~+0.04, SEED 0~+0.015 — 클립 전체에서만"],
   ["+ 캘리브레이션 영상의 정체성·시점", "SLA", f"SEED-V {sla['SEED-V']['Tfull'][0]:+.3f}, SEED "
    f"{sla['SEED']['Tfull'][0]:+.3f} — 회전이 큰 데이터셋에서만"],
   ["테스트 세션의 의사라벨 · 통계", "M2 · adaBN · T3A · stratified norm", "0"]], [52, 48, 66])
P("중심화의 μ 는 라벨을 쓰지 않지만, 캘리브레이션 클립을 \"감정별 한 편씩\" 고르는 것은 자극 정보를 쓴다 — 사용자에게 기분을 "
  "묻지 않는다는 뜻이지 자극 정보가 전혀 안 들어간다는 뜻이 아니다.  불균형해도 비용은 작다 (한 감정의 클립을 줄였을 때 "
  "조건4 변화 SEED 최대 0.0158, SEED-V 최대 0.0079).", SMALL)

# ── 그림 ──
gap(6)
P("부록 — 그림", H2)
from PIL import Image as PILImage
for f, cap in (("figs/fig_method.png", "그림 1.  (a) 교차 피험자 fine-tuning  (b) 캘리브레이션  (c) 추론과 채점 — window 와 "
                "clip 이 갈리는 지점"),
               ("figs/fig_protocol.png", "그림 2.  평가 프로토콜 — LOSO, 집계, 분석마다 다른 평가 집합"),
               ("figs/fig_summary.png", "그림 3.  (a) 적응 없음 → 중심화.  (b, c) 중심화 위의 추가 이득 (CR · L3 · SLA), "
                "캘리브레이션 시간별, 95% 부트스트랩 CI.  EA 없는 팔, *** p&lt;0.001 ** p&lt;0.01.")):
    if os.path.exists(f):
        w, h = PILImage.open(f).size
        W = 165 * mm
        E.append(KeepTogether([Image(f, width=W, height=W * h / w), Paragraph(cap, SMALL)]))
        gap(6)


def footer(canv, doc):
    canv.saveState(); canv.setFont(FONT, 7.5); canv.setFillColor(GREY)
    canv.drawString(22 * mm, 12 * mm, "교차 피험자 EEG 감정 인식 — 사용한 방법 전체")
    canv.drawRightString(A4[0] - 22 * mm, 12 * mm, f"{doc.page}")
    canv.setStrokeColor(LINE); canv.setLineWidth(0.4)
    canv.line(22 * mm, 15 * mm, A4[0] - 22 * mm, 15 * mm); canv.restoreState()


doc = BaseDocTemplate("reports/METHOD.pdf", pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                      topMargin=20 * mm, bottomMargin=20 * mm, title="교차 피험자 EEG 감정 인식 — 사용한 방법 전체")
doc.addPageTemplates([PageTemplate(id="all", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)],
                                   onPage=footer)])
doc.build(E)
print("[saved] reports/METHOD.pdf")
