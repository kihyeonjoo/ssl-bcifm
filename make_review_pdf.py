"""선행연구 조사와 연구 평가 (PDF) — 2026-10-04 개정판 (v2).

v1 (오전) 은 중심화의 선행연구와 novelty 판정, 실험 1 (stratified norm) 까지였다.  v2 는
그 뒤의 결과 (V3 진단, 감정 라벨 보정, SLA, 정밀도 점검, SEED-IV 진행) 를 넣고, 방법마다
선행연구 근거와 기여를 평가하고, 논문 방향을 정리한다.  문헌은 2026-10-04 에 웹에서 확인했고
각 항목에 출처 링크를 단다.  수치는 results/ 에서 직접 읽는다.  v1: reports/archive/.

    python make_review_pdf.py  → reports/REVIEW.pdf
"""
from __future__ import annotations

import datetime
import os
import sys

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
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
                                KeepTogether)
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
WARN = colors.HexColor("#fbeae4")
NOTE = colors.HexColor("#eef4ef")


def st(name, size=9.3, lead=14.2, color=INK, font=F, space=4, left=0):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead, textColor=color, spaceAfter=space,
                          leftIndent=left, alignment=TA_LEFT)


BODY = st("b")
BUL = st("bul", left=10)
H1 = st("h1", 15.5, 20.5, INK, FB, 5)
H2 = st("h2", 11.5, 16, ACC, FB, 5); H2.keepWithNext = 1
H3 = st("h3", 9.8, 14, INK, FB, 3); H3.keepWithNext = 1
SMALL = st("s", 8.0, 11.6, GREY, space=3)
CELL = st("c", 7.8, 10.8, INK, space=0)
CELLB = st("cb", 7.8, 10.8, INK, FB, space=0)
BOX = ParagraphStyle("box", fontName=F, fontSize=9.2, leading=14.0, textColor=INK, backColor=NOTE, borderPadding=7,
                     spaceAfter=8, leftIndent=4, rightIndent=4)
WBOX = ParagraphStyle("wbox", parent=BOX, backColor=WARN)

E = []


def P(t, s=BODY):
    E.append(Paragraph(t, s))


def B(items):
    for t in items:
        E.append(Paragraph("• " + t, BUL))


def gap(h=5):
    E.append(Spacer(1, h))


def T(rows, widths, hl=None):
    data = [[Paragraph(str(c), CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], hAlign="LEFT")
    s = [("VALIGN", (0, 0), (-1, -1), "TOP"),
         ("TOPPADDING", (0, 0), (-1, -1), 3.0), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
         ("LEFTPADDING", (0, 0), (-1, -1), 4),
         ("BACKGROUND", (0, 0), (-1, 0), BG),
         ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
         ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK)]
    for r in (hl or []):
        s.append(("BACKGROUND", (0, r), (-1, r), WARN))
    t.setStyle(TableStyle(s))
    group = [t]
    if E and isinstance(E[-1], Paragraph) and E[-1].style in (H2, H3):
        group.insert(0, E.pop())
    E.append(KeepTogether(group))
    gap(5)


def L(text, key):
    return f'<link href="{U[key]}" color="#2c5f8a">{text}</link>'


U = dict(
    he_wu="https://arxiv.org/abs/1808.05464", junq="https://iopscience.iop.org/article/10.1088/1741-2552/ad4f18",
    ea25="https://arxiv.org/abs/2502.09203", zanini="https://www.researchgate.net/publication/319223543",
    rpa="https://pubmed.ncbi.nlm.nih.gov/30596565/",
    adabn="https://www.sciencedirect.com/science/article/abs/pii/S003132031830092X",
    fdez="https://pmc.ncbi.nlm.nih.gov/articles/PMC7888301/", trap="https://arxiv.org/abs/2606.06647",
    bench="https://arxiv.org/abs/2604.16926", wimpff="https://arxiv.org/abs/2311.18520",
    labram="https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf",
    cbramod="https://arxiv.org/abs/2412.07236",
    fmbench="https://www.researchgate.net/publication/394940447_EEG-FM-Bench_A_Comprehensive_Benchmark_for_the_Systematic_Evaluation_of_EEG_Foundation_Models",
    compass="https://arxiv.org/abs/2601.17883", suo="https://arxiv.org/abs/2607.27655",
    brook="https://www.medrxiv.org/content/10.1101/2024.01.16.24301366v1", eegain="https://arxiv.org/abs/2505.18175",
    liblock="https://doi.org/10.1109/tpami.2020.2973153", kilg="https://arxiv.org/abs/2508.00531",
    kamrud="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8125354/",
    lotte="https://www.researchgate.net/publication/277605013",
    prereg="https://neurips.cc/virtual/2021/workshop/21885", prereg2="https://arxiv.org/abs/2311.18807",
    msmda="https://arxiv.org/abs/2107.07740", dmmr="https://ojs.aaai.org/index.php/AAAI/article/view/27819",
    clisa="https://arxiv.org/abs/2109.09559", ta2cl="https://arxiv.org/abs/2605.22379",
    mgcrl="https://arxiv.org/abs/2607.04139", cate="https://pmc.ncbi.nlm.nih.gov/articles/PMC12473790/",
    haxby="https://www.researchgate.net/publication/341831749",
    jiahui="https://www.sciencedirect.com/science/article/pii/S1053811919310493",
    srm="https://papers.nips.cc/paper/5855-a-reduced-dimension-fmri-shared-response-model",
    rsrm="https://ieeexplore.ieee.org/document/9483745/", parra="https://www.parralab.org/isc/",
    score="https://arxiv.org/abs/2608.19134", relrep="https://iclr.cc/virtual/2023/oral/12532",
    dcal="https://arxiv.org/abs/2101.06395",
    veil="https://proceedings.neurips.cc/paper/2021/hash/4d7a968bb636e25818ff2a3941db08c1-Abstract.html",
    face="https://arxiv.org/abs/2503.18998",
    bhosale="https://www.sciencedirect.com/science/article/abs/pii/S1746809421008867",
    faced="https://www.nature.com/articles/s41597-023-02650-w", dcdp="https://arxiv.org/abs/2509.01135",
    survey="https://arxiv.org/abs/2604.27033",
)

# ── 수치 (results/ 에서) ─────────────────────────────────────────────────
def m(a):
    return float(np.asarray(a, float).mean())


def pr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    return float(d.mean()), float(stats.wilcoxon(a, b).pvalue), int((d > 0).sum()), len(d)


DS = (("SEED", ""), ("SEED-V", "seedv_"))
CP = {d: np.load(f"results/{p}calib_protocol_noea_r10.npz") for d, p in DS}
SL = {d: np.load(f"results/{p}sla_noea.npz") for d, p in DS}
CR = {d: np.load(f"results/{p}calib_rotation_noea.npz") for d, p in DS}
STN = {d: np.load(f"results/{p}stratnorm.npz") for d, p in DS}
cen = {d: pr(CP[d]["Tfull__proto_clip"], CP[d]["none__proto_clip"]) for d, _ in DS}
sla = {d: pr(SL[d]["Tfull__sla__clip"], SL[d]["Tfull__center__clip"]) for d, _ in DS}
l3 = {d: pr(SL[d]["Tfull__l3__clip"], SL[d]["Tfull__center__clip"]) for d, _ in DS}
crd = {d: pr(CR[d]["Tfull__cr__clip"], CR[d]["Tfull__center__clip"]) for d, _ in DS}
f3 = lambda x: f"{x[0]:+.3f} ({x[2]}/{x[3]})"

# ════════════════════════════════════════════════════════════════════════
P("선행연구 조사와 연구 평가 — 개정판", H1)
P("방법별 선행연구 근거 · 기여 평가 · 논문 방향 (지도교수 · 리뷰어 관점)", SMALL)
P(f"{datetime.date.today():%Y-%m-%d} · 문헌은 이 날짜에 웹에서 직접 확인했다 (링크를 따라가면 원문).  수치는 results/ 의 "
  "원본에서 읽었다.  오전판(v1)은 reports/archive/ 에 있다.", SMALL)
gap(5)
P("<b>결론부터.</b>  이 연구에서 쓴 세 연산 — <b>EA, 도메인 중심화, SLA</b> — 은 모두 기존 기법이거나 그 직접 응용이다.  "
  "새 알고리즘을 제안하는 논문으로 내면 리뷰에서 떨어진다.  <b>그러나 연구로서의 기여는 분명하다</b>: (1) 테스트 데이터를 "
  "전혀 쓰지 않는 <b>현실적 캘리브레이션</b>을 시청 시간으로 회계했고, (2) 파운데이션 모델 특징 공간에서 오차를 "
  "<b>오프셋 · 분산 · 회전</b>으로 분해해 왜 단순한 중심화가 라벨 없는 방법의 천장인지 설명했고, (3) 이것을 "
  "<b>사전 등록 · 다중 데이터셋</b>으로 재현했고, (4) 남은 회전을 표준 캘리브레이션 영상의 시점 대응으로 회수하는 "
  "<b>SLA</b> — hyperalignment 의 EEG 캘리브레이션 첫 적용 — 를 보였다 (SEED-V, 세 번째 데이터셋 진행 중).  "
  "논문은 <b>'분석 + 진단에서 나온 최소 보정'</b> 으로 세우고, 'cross-dataset 효과' 가 아니라 <b>'다중 데이터셋 재현'</b> "
  "으로 쓴다.", BOX)

# ── 1 ──
P("1. 이 연구는 어디에 있나 — 분야 지도", H2)
T([["갈래", "무엇을 하나", "테스트 사용자 데이터", "대표 연구", "본 연구와의 관계"],
   ["교차 피험자 감정 인식 — 도메인 적응 (DA)", "라벨 없는 타깃 데이터로 분포 정렬", "<b>평가 데이터 그 자체</b> (transductive)",
    L("MS-MDA 2021", "msmda") + ", " + L("CLISA 2022", "clisa"), "보고 정확도가 높지만 배포 조건과 다르다"],
   ["— 도메인 일반화 (DG)", "타깃 없이 불변 표현 학습", "없음",
    L("DMMR (AAAI 2024)", "dmmr") + ", " + L("DCDP 2025", "dcdp"), "우리 '적응 없음' 과 같은 조건"],
   ["— 캘리브레이션 기반 (본 연구)", "짧은 캘리브레이션으로 보정", "<b>캘리브레이션 세션만</b>",
    L("Lotte 2015", "lotte") + " (캘리브레이션 시간), " + L("FACE 2025", "face"), "비용을 시청 시간으로 잰다"],
   ["EEG 파운데이션 모델", "대규모 사전학습 후 fine-tuning", "보통 미사용",
    L("LaBraM", "labram") + ", " + L("CBraMod", "cbramod") + ", " + L("EEG-FM-Bench", "fmbench"),
    "적응·캘리브레이션을 다루지 않는다"],
   ["FM 의 테스트 시점 적응 (TTA)", "Tent · SHOT · T3A 등", "평가 데이터 (온라인)",
    L("NeuroAdapt-Bench 2026", "bench") + ", " + L("Wimpff 2024", "wimpff"), "감정·중심화가 없다"],
   ["피험자 간 정렬", "공분산 · 평균 · 회전 맞춤", "도메인 통계",
    L("EA", "he_wu") + ", " + L("RPA", "rpa") + ", " + L("hyperalignment", "haxby"), "세 연산의 원조"],
   ["평가 방법론", "누수 · 분할 · 자극 반복 문제", "—",
    L("Li 2021 TPAMI", "liblock") + ", " + L("Kamrud 2021", "kamrud") + ", " + L("Kilgallen 2025", "kilg"),
    "우리 프로토콜의 근거 · 남은 위협"]],
  [34, 32, 28, 40, 32])

# ── 2 ──
P("2. 방법별 선행연구와 기여 평가", H2)
P("기여 판정: <b>없음</b> = 기존 그대로 · <b>낮음</b> = 알려진 기법의 작은 변형 · <b>중</b> = 새 설정에서의 새 근거 · "
  "<b>높음</b> = 이 분야에서 찾지 못한 기여.", SMALL)

P("2-1. Euclidean Alignment (입력 단계)", H3)
P("원조: " + L("He &amp; Wu, IEEE TBME 2020", "he_wu") + " — 도메인 평균 공분산으로 백색화, 라벨 불필요.  딥러닝과의 체계적 "
  "평가: " + L("Junqueira et al., J. Neural Eng. 2024", "junq") + " (타깃 피험자 +4.33%, 수렴 시간 70% 감소).  재검토: "
  + L("Wu 2025", "ea25") + ".  리만 공간의 원형: " + L("Zanini et al. 2018", "zanini") + " (re-centering).")
P("<b>우리가 한 것</b>: diag 변형 (채널 진폭만), 테스트 쪽 R̄ 를 캘리브레이션 창으로만 추정, 그리고 기전 측정 — EA 는 "
  "도메인 <i>사이</i>의 분산을 줄이지 않고 각 도메인 <i>안</i>의 판별력을 높인다 (자기 도메인 상한 +0.032, p=0.036).  "
  "현실적 조건에서 두 데이터셋 모두 재학습 비교의 감지 한계 아래.")
P("<b>기여: 방법 없음 · 기전 해석 중 (근거 약함, 단일 검정)</b>.  논문에서는 '왜 EA 를 뺐는가' 의 근거로 짧게 쓴다.", SMALL)

P("2-2. 도메인 중심화 (특징 단계) — 주 결과", H3)
P("가장 가까운 선행연구: " + L("Fdez et al. 2021 — stratified normalization", "fdez") + " (같은 SEED, (참가자,세션)별 평균 빼기 + "
  "분산 나누기, 은닉층에서 학습 중).  " + L("AdaBN (Li et al. 2018)", "adabn") + " (도메인별 BN 통계), "
  + L("Identity Trap (Lin et al. 2026)", "trap") + " (FM 임베딩의 피험자 축을 LEACE 로 제거, 전원 라벨로 적합 — 배포 불가), "
  + L("Zanini 2018", "zanini") + " (공분산 re-centering).")
T([["", "선행연구", "본 연구"],
   ["연산", "평균 빼기 (+ 분산 나누기)", "평균 빼기만 — <b>분산은 더해 봐야 0</b> (V2: Holm 0/16)"],
   ["테스트 통계의 출처", "미명시 또는 세션 전체 (transductive)", "<b>캘리브레이션 창만</b> (평가 데이터 미사용)"],
   ["비용", "논의 없음", "<b>시청 시간</b> — 감정당 20초 (총 1분) 부터"],
   ["표현", "손설계 특징 · 소형 CNN", "EEG 파운데이션 모델의 CLS"],
   ["효과 (현실적, 클립 전체)", "—", f"SEED {f3(cen['SEED'])}, SEED-V {f3(cen['SEED-V'])}"]], [34, 56, 76])
P("<b>기여: 방법 낮음 (연산은 기존) · 회계와 측정 높음</b>.  transductive (SEED 0.7590) 와 현실적 (0.7262, 11분) 의 차이, "
  "필요한 시청 시간, 모든 감정을 포함해야 한다는 조건 (한 감정만 쓰면 중심화 안 한 것보다 나쁨, 0/15) 은 선행연구에서 "
  "찾지 못했다.", SMALL)

P("2-3. 오차 분해와 '라벨 없는 천장' (V3 진단)", H3)
P("관련: 세션을 도메인으로 다루는 " + L("MS-MDA", "msmda") + ", FM 의 피험자 축 " + L("Identity Trap", "trap") + ", 개념 관계는 "
  "보존되고 좌표 방향만 다르다는 " + L("SCORE (2026, EEG→이미지)", "score") + ".  <b>우리가 한 것</b>: 남은 오차가 세션별 감정 방향의 "
  "회전이고 (학습 피험자 세션 일치 0.965 / 0.860 대 처음 보는 피험자 0.833 / 0.414), 그 공유 부분을 지우면 오히려 "
  "나빠지며 (SEED-V −0.025), 라벨 없는 TTA (M2 · adaBN · T3A · LA) 가 모두 0 이라는 것.")
P("<b>기여: 중~높음</b> — '왜 단순한 것이 이기는가' 를 기전으로 설명한다.  " + L("NeuroAdapt-Bench", "bench") + " 의 'T3A 가 "
  "가장 안정적' 이라는 임상 결론이 감정에서는 성립하지 않는다는 새 정보도 준다.", SMALL)

P("2-4. 감정 라벨 보정 (CR / L2, L3, L4)", H3)
P("원조: 라벨 붙은 클래스 평균으로 회전 — " + L("RPA (Rodrigues 2019)", "rpa") + ".  prototype 보정 — few-shot 의 표준 "
  "(" + L("distribution calibration, ICLR 2021", "dcal") + ").  EEG 감정의 few-shot 적응 — " + L("FACE 2025", "face") + ", "
  + L("Bhosale 2022", "bhosale") + ".  " + L("Veilleux 2021", "veil") + " 는 전달식 few-shot 이 클래스 균형 가정으로 부풀려짐을 "
  "지적한다 — 우리는 그 가정을 쓰지 않았다.")
P(f"<b>결과</b>: 클립 전체에서만 효과 (SEED-V CR {f3(crd['SEED-V'])}, L3 {f3(l3['SEED-V'])}; SEED 는 0 근처), 짧은 예산에서는 0 "
  "또는 음수.  가장 단순한 L3 가 회전 이상이고 세 방법이 같은 곳에 닿는다 → 병목은 라벨의 양.")
P("<b>기여: 방법 없음 · 근거 중</b> — '이미 지불한 캘리브레이션의 자극 라벨이 얼마나 더 주는가' 의 정량.  논문에서는 대조군.",
  SMALL)

P("2-5. SLA — 자극 시점 정렬", H3)
P("원조: fMRI 의 response hyperalignment " + L("(Haxby et al.)", "haxby") + " 와 새 피험자 적용 " + L("(Jiahui et al. 2020)", "jiahui")
  + ", shared response model " + L("(Chen et al. NeurIPS 2015)", "srm") + " 과 그 EEG 적용 " + L("(RSRM 2021)", "rsrm") + ", EEG 의 "
  "피험자 간 상관 " + L("(Parra lab)", "parra") + ".  같은 영상 구간을 학습 단계에서 맞추는 대조학습 "
  + L("CLISA", "clisa") + " · " + L("TA2CL", "ta2cl") + ".  배포 시점 직교 정렬 " + L("SCORE", "score") + " (EEG–이미지 랜드마크).  "
  "앵커 기반 회전 불변 표현 " + L("(relative representations, ICLR 2023)", "relrep") + ".")
T([["", "선행연구", "SLA"],
   ["대응점", "같은 자극의 시점 (fMRI) · 교차 모달 랜드마크 (SCORE)", "<b>표준 캘리브레이션 영상의 같은 시점</b> (EEG 4초 창)"],
   ["언제", "학습 단계 (CLISA · TA2CL)", "<b>배포 시점</b>, 재학습 없음, 새 사용자의 세션마다"],
   ["라벨", "—", "<b>감정 라벨 불필요</b> (영상 정체성과 시점만)"],
   ["효과", "—", f"SEED-V {f3(sla['SEED-V'])} (감정 라벨 방법 L3 보다 높음), SEED {f3(sla['SEED'])}"]], [26, 64, 76])
P("<b>기여: 방법 중 (기법은 기존, EEG 캘리브레이션에서의 사용은 찾지 못함) · 효과는 데이터셋 의존</b>.  사전 등록 판정은 실패 "
  "(강등 전 부분) 이고, SEED-IV 에서 재현과 '어긋남 → 이득' 예측을 검증 중이다.  재현되면 논문의 가장 새로운 조각, 안 되면 "
  "'회전은 감정 응답 고유의 개인차' 라는 분석 결론이 된다.", SMALL)

P("2-6. 평가 프로토콜과 재현", H3)
P("근거: 누수와 분할의 함정 " + L("(Li et al. TPAMI 2021", "liblock") + ", " + L("Kamrud 2021", "kamrud") + ", "
  + L("Brookshire 2024)", "brook") + ", 체크포인트를 테스트로 고르면 SEED 에서 +0.104 부풀려짐 " + L("(Suo &amp; Li 2026)", "suo")
  + ", 평가 불일치 " + L("(EEGain 2025)", "eegain") + ", ML 의 사전 등록 " + L("(NeurIPS 워크숍", "prereg") + ", "
  + L("Pre-registration for Predictive Modeling)", "prereg2") + ".")
P("<b>기여: 높음</b> — 사전 등록 (V1~V6), Holm 보정, 최소 감지 효과, 시드 3, 수치 자동 검증 (219 검사), 사전학습 밖 데이터셋 "
  "재현 (9개 기준 중 8), 거짓 양성 자가 적발 (시간 가중), 정밀도 점검까지 한 연구는 이 분야에서 찾지 못했다.", SMALL)

T([["방법 / 결과", "기존 기법?", "기여 판정", "논문에서의 위치"],
   ["EA (diag)", "예 (He &amp; Wu 2020)", "방법 없음 · 기전 중", "배경, 뺀 이유"],
   ["도메인 중심화", "예 (stratified norm, AdaBN)", "방법 낮음 · <b>회계 높음</b>", "<b>주 결과</b>"],
   ["분산 정규화 (V2)", "예 (Fdez 2021)", "음성 결과 중", "중심화의 어느 부분이 일하는가"],
   ["오차 분해 · 라벨 없는 천장 (V3)", "부분 (Identity Trap, SCORE)", "<b>중~높음</b>", "<b>핵심 분석 절</b>"],
   ["감정 라벨 보정 (CR · L3 · L4)", "예 (RPA, few-shot)", "근거 중", "대조군"],
   ["SLA", "기법은 예 (hyperalignment)", "<b>중</b> (EEG 첫 적용), 데이터셋 의존", "<b>방법 절</b> (V6 결과에 따라)"],
   ["프로토콜 · 사전 등록 · 재현", "—", "<b>높음</b>", "논문 전체의 신뢰 근거"]], [44, 42, 40, 40], hl=[2, 4, 6])

# ── 3 ──
P("3. 리뷰어로서 — 예상 지적과 대응", H2)
T([["#", "지적", "현재 상태", "대응"],
   ["M1", "\"이건 stratified normalization 이다\"", "<b>답함</b> — 분산 부분은 0 (V2), 이득은 전부 평균 제거",
    "방법이 아니라 회계·분해로 프레이밍"],
   ["M2", "백본이 LaBraM 하나", "남음", "선택: CBraMod (GPU 큼).  주장을 'LaBraM 에서' 로 한정해도 됨"],
   ["M3", "세 데이터셋이 모두 같은 실험실 (SJTU BCMI)", "남음 (SEED-IV 도 같은 곳)",
    L("FACED", "faced") + " (칭화대, 123명) 가 막는다 — 서버에 없음"],
   ["<b>M4</b>", "<b>반복 자극 혼입</b> — 학습·평가 피험자가 같은 영상을 본다 " + L("(Kilgallen 2025)", "kilg"),
    "<b>새로 확인한 위협</b>.  우리 LOSO 전체에 해당, gate 0 에서 자극 고유 신호도 확인 (+0.017 / +0.027)",
    "세션마다 영상이 다른 SEED-V·SEED-IV 에서 <b>보지 못한 영상</b>으로 평가 (교차 세션 LOSO, GPU 재학습)"],
   ["M5", "절대 정확도가 문헌보다 낮다 (DMMR SEED 88%)", "근거 있음",
    "깨끗한 프로토콜에서는 0.53~0.55 " + L("(EEGain", "eegain") + ", " + L("Suo &amp; Li)", "suo") + " — DA·DG·캘리브레이션 표로 구분"],
   ["M6", "\"cross-dataset 효과\" 표현", "<b>고쳐야 함</b>",
    "분야에서 cross-dataset 은 한 데이터셋으로 학습해 다른 데이터셋에 쓰는 것 " + L("(CATE", "cate") + ", "
    + L("MGCRL)", "mgcrl") + " — 우리 것은 '다중 데이터셋 재현'"],
   ["M7", "SLA 가 한 데이터셋에서만", "V6 진행 중", "SEED-IV 결과 + 어긋남-이득 관계"],
   ["M8", "표본 15·16명", "MDE 로 명시", "감지 한계 아래 효과는 주장하지 않음"]], [9, 46, 52, 59], hl=[4])
P("<b>M4 가 이번 조사에서 새로 찾은 가장 중요한 위협이다.</b>  SEED 계열 교차 피험자 평가는 모두 같은 영상을 쓰므로 모델이 "
  "감정이 아니라 영상을 알아볼 수 있다.  분야 전체의 문제라 우리만의 약점은 아니지만, 정직한 평가를 내세우는 논문이라면 "
  "직접 다뤄야 한다.  SLA 는 캘리브레이션 영상의 정체성만 쓰고 평가 영상의 정체성은 쓰지 않으므로 이 혼입을 새로 만들지는 "
  "않는다.", WBOX)

# ── 4 ──
P("4. 지도교수로서 — 논문 방향", H2)
P("<b>제안하지 말고, 측정하고 설명하라.</b><br/>"
  "<strike>\"우리는 새로운 정렬 방법을 제안한다\"</strike><br/>"
  "→ <b>\"현실적 캘리브레이션 아래에서, EEG 파운데이션 모델의 교차 피험자 감정 인식은 무엇으로 충분하고, 왜 그런지, "
  "무엇이 남는지 — 그리고 남은 것을 이미 지불한 시청 시간에서 어떻게 회수하는지\"</b>", BOX)
P("가제", H3)
B(["<i>What Does a Short Calibration Buy? Offset, Rotation, and Stimulus-Locked Alignment for Cross-Subject EEG "
   "Emotion Recognition with Foundation Models</i>",
   "<i>Calibrating EEG Foundation Models for New Users: A Pre-registered Analysis and a Minimal Fix</i>"])
P("기여 네 개 (순서대로)", H3)
B(["<b>회계</b> — 테스트 데이터를 전혀 쓰지 않는 현실적 캘리브레이션과 시청 시간 예산.  transductive 대비 차이를 정량화.",
   f"<b>주 결과</b> — 1차 모멘트 제거 (중심화) 하나가 라벨 없는 적응의 거의 전부다 (SEED {cen['SEED'][0]:+.3f}, "
   f"SEED-V {cen['SEED-V'][0]:+.3f}).  분산 정규화·pseudo-label·TTA 는 0.",
   "<b>기전</b> — 남은 오차는 세션별 감정 방향의 회전이고 일반화 실패다.  그래서 라벨 없는 방법은 중심화가 천장이다.",
   f"<b>최소 보정</b> — 그 회전을 표준 캘리브레이션 영상의 시점 대응으로 회수하는 SLA (SEED-V {sla['SEED-V'][0]:+.3f}, 감정 라벨 "
   "불필요), 회전이 작은 데이터셋에서는 회수할 것이 없음 — 어긋남이 이득을 예측하는지 사전 등록 검증."])
P("쓸 표현 · 피할 표현", H3)
T([["피할 표현", "대신", "이유"],
   ["새 정렬 알고리즘 제안", "최소 연산이 충분함을 보이고 이유를 설명", "세 연산 모두 기존 기법"],
   ["cross-dataset 효과", "다중 데이터셋 재현 (사전학습 밖 포함)", "분야의 cross-dataset 은 데이터셋 간 전이"],
   ["라벨 없음", "사용자 라벨 없음 (자극 선택 정보는 씀)", "캘리브레이션 클립을 감정별로 고른다"],
   ["FM 덕분에 된다", "FM 특징 공간에서 측정한 결과", "FM 없는 기준선 비교가 없다"],
   ["SLA 가 회전을 고친다", "어긋남이 클 때 회수한다 (SEED-V), 작으면 0 (SEED)", "효과가 데이터셋 의존"]], [44, 66, 56])
P("제출 전 실험 — 우선순위", H3)
T([["#", "실험", "막는 지적", "비용", "상태"],
   ["1", "SEED-IV 에서 SLA · 중심화 재현 (사전 등록)", "M7, 일반성", "GPU 약 13시간", "<b>진행 중</b>"],
   ["2", "<b>보지 못한 영상으로 평가</b> (SEED-V·SEED-IV 교차 세션 LOSO)", "<b>M4</b>", "GPU 재학습 (데이터셋당 약 1일)",
    "제안"],
   ["3", "진짜 cross-dataset 전이 (SEED 로 학습 → SEED-V·IV, 정서가 대응)", "M6 (원하면)", "GPU 특징 추출 + CPU", "선택"],
   ["4", "FM 없는 기준선 (손설계 특징 + 같은 보정)", "'FM' 주장 시", "CPU", "선택"],
   ["5", "두 번째 백본 (CBraMod) · FACED", "M2, M3", "GPU 큼, FACED 는 다운로드", "보류 (사용자 결정)"]],
  [7, 70, 26, 36, 27])
P("2번이 가장 중요하다.  M4 를 막지 못하면 '정직한 평가' 라는 이 논문의 중심 주장이 약해진다.  결과가 어느 쪽이든 쓸 수 있다 — "
  "보지 못한 영상에서도 중심화가 유지되면 강한 근거가 되고, 크게 떨어지면 분야 전체에 대한 중요한 측정이 된다.")
P("어디에 낼까 (사용자 조건: 최상위 학회 + BK21 인정)", H3)
B(["10-04 오전에 확인한 BK21 목록 기준 <b>AAAI · IJCAI · ACM MM</b> 이 IF 4 이고 본 트랙 (Regular) 이어야 인정된다 — 혼합 "
   "프레이밍 (분석 + 최소 보정) 과 가장 잘 맞는다.  ACM MM 은 감성 컴퓨팅 맥락이 자연스럽다.",
   "NeurIPS · ICML 본 트랙은 방법 신규성 부족으로 어렵다.  NeurIPS Datasets &amp; Benchmarks 는 실험 2·5 까지 하면 가능성이 "
   "있지만 BK 인정 방식은 따로 확인해야 한다.",
   "저널 (J. Neural Eng., IEEE TAFFC, TNSRE) 은 엄밀성을 높이 사지만 사용자 조건 (최상위 학회) 밖이다."])

# ── 5 ──
P("5. 종합", H2)
P("<b>리뷰어로서</b>: 방법의 신규성은 낮다 — 세 연산 모두 원조가 있다.  그러나 '현실적 조건에서 무엇이 충분하고 왜 그런가' 에 "
  "대한 답과 그 엄밀성 (사전 등록, 다중 데이터셋 재현, 음성 결과 보고) 은 이 분야에서 드물다.  M4 (반복 자극) 와 M6 (표현) 를 "
  "다루면 수락 가능한 분석 논문이다.")
P("<b>지도교수로서</b>: 방향은 맞다.  지금 가진 것으로 분석 논문의 뼈대는 완성됐고, SLA 가 SEED-IV 에서 재현되면 '분석이 가리킨 "
  "성분을 공짜 정보로 회수한다' 는 방법 절이 붙는다.  다음 투자는 새 방법이 아니라 <b>보지 못한 영상 평가 (실험 2)</b> 다.")
T([["단계", "질문", "판정"],
   ["V1", "SEED-V 재현 (9개 기준)", "8 재현 · 1 실패 (시간 가중)"],
   ["V2", "분산 정규화가 더하는가", "아니다 (Holm 0/16)"],
   ["V3", "남은 오차의 정체", "세션별 회전, 라벨 없는 천장"],
   ["V4", "감정 라벨 회전 (CR)", "부분 (SEED-V 만)"],
   ["V4 부록 A", "어떤 라벨 방법이 나은가", "L3 ≥ 회전"],
   ["V5", "SLA", "실패 (강등 전 부분), 정밀도 점검 결론 불변"],
   ["V6", "SEED-IV 재현 · 어긋남 → 이득 예측", "진행 중"]], [22, 70, 74])

# ── 참고문헌 ──
P("참고문헌 (2026-10-04 확인)", H2)
refs = [("He &amp; Wu 2020, EA (IEEE TBME 67:399–410)", "he_wu"), ("Junqueira et al. 2024, EA + 딥러닝 (J. Neural Eng.)", "junq"),
        ("Wu 2025, EA 재검토", "ea25"), ("Zanini et al. 2018, 리만 re-centering", "zanini"), ("Rodrigues et al. 2019, RPA", "rpa"),
        ("Li et al. 2018, AdaBN", "adabn"), ("Fdez et al. 2021, stratified normalization", "fdez"),
        ("Lin et al. 2026, The Identity Trap", "trap"), ("Lee et al. 2026, NeuroAdapt-Bench", "bench"),
        ("Wimpff et al. 2024, 캘리브레이션 없는 온라인 TTA", "wimpff"), ("Jiang et al. 2024, LaBraM (ICLR)", "labram"),
        ("Wang et al. 2025, CBraMod (ICLR)", "cbramod"), ("EEG-FM-Bench 2025", "fmbench"), ("EEG-FM-Compass 2026", "compass"),
        ("Suo &amp; Li 2026, 평가 프로토콜", "suo"), ("Brookshire et al. 2024, EEG 데이터 누수", "brook"), ("EEGain 2025", "eegain"),
        ("Li et al. 2021, block design (TPAMI)", "liblock"), ("Kilgallen et al. 2025, 반복 자극 혼입", "kilg"),
        ("Kamrud et al. 2021, 데이터 분할", "kamrud"), ("Lotte 2015, 캘리브레이션 시간 (Proc. IEEE)", "lotte"),
        ("NeurIPS 사전 등록 워크숍 2021", "prereg"), ("Pre-registration for Predictive Modeling 2023", "prereg2"),
        ("Chen et al. 2021, MS-MDA", "msmda"), ("DMMR, AAAI 2024", "dmmr"), ("Shen et al. 2022, CLISA", "clisa"),
        ("Xie et al. 2026, TA2CL", "ta2cl"), ("Weng et al. 2026, MGCRL (cross-dataset)", "mgcrl"),
        ("CATE, cross-corpus EEG 감정", "cate"), ("Haxby et al., hyperalignment", "haxby"),
        ("Jiahui et al. 2020, 새 피험자 hyperalignment", "jiahui"), ("Chen et al. 2015, SRM (NeurIPS)", "srm"),
        ("RSRM 의 EEG 적용 2021", "rsrm"), ("Parra lab, EEG ISC", "parra"), ("SCORE 2026", "score"),
        ("Moschella et al. 2023, relative representations (ICLR)", "relrep"), ("Yang et al. 2021, distribution calibration", "dcal"),
        ("Veilleux et al. 2021, 현실적 전달식 few-shot", "veil"), ("FACE 2025", "face"), ("Bhosale et al. 2022", "bhosale"),
        ("Chen et al. 2023, FACED (Sci. Data)", "faced"), ("DCDP 2025, 도메인 일반화 prototype", "dcdp"),
        ("교차 피험자 일반화 서베이 2026", "survey")]
for i in range(0, len(refs), 2):
    pair = refs[i:i + 2]
    P(" &nbsp;·&nbsp; ".join(L(t, k) for t, k in pair), SMALL)


def footer(c, d):
    c.saveState(); c.setFont(F, 7.5); c.setFillColor(GREY)
    c.drawString(22 * mm, 12 * mm, "선행연구 조사와 연구 평가 — 개정판")
    c.drawRightString(A4[0] - 22 * mm, 12 * mm, str(d.page))
    c.setStrokeColor(LINE); c.setLineWidth(0.4)
    c.line(22 * mm, 15 * mm, A4[0] - 22 * mm, 15 * mm); c.restoreState()


doc = BaseDocTemplate("reports/REVIEW.pdf", pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm,
                      bottomMargin=20 * mm, title="선행연구 조사와 연구 평가 — 개정판")
doc.addPageTemplates([PageTemplate(id="a", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)],
                                   onPage=footer)])
doc.build(E)
print("[saved] reports/REVIEW.pdf")
