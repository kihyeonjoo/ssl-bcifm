"""보고서 PDF 의 약어를 풀어 쓴다 (2026-10-04 사용자 요청: "약어로 쓰지 말고 풀어서").

PDF 스크립트는 reportlab Paragraph 를 이 모듈의 ``Paragraph`` (같은 클래스의 하위 클래스) 로 바꿔 쓰기만 하면 된다 —
찍히는 글자에만 적용되고, 코드 변수명·결과 키·링크 주소는 건드리지 않는다.

순서: (1) 보호할 고유 표현을 자리표시자로 바꾸고  (2) 문장 단위 규칙  (3) 낱말 규칙 + 뒤따르는 조사를 받침에 맞게 고침
(예: "EA 는" → "유클리드 정렬은", "L3 가" → "prototype 혼합이")  (4) 자리표시자 복원.  태그 (<...>) 안은 바꾸지 않는다.
"""
from __future__ import annotations

import re

from reportlab.platypus import Paragraph as _RLParagraph

PROTECT = ("EEG-FM-Bench", "EEG-FM-Compass")

PHRASES = (
    ("L2 정규화", "단위 길이로 정규화"), ("원 가중치 쪽 L2 규제", "원 가중치 쪽으로 당기는 제곱합 규제"),
    ("최소 감지 효과 MDE", "최소 감지 효과"),
    ("CR 회전", "라벨 회전"), ("L3 prototype 혼합", "prototype 혼합"),
    ("CLS 토큰", "분류 토큰"),
    ("(n.s.)", "(유의하지 않음)"), ("n.s.", "유의하지 않음"),
    ("(full 은 상관까지 없앤다)", "(전체 행렬 방식은 상관까지 없앤다)"),
    ("full 대 diag 의 차이는", "전체 행렬 방식과 대각 방식의 차이는"), ("— full 은 2개 fold", "— 전체 행렬 방식은 2개 fold"),
    (" (DA)", ""), (" (DG)", ""),
    ("no-EA", "유클리드 정렬 없는"),
    ("diag EA", "대각 유클리드 정렬"),
    ("EA (diag)", "유클리드 정렬 (대각)"),
    ("SLA+L3", "자극 시점 정렬 + prototype 혼합"),
    ("CR / L2 (자극 라벨 회전)", "라벨 회전 (자극 라벨)"),
    ("회전 (CR / L2) 이상이고", "라벨 회전 이상이고"),
    ("CR 과 L2 는 같은 방법이며", "V4 의 라벨 회전과 부록 A 의 라벨 회전은 같은 방법이며"),
    ("CR / L2", "라벨 회전"),
    ("CR − 기준선", "라벨 회전 (V4) − 기준선"), ("L2 − 기준선", "라벨 회전 (부록 A) − 기준선"),
    ("M1 — 감정 편향을 뺀 중심화", "감정 편향을 뺀 중심화"),
    ("M2 — 의사라벨 Procrustes 회전", "의사라벨 Procrustes 회전"),
    ("M2 · adaBN · T3A · stratified norm", "의사라벨 회전 · 적응형 배치 정규화 · 테스트 시점 템플릿 조정 · "
                                           "stratified normalization"),
    ("(M2 · adaBN · T3A · LA) 가", "(의사라벨 회전 · 적응형 배치 정규화 · 테스트 시점 템플릿 조정 · 잠재 정렬) 이"),
    ("(M2 · adaBN · T3A · LA)", "(의사라벨 회전 · 적응형 배치 정규화 · 테스트 시점 템플릿 조정 · 잠재 정렬)"),
    ("adaBN · Latent Alignment · T3A", "적응형 배치 정규화 · 잠재 정렬 · 테스트 시점 템플릿 조정"),
    ("LEACE 로", "최소제곱 개념 삭제로"),
    ("95% CI", "95% 신뢰구간"),
    ("layer-wise lr decay", "층별 학습률 감쇠"),
)

TOKENS = (
    ("SLA", "자극 시점 정렬"), ("EA", "유클리드 정렬"), ("CR", "라벨 회전"), ("L2", "라벨 회전"),
    ("L3", "prototype 혼합"), ("L4", "분류기 적응"), ("TTA", "테스트 시점 적응"), ("FM", "파운데이션 모델"),
    ("LOSO", "피험자 하나 빼기 교차검증"), ("SD", "피험자 내 평가"), ("MDE", "최소 감지 효과"),
    ("ISC", "피험자 간 상관"), ("DA", "도메인 적응"), ("DG", "도메인 일반화"),
    ("AdaBN", "적응형 배치 정규화"), ("adaBN", "적응형 배치 정규화"), ("T3A", "테스트 시점 템플릿 조정"),
    ("LA", "잠재 정렬"), ("CLS", "분류 토큰"), ("RPA", "리만 Procrustes 분석"),
    ("RSRM", "강건 공유 응답 모델"), ("SRM", "공유 응답 모델"), ("LEACE", "최소제곱 개념 삭제"),
    ("BN", "배치 정규화"), ("CI", "신뢰구간"), ("PCA", "주성분 분석"), ("MLP", "다층 퍼셉트론"), ("lr", "학습률"),
)

# 받침에 따라 바뀌는 조사 (받침 없음, 받침 있음).  '로' 류는 ㄹ 받침이면 받침 없음 쪽을 쓴다.
_PAIRS = (("는", "은"), ("가", "이"), ("를", "을"), ("와의", "과의"), ("와", "과"),
          ("로는", "으로는"), ("로도", "으로도"), ("로", "으로"))
_FIXED = ("에서는", "에서도", "만으로", "에서", "에는", "에도", "보다", "까지", "처럼", "의", "에", "도", "만")
_ALL = sorted({p for pair in _PAIRS for p in pair} | set(_FIXED), key=len, reverse=True)
_PART = "(?P<sp> ?)(?P<pt>" + "|".join(map(re.escape, _ALL)) + r")(?=[\s,.)·\]:;!?/~—]|$)"


def _final(word: str) -> int:
    """마지막 한글 음절의 받침 번호 (0 = 받침 없음, 8 = ㄹ).  한글이 아니면 0."""
    ch = word[-1]
    return (ord(ch) - 0xAC00) % 28 if "가" <= ch <= "힣" else 0


def _particle(term: str, pt: str) -> str:
    j = _final(term)
    for a, b in _PAIRS:
        if pt in (a, b):
            if a.startswith("로"):                       # 로 / 으로
                return a if j in (0, 8) else b
            return b if j else a
    return pt


def _tokens(seg: str) -> str:
    for ab, full in TOKENS + (("diag", "대각"),):
        tail = r"(?![A-Za-z0-9_(])" if ab == "diag" else r"(?![A-Za-z0-9_])"   # 수식의 diag(·) 는 그대로
        rx = re.compile(r"(?<![A-Za-z0-9_\-])" + re.escape(ab) + tail + f"(?:{_PART})?")
        seg = rx.sub(lambda m: full + (_particle(full, m.group("pt")) if m.group("pt") else ""), seg)
    return seg


def expand(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    keep = {}
    for i, w in enumerate(PROTECT):
        key = f"\x00{i}\x00"
        if w in text:
            text = text.replace(w, key); keep[key] = w
    parts = re.split(r"(<[^>]+>)", text)
    for k, seg in enumerate(parts):
        if seg.startswith("<"):
            continue
        for a, b in PHRASES:
            seg = seg.replace(a, b)
        parts[k] = _tokens(seg)
    out = "".join(parts)
    # 약어와 풀이를 함께 썼던 자리 ("SLA — 자극 시점 정렬", "SLA (자극 시점 정렬)") 가 겹치지 않게 하나로 합친다
    terms = "|".join(sorted({re.escape(f) for _, f in TOKENS}, key=len, reverse=True))
    head = rf"(?P<t>{terms})(?P<tag>(?:</[a-z]+>)?)"
    out = re.sub(head + r"\s*—\s*(?P=t)", r"\g<t>\g<tag>", out)                    # "X — X"
    out = re.sub(head + r"\s*\(\s*(?P=t)\s*\)", r"\g<t>\g<tag>", out)             # "X (X)"
    out = re.sub(head + r"\s*\(\s*(?P=t)\s*,\s*", r"\g<t>\g<tag> (", out)          # "X (X, ...)" → "X (...)"
    for key, w in keep.items():
        out = out.replace(key, w)
    return out


class Paragraph(_RLParagraph):
    """reportlab Paragraph 와 같지만 찍기 전에 약어를 풀어 쓴다."""

    def __init__(self, text, *a, **k):
        super().__init__(expand(text), *a, **k)


if __name__ == "__main__":
    for s in ("EA 는 도메인 <i>사이</i>가 아니라", "SLA 가 감정 라벨 방법 L3 보다 높고", "가장 단순한 L3 가 회전 (CR / L2) 이상이고",
              "CR 과 L2 는 같은 방법이며 선택 지표", "중심화된 창 특징의 평균을 L2 정규화해", "EEG-FM-Bench 와 FM 의 CLS 특징",
              "라벨 없는 TTA (M2 · adaBN · T3A · LA) 가 모두 0", "no-EA 팔 SEED-V CR +0.032, SLA+L3 − L3", "SD 는 LOSO 와 달리",
              "최소 감지 효과 MDE = ...", "EA 이득은", "SLA 로 회수", "AdaBN (Li et al. 2018) (도메인별 BN 통계)",
              '<link href="https://x/EA">EA (He &amp; Wu)</link> 를 쓴다', "layer-wise lr decay 0.65, lr 5e-4"):
        print(f"  {s}\n→ {expand(s)}")
