"""
데이터셋별 설정 — SEED 전용 상수를 한곳으로 모은다.

이 파일이 생기기 전에는 `N_CLS=3`, `N_SUBJ=15`, `range(1,16)`, `"/mnt/data/original/SEED"`,
출력 문자열의 `/15` 가 여러 파일에 흩어져 있었다.  SEED-V(5클래스, 16명)로 옮기려면
그 전부를 찾아 고쳐야 하고, 하나라도 놓치면 조용히 틀린 결과가 나온다.

**SEED 설정의 값은 리팩터링 전 상수와 정확히 같다** — 그래야 기존 결과가 재현된다.

쓰는 법
-------
    from dataset_config import CFG            # 기본값(SEED) 또는 환경변수 DATASET
    CFG.n_classes, CFG.subjects, CFG.root

환경변수 ``DATASET=seedv`` 로 바꾸거나, 스크립트에서 ``set_dataset("seedv")``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    n_classes: int
    n_subjects: int
    n_sessions: int
    n_clips_per_session: int
    clips_per_class: int          # 세션당 클래스별 클립 수
    fs: int                       # 샘플링 (Hz)
    seg: int                      # 창 길이 (샘플)
    n_val_subjects: int           # fold 당 검증 피험자 수
    loader: str                   # data/ 아래 클래스 이름
    cache_dir: str                # 미세조정 특징 캐시
    ckpt_prefix: str              # A팔(EA 있음) 체크포인트 이름 접두사
    noea_ckpt_prefix: str         # no-EA 팔 체크포인트 이름 접두사
    class_names: tuple            # 라벨 0..n-1 의 이름 (출력용)
    mde: float | None             # 재학습 비교용 최소 감지 효과.
                                  # None 이면 그 비교의 짝지은 차이에서 계산한다.
    arm_csv: str                  # A팔(EA 있음) LOSO 결과 CSV
    noea_csv: str                 # no-EA 팔 LOSO 결과 CSV
    note: str = ""

    @property
    def subjects(self) -> List[int]:
        return list(range(1, self.n_subjects + 1))

    @property
    def sessions(self) -> List[int]:
        return list(range(1, self.n_sessions + 1))

    @property
    def chance(self) -> float:
        return 1.0 / self.n_classes

    @property
    def sec_per_win(self) -> float:
        return self.seg / self.fs

    @property
    def median_clip_sec(self) -> float:
        """클립 길이 중앙값(초) — 출력 문자열용.  캐시 메타에서 측정한 값이다
        (SEED 232초 균일, SEED-V 164초이고 52~296초로 흩어진다)."""
        # SEED-IV 는 2026-10-04 세션 1 에서 잰 중앙값 35.5창 = 142초.
        return {"seed": 232.0, "seedv": 164.0, "seediv": 142.0, "deap": 60.0}[self.name]

    @property
    def n_eval_clips(self) -> int:
        """감정별 1클립을 떼고 남는 평가 클립 수 (캘리브레이션 프로토콜)."""
        return self.n_clips_per_session - self.n_classes

    @property
    def max_k(self) -> int:
        """few-shot 에서 가능한 감정별 최대 클립 수 (평가용 1편을 뺀 나머지)."""
        return self.clips_per_class - 1

    def prefix(self, stem: str) -> str:
        """결과 파일 이름에 데이터셋 접두사.  SEED 는 기존 이름을 유지한다 —
        접두사를 붙이면 지금까지의 결과 파일을 전부 옮겨야 한다."""
        return stem if self.name == "seed" else f"{self.name}_{stem}"


SEED = DatasetConfig(
    name="seed",
    root="/mnt/data/original/SEED",
    n_classes=3, n_subjects=15, n_sessions=3,
    n_clips_per_session=15, clips_per_class=5,
    fs=200, seg=800, n_val_subjects=2,
    loader="SEEDRawDataset",
    cache_dir="cache_ft", ckpt_prefix="a02_", noea_ckpt_prefix="noea_",
    class_names=("부정", "중립", "긍정"),
    # A팔 확정값.  짝지은 차이 sd 0.0496 (diag−EA없음), n=15, α=0.05, power=0.80
    # (reports/EA_ANALYSIS.md).  이 값으로 보고된 판정이 있으므로 바꾸지 않는다.
    mde=0.0386,
    arm_csv="results/s0_diag_a02.csv", noea_csv="results/noea.csv",
    note="LaBraM 사전학습에 포함됨 — 외부 검증이 아니다",
)

SEEDV = DatasetConfig(
    name="seedv",
    root="/mnt/data/original/SEED-V",
    n_classes=5, n_subjects=16, n_sessions=3,
    n_clips_per_session=15, clips_per_class=3,
    fs=200, seg=800, n_val_subjects=2,
    loader="SEEDVRawDataset",
    cache_dir="cache_ft_seedv", ckpt_prefix="seedv_a02_",
    noea_ckpt_prefix="seedv_noea_",
    class_names=("disgust", "fear", "sad", "neutral", "happy"),
    # SEED 의 0.0386 을 쓰면 안 된다 — 짝지은 차이의 sd 가 데이터셋마다 다르다.
    mde=None,
    arm_csv="results/seedv_a02.csv", noea_csv="results/seedv_noea.csv",
    note="LaBraM 사전학습에 포함되지 않음 — 외부 검증 대상. "
         "클래스 5개이므로 우연 0.2, 평가 클립 10개, few-shot k 최대 2",
)

SEEDIV = DatasetConfig(
    name="seediv",
    root="/mnt/data/original/SEED-IV",
    n_classes=4, n_subjects=15, n_sessions=3,
    n_clips_per_session=24, clips_per_class=6,
    fs=200, seg=800, n_val_subjects=2,
    loader="SEEDIVRawDataset",
    cache_dir="cache_ft_seediv", ckpt_prefix="seediv_a02_",
    noea_ckpt_prefix="seediv_noea_",
    class_names=("neutral", "sad", "fear", "happy"),
    mde=None,
    # EA 팔은 돌리지 않는다 (V5 SLA 재현은 no-EA 팔이 주 입력).  arm_csv 는 형식상 둔다.
    arm_csv="results/seediv_a02.csv", noea_csv="results/seediv_noea.csv",
    note="V5 SLA 의 세 번째 데이터셋 (2026-10-04).  세션마다 다른 24개 영상, "
         "같은 세션 안에서는 전원 같은 영상 (클립 길이·라벨 일치 확인).  4클래스, 우연 0.25",
)

DEAP = DatasetConfig(
    name="deap",
    root="/mnt/data/original/DEAP",
    n_classes=4, n_subjects=32, n_sessions=1,
    # 클래스별 클립 수는 사람마다 다르다 (본인 평점 라벨).  10 은 명목값 (40편 / 4사분면) — 캘리브레이션 영상은
    # 학습 피험자 평균 평점의 사분면으로 고른다 (분석 코드 쪽).
    n_clips_per_session=40, clips_per_class=10,
    fs=200, seg=800, n_val_subjects=2,
    loader="DEAPRawDataset",
    cache_dir="cache_ft_deap", ckpt_prefix="deap_a02_",
    noea_ckpt_prefix="deap_noea_",
    class_names=("LVLA", "LVHA", "HVLA", "HVHA"),
    mde=None,
    arm_csv="results/deap_a02.csv", noea_csv="results/deap_noea.csv",
    note="외부 검증 (2026-10-05, 사용자 결정): 다른 실험실, 32명이 같은 음악 영상 40편 (1분), 한 세션.  라벨은 "
         "본인 평점의 정서가·각성도 사분면 (임계 5) — 같은 영상이라도 사람마다 다르다.  이진 V/A 는 같은 모델에서 "
         "q//2, q%2.  전처리판 128 Hz · 4-45 Hz → 200 Hz 재샘플, 기준 3초 제외, 시행당 4초 창 15개.  우연 0.25",
)

_ALL = {"seed": SEED, "seedv": SEEDV, "seediv": SEEDIV, "deap": DEAP}
_current = _ALL[os.environ.get("DATASET", "seed")]


def set_dataset(name: str) -> DatasetConfig:
    global _current
    _current = _ALL[name]
    return _current


def get() -> DatasetConfig:
    return _current


class _Proxy:
    """``CFG.n_classes`` 가 항상 **현재** 설정을 보게 한다.

    모듈 임포트 시점에 값을 복사해 두면 ``set_dataset`` 이 늦게 불렸을 때
    옛 값이 남는다."""

    def __getattr__(self, k):
        return getattr(_current, k)

    def __repr__(self):
        return repr(_current)


CFG = _Proxy()
