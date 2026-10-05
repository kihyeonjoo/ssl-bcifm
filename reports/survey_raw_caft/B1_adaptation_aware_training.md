# B1. 배포 때의 적응(정규화) 연산을 학습 때부터 같은 방식으로 수행하는 학습법 — CAFT ① 관련 연구와 새로움 판정

기준일 2026-10-05. 대상: CAFT(Calibration-Aware Fine-Tuning, 캘리브레이션 인지 파인튜닝)의 ① 배치 안 즉석 중심화(와 ②와의 결합).

**조사 방법.** 모든 논문의 제목·저자·연도·게재처를 arXiv 초록 페이지, Crossref, OpenAlex, PMLR·NeurIPS·JMLR·ISCA 페이지, Europe PMC 중 하나 이상에서 확인했다.

- **원문(PDF·전문)을 직접 읽은 논문:** ARM, 잠재 정렬, SPD 도메인별 배치 정규화, TaskNorm, Mazankiewicz 2020, Du 2017, 계층화 정규화, CLISA, CL-SSTER, TA2CL, Wu & Johnson 2021, Summers & Dinneen 2020, Schneider 2020, Nado 2020, DSBN, Dubey 2021, Blanchard 2021, Zhang 2013, Anastasakos 1996, Strand & Egeberg 2004, Matching Networks.
- **초록만 확인한 논문:** Kwak 2023(BCM), Busso 2013, Saon 2013, Miao 2015, BEN, MetaNorm, MABN, FedBN, SiloBN, Li G. 2023. 표에 "초록 기준"으로 표시했다.
- **한계:** 조사 중반부터 arXiv API가 요청 과다(HTTP 429)로 막혀, 검색은 웹 검색·OpenAlex·Crossref로, 서지 확인은 arXiv 초록 페이지로 했다. 그래서 2026년 하반기 arXiv 전용 EEG 논문 일부는 빠졌을 수 있다.
- **A1~A5와의 관계:** Kobler 2022, 잠재 정렬(Bakas), 계층화 정규화(Fdez), Personal-Zscore, CLISA, CL-SSTER, AdaBN은 A1~A5에 이미 있다. 여기서는 그 논문들의 **학습 쪽 세부**(배치 구성, 학습 중 정규화)를 원문에서 새로 확인해 덧붙였다. 특히 CLISA의 학습 중 계층화 정규화는 A3·A5에 기록되지 않았던 내용이다.

**약어.**
- 적응적 위험 최소화(Adaptive Risk Minimization, ARM), 배치 정규화(BN), 적응형 배치 정규화(AdaBN), 도메인별 배치 정규화(DSBN)
- 화자 적응 학습(Speaker Adaptive Training, SAT), 켑스트럼 평균·분산 정규화(CMVN), 단어 오류율(WER)
- 잠재 정렬(Latent Alignment, LA), 유클리드 정렬(EA), 대칭 양의 정부호(SPD)
- 피험자 하나 빼기 교차검증(LOSO), 파운데이션 모델(FM), 교차 엔트로피(CE), 자극 시점 정렬(SLA)
- 운동 상상(MI), 운동 실행(ME), 사건 관련 전위(ERP), 표면 근전도(sEMG), 인간 행동 인식(HAR)
- 배치 효과 정규화(BEN), 문맥 내 위험 최소화(ICRM), 기준 보정 모듈(BCM), 반복 특징 정규화(IFN)
- 차등 엔트로피(DE), 선형 동역학계(LDS), 피험자 간 상관(ISC), 경험적 위험 최소화(ERM)

---

## 0. 핵심 결론

- **① 의 원리는 새롭지 않다. 여러 분야에서 서로 다른 이름으로 반복해 나왔다.**
  - 원리: 배포 때 할 그룹(사람) 단위 정규화를, 학습 때 그룹 단위로 묶은 배치에서 똑같이 하고, 그 위에서 손실을 건다.
  - 음성: 화자 적응 학습(SAT, 1996)과 CMVN 관행이 있다. CMVN은 "정규화된 특징으로 모델을 학습"한다.
  - 생체신호:
    - 다중 스트림 AdaBN(sEMG, 2017): 학습 때는 세션별 BN 통계를 쓰고, 배포 때는 무라벨 캘리브레이션 데이터로 BN 통계를 바꾼다.
    - 사용자별 배치 BN(HAR, 2020)도 같은 방식이다.
  - 일반 기계학습:
    - **ARM-BN(NeurIPS 2021)**: "학습 배치를 한 도메인에서만 뽑고, 시험 때 정규화 통계를 다시 계산"한다.
    - MetaBN/TaskNorm(ICML 2020), BEN(2022)도 같은 원리다.
  - EEG:
    - 계층화 정규화(SEED, 2021), SPD 도메인별 BN(2022)
    - **잠재 정렬(2023 arXiv / 2025 JNE)**: "피험자별 BN을 학습과 추론에서" 적용해 "학습과 추론의 행동 차이를 없앤다". 학습 배치는 피험자 4명 × 12시행이다.
  - 결론: ①을 '제안 기법'으로 내세우면 리뷰어가 ARM-BN·잠재 정렬·계층화 정규화를 바로 반례로 든다.
- **CAFT의 배치 설계를 이루는 세 요소도 CLISA(IEEE TAFFC 2022/2023) 한 논문에 이미 모두 들어 있다.**
  - 세 요소: 같은 자극 위치로 맞춘 다피험자 배치, 배치 안 피험자별 정규화, 같은 자극 끌어당기기.
  - CLISA는 대조 학습 단계에서 다음을 한다.
    - "피험자 2명 × 모든 시행의 같은 시간 구간"으로 미니배치를 만든다.
    - 그 안에서 피험자별 z-점수(계층화 정규화)를 인코더 입력·풀링 출력·투영기에 적용한다.
    - 같은 구간 쌍을 InfoNCE로 끌어당긴다.
  - CL-SSTER(NeuroImage 2024)도 같은 설계다.
  - A3·A5는 CLISA를 '같은 자극 쌍' 관점에서만 기록했다. 따라서 이 학습 중 정규화를 근거로 반드시 인용·비교해야 한다.
- **정직하게 남는 차별점은 '배포 프로토콜과의 정확한 일치'와 '설정'이다. 새 원리라기보다 구현이다.**
  - **(a) 배포 프로토콜과의 일치**
    - CAFT의 배포는 시험과 분리된 짧은, 감정 균형 캘리브레이션 블록의 평균만 쓴다(비전이적). 학습 배치는 그 블록의 모양(같은 세션, 감정마다 같은 수, 같은 자극 위치, 짧은 길이)을 흉내 낸다.
    - 잠재 정렬·계층화 정규화·CLISA·SPD 도메인별 BN·ARM-BN은 모두 시험 데이터 자체(전체, 시험 배치, 스트림)로 통계를 낸다.
    - 분리된 짧은 데이터로 배포한 선례는 Du 2017(무라벨 캘리브레이션)과 Kwak 2023(1분 휴지기 기준 신호)이 있다. 다만 배치 구성을 캘리브레이션에 맞추지는 않았다.
  - **(b) 설정:** BN이 없는 사전학습 EEG 파운데이션 모델(LaBraM, LayerNorm 트랜스포머)을 전체 파인튜닝한다. 풀링 임베딩 한 곳에서 평균만 빼는 연산(새 파라미터 0)으로 ARM-BN형 학습을 구현했다.
  - **(c) 결합:** 배포 쪽 프로토타입 코사인 분류·SLA와 한 틀로 묶인다.
  - 권장 서술: "알려진 원리(SAT/ARM/MetaBN)를 FM 감정 캘리브레이션 프로토콜에 정확히 맞춰 구현하고 그 효과를 정량화했다."
  - 분야 관례: 응용 저널(TAFFC·JNE·TNSRE)에서는 이 수준도 기여로 인정된다. 그러나 '새 방법'으로 주장하면 거절 사유가 된다.
- **문헌에서 본 기대 효과 크기: '시험 때만 정규화'에 '학습 때도 같은 정규화'를 더한 추가 이득은 작고 들쭉날쭉하다.**

  | 비교 | 추가 이득 |
  |---|---|
  | 잠재 정렬 대 AdaBN(시험 때만) | +0.3~1.8%p (MI·ME·수면·P300) |
  | 다중 스트림 대 일반 AdaBN | +0.2~1.2%p |
  | ARM-BN 대 BN 적응, FEMNIST | 평균 +3.2%p, 최악 사용자 −1.2%p |
  | ARM-BN 대 BN 적응, WILDS FMoW | 평균 −9.6%p (오히려 악화) |

  우리 중간 결과(중심화 뒤 클립 정확도 +0.06, 5/5)는 이 범위보다 크다. 16명 전체와 여러 데이터셋에서 재현되면 그 자체로 보고할 가치가 있다. 재현되지 않아도 그대로 보고한다.
- **꼭 인용할 것과 꼭 비교할 것**
  - **꼭 인용:** ARM(Zhang 2021), 잠재 정렬(Bakas 2025), 계층화 정규화(Fdez 2021), CLISA(Shen 2023)·CL-SSTER(Shen 2024), 다중 스트림 AdaBN(Du 2017), BCM(Kwak 2023), TaskNorm/MetaBN(Bronskill 2020), SAT(Anastasakos 1996), SPD 도메인별 BN(Kobler 2022). 원칙 문장은 Matching Networks의 "test and train conditions must match"를 쓴다.
  - **꼭 비교((C)절 참고):**
    - 배포 때만 중심화(현재 기준선)
    - ARM-BN형: 피험자 배치이지만 구성을 맞추지 않음
    - 평균·분산 표준화형: 잠재 정렬·계층화 정규화
    - 전이적 상한: 시험 세션 전체 평균
    - 감정 불균형 캘리브레이션
    - 문맥·대상 분리(MetaBN형)
- **이름이 겹친다.**
  - "Calibration-Aware Fine-Tuning"이라는 이름은 LLM 확률 보정 논문(Xiao 외, ICML 2025)이 이미 쓰고 있다.
  - 약어 CAFT도 Concept-Aware Fine-Tuning(2025), Concept Ablation Fine-Tuning(2025)과 겹친다.
  - 기계학습에서 'calibration'은 대개 확률 보정을 뜻한다. 이름을 바꾸거나 초록 첫 문장에서 뜻을 분명히 해 두는 것이 좋다.

---

## (A) 논문 표

열 '새로움 위협'은 CAFT ①(과 ①+② 조합)의 새로움 주장에 대한 위협이다. "(원리)"는 원리 수준에서만 겹친다는 뜻이다.

### A-1. 일반 기계학습: 그룹(도메인) 단위 배치로 학습해 배포 때의 적응을 흉내

| 논문 (첫 저자, 연도, 학회/저널) | 링크 | 무엇을 하나 (1~2문장) | CAFT 와의 관계 (같은 점 / 다른 점) | 새로움 위협 |
|---|---|---|---|---|
| Zhang M., 2021, NeurIPS 2021 — *Adaptive Risk Minimization: Learning to Adapt to Domain Shift* (ARM) | [arXiv 2007.02931](https://arxiv.org/abs/2007.02931), [코드](https://github.com/henrikmarklund/arm) | 도메인별로 나뉜 학습 데이터에서 '무라벨 배치로 적응한 뒤의 손실'을 직접 최소화하는 메타학습 틀이다. ARM-BN은 학습 배치를 한 도메인에서만 뽑아 그 배치로 BN 통계를 내고, 시험 때도 시험 배치로 통계를 다시 낸다. 그 밖에 ARM-CML(문맥 망 평균을 입력에 붙임), ARM-LL(학습된 무라벨 손실)이 있다. | **같음:** 그룹 단위 배치로 배포 적응 연산을 학습에서 똑같이 한다. 통계를 통해 기울기가 흐르고, 사용자(FEMNIST 필자)를 도메인으로 본다. 사전학습 ResNet-50 미세조정 사례(Tiny ImageNet-C)도 있다. **다름:** 모든 BN 층에서 평균·분산을 쓴다. 통계는 시험 배치 자체로 낸다(전이적). 클래스 균형·자극 일치가 없고, 일관성 손실이 없으며, LayerNorm 모델에는 그대로 쓸 수 없다. | **높음** |
| Bronskill J., 2020, ICML 2020 (PMLR 119:1153–1164) — *TaskNorm: Rethinking Batch Normalization for Meta-Learning* | [PMLR](https://proceedings.mlr.press/v119/bronskill20a.html), [arXiv 2003.03284](https://arxiv.org/abs/2003.03284) | MetaBN은 "컨텍스트 집합만으로 통계를 내 컨텍스트·대상을 모두 정규화하고, 메타학습·메타시험에서 똑같이" 한다(비전이적). 컨텍스트가 작으면 추정 잡음이 커서, 인스턴스 통계와 학습된 비율 α로 섞는 TaskNorm을 제안한다. 시험 집합으로 통계를 내는 전이적 BN은 대상 집합의 클래스 비율이 학습 때와 다르면 실패한다고 보였다. | **같음:** 캘리브레이션 블록을 '컨텍스트 집합'으로 보면, 블록 평균을 시험 창에 적용하는 CAFT 배포는 MetaBN과 같은 구조다. 작은 컨텍스트의 추정 잡음 문제도 제기한다. **다름:** CAFT 학습은 평균을 낸 그 15창에 바로 손실을 건다(자기 포함). 그래서 MetaBN보다 전이적 BN에 가깝다. 과제(클래스 집합)가 매번 바뀌는 소수샷 설정이다. | 중간 |
| Vinyals O., 2016, NeurIPS 2016 — *Matching Networks for One Shot Learning* | [NeurIPS](https://proceedings.neurips.cc/paper/2016/hash/90e1357833654983612fb05e3ec9148c-Abstract.html), [arXiv 1606.04080](https://arxiv.org/abs/1606.04080) | 소수샷 분류를 위해 학습도 '클래스당 몇 개' 에피소드로 짠다. 원문: "our training procedure is based on a simple machine learning principle: test and train conditions must match." | **같음:** 배포 조건을 학습 배치로 재현한다는 CAFT 배치 설계의 원칙을 가장 짧게 표현한 고전 인용이다. **다름:** 라벨 있는 지지 집합을 쓰고, 정규화와는 무관하다. | 낮음 (원리) |
| Lin A., 2022, MLCB 2022 (PMLR 200:74–93) — *Incorporating knowledge of plates in batch normalization improves generalization of deep learning for microscopy images* (BEN) | [PMLR](https://proceedings.mlr.press/v200/lin22a.html), [bioRxiv](https://doi.org/10.1101/2022.10.14.512286) | 실험 배치(플레이트)와 딥러닝 미니배치를 일치시킨다. "학습 배치를 항상 같은 실험 배치에서 뽑아" BN이 배치 효과를 학습·추론 모두에서 표준화하게 한다. RxRx1-WILDS에서 최고 성능을 냈다(초록 기준). | **같음:** ARM-BN과 같은 원리를 생물학의 '배치 효과'(우리의 피험자 효과에 해당)에 적용했다. **다름:** 전 층 BN 평균·분산을 쓰고, 시험 때 같은 플레이트 묶음 통계를 쓴다(전이적). 영상 데이터다. | 중간 |
| Dubey A., 2021, CVPR 2021 — *Adaptive Methods for Real-World Domain Generalization* | [arXiv 2103.15796](https://arxiv.org/abs/2103.15796) | 도메인의 무라벨 소수 표본으로 특징 평균(커널 평균 임베딩)을 내 '도메인 프로토타입'으로 만들고, 분류기 입력에 이어 붙인다(F(x)=MLP(concat(F_ft(x), μ))). 학습도 도메인별 프로토타입으로 한다. 적응형 분류기의 일반화 한계를 증명했다. | **같음:** 새 도메인의 소량 무라벨 표본 평균을 배포에 쓰고, 학습에서도 같은 방식으로 계산한다. **다름:** 평균을 빼지 않고 조건 입력으로 쓴다(학습 파라미터 필요). 프로토타입 망은 따로 학습한다. | 중간 |
| Gupta S., 2024, ICLR 2024 — *Context is Environment* (ICRM) | [arXiv 2309.09888](https://arxiv.org/abs/2309.09888), [mlanthology](https://mlanthology.org/iclr/2024/gupta2024iclr-context/) | 시험 환경의 무라벨 예를 문맥으로 받아 예측하는 모델을, 환경별 문맥 시퀀스로 학습한다(문맥 내 위험 최소화). | **같음:** 환경(=피험자) 단위 문맥으로 학습하고, 배포 때 같은 형태의 문맥을 쓴다. **다름:** 정규화가 아니라 주의 기반 문맥 조건화다. 시험 스트림을 문맥으로 쓴다. | 낮음 |
| Chang W.-G., 2019, CVPR 2019 (7346–7354) — *Domain-Specific Batch Normalization for Unsupervised Domain Adaptation* (DSBN) | [doi](https://doi.org/10.1109/CVPR.2019.00753), [arXiv 1906.03950](https://arxiv.org/abs/1906.03950) | 원천·대상 도메인이 BN 층만 따로 갖고 나머지 파라미터는 공유한다. 대상 도메인은 의사라벨로 두 단계 학습한다. 다원천으로 확장했다. | **같음:** 도메인별 정규화 통계로 학습하고 추론한다. **다름:** 대상 도메인 데이터를 학습에 쓴다(전이적 비지도 적응). 새 도메인마다 다시 학습해야 한다. | 낮음 |
| Li X., 2021, ICLR 2021 — *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization* / Andreux M., 2020, MICCAI DCL 워크숍 (LNCS) — *Siloed Federated Learning for Multi-centric Histopathology Datasets* (SiloBN) | [arXiv 2102.07623](https://arxiv.org/abs/2102.07623), [doi SiloBN](https://doi.org/10.1007/978-3-030-60548-3_13) | 연합학습에서 BN(통계·파라미터)을 기관별로 남겨 기관 간 특징 이동을 흡수한다. SiloBN은 BN 통계만 기관별로 두고 학습 파라미터는 공유한다(초록 기준). | **같음:** 그룹(기관)별 정규화를 학습과 배포에서 똑같이 한다. **다름:** 시험 그룹이 학습 때 본 기관이다(새 사용자가 아님). 동기가 프라이버시다. | 낮음 |

### A-2. 시험 시점에만 정규화 — CAFT 이전 기준선('배포 때만 중심화')에 해당

| 논문 | 링크 | 무엇을 하나 | CAFT 와의 관계 | 위협 |
|---|---|---|---|---|
| Li Y., 2018, Pattern Recognition 80:109–117 — *Adaptive Batch Normalization for practical domain adaptation* (AdaBN) [A2 인용] | [doi](https://doi.org/10.1016/j.patcog.2018.03.005), [arXiv 1603.04779](https://arxiv.org/abs/1603.04779) | 학습은 보통 BN으로 하고, 시험 때 BN 통계만 대상 도메인 통계로 바꾼다. | **같음:** 배포 때 대상 통계로 정규화한다. **다름:** 학습 쪽 그룹 정규화가 없다. CAFT ① 이전 기준선과 같은 위치다. ARM 논문은 이 부분이 기존 기법이고 '학습 쪽 절반'만 ARM-BN의 새 부분이라고 명시했다. | 낮음 |
| Schneider S., 2020, NeurIPS 2020 — *Improving robustness against common corruptions by covariate shift adaptation* | [arXiv 2006.16971](https://arxiv.org/abs/2006.16971) | 시험 표본 수 n이 작을 때 원천 통계와 섞는다: μ̄ = N/(N+n)·μs + n/(N+n)·μt. N은 의사표본 수이고, n<32이면 N∈[8,128]을 권장한다. | **같음:** 짧은 캘리브레이션의 추정 잡음 문제를 다룬다. **다름:** 학습은 그대로 두고 배포 쪽만 수축한다. CAFT의 '수축 평균' 대조로 쓸 수 있다. | 낮음 |
| Nado Z., 2020, arXiv — *Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift* | [arXiv 2006.10963](https://arxiv.org/abs/2006.10963) | 예측 직전 무라벨 소배치로 BN 통계를 다시 낸다. 손상 이미지에서 이득이 크다. 원문: "mixed results when used alongside pre-training, and does not seem to perform as well under more natural types of dataset shift". | **같음:** 배포 쪽 그룹 통계를 쓴다. **다름:** 학습 쪽 일치가 없다. '사전학습 모델에서 결과가 엇갈린다'는 경고는, FM 설정의 CAFT가 학습 쪽 일치를 넣는 근거로 인용할 수 있다. | 낮음 |

### A-3. 정규화 통계의 추정 잡음, 학습과 추론의 일치

| 논문 | 링크 | 무엇을 하나 | CAFT 와의 관계 | 위협 |
|---|---|---|---|---|
| Wu Y., 2021, arXiv 기술 보고서 — *Rethinking "Batch" in BatchNorm* (Wu & Johnson) | [arXiv 2105.07576](https://arxiv.org/abs/2105.07576) | BN의 '정규화 배치' 크기·구성에 따른 학습 잡음과 학습-추론 불일치를 체계적으로 분석했다. 다중 도메인에서는 "모집단 통계 계산 방식이 SGD 학습 때 정규화하던 방식과 일치해야 하며, 아니면 추론에서 일반화하지 못한다"고 했다. 배치 통계를 통해 다른 표본의 정보를 쓰는 '배치 안 정보 누설'도 경고했다. | **같음:** CAFT ①의 정당화(배포 통계 계산 방식 = 학습 정규화 방식)를 직접 뒷받침한다. **다름:** 이미지 검출·분류이고 사용자 적응이 아니다. 정보 누설 경고는 CAFT 학습의 자기 포함 평균을 점검할 거리다. | 낮음 |
| Summers C., 2020, ICLR 2020 — *Four Things Everyone Should Know to Improve Batch Normalization* | [arXiv 1906.03548](https://arxiv.org/abs/1906.03548) | 추론 때 현재 예를 정규화 통계에 가중 반영해 학습-추론 불일치를 고친다. 작은·중간 배치에서 Ghost BN의 정규화 효과를 확인했다. | **같음:** 추정 방식의 학습-추론 일치를 다룬다(방향은 반대로, 추론을 학습에 맞춘다). **다름:** 그룹·사용자 개념이 없다. | 낮음 |
| Hoffer E., 2017, NeurIPS 2017 — *Train longer, generalize better: closing the generalization gap in large batch training of neural networks* (Ghost BN) | [arXiv 1705.08741](https://arxiv.org/abs/1705.08741) | 큰 배치를 작은 가상 배치로 쪼개 그 안에서 BN 통계를 계산한다. | **같음:** 학습 때 통계 계산 단위를 작게 잡아 잡음 수준을 조절한다. 형식은 CAFT의 '피험자당 15창 평균'과 같다. **다름:** 그룹 동질성이나 배포 일치가 목적이 아니다. | 낮음 |
| Ioffe S., 2017, NeurIPS 2017 — *Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models* | [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/c54e7837e0cd0ced286cb5995327d1ab-Abstract.html) | 학습 중 배치 통계를 이동평균 쪽으로 보정해 소배치·비독립 배치에서 학습과 추론의 차이를 줄인다. | 추정 잡음 처리의 고전이다. 그룹 개념은 없다. | 낮음 |

### A-4. 음성·음성 감정: '정규화된 공간에서 표준 모델을 학습'

| 논문 | 링크 | 무엇을 하나 | CAFT 와의 관계 | 위협 |
|---|---|---|---|---|
| Anastasakos T., 1996, ICSLP 1996 (1137–1140) — *A compact model for speaker-adaptive training* (SAT). 후속: Anastasakos 1997 ICASSP, *Speaker adaptive training: a maximum likelihood approach to speaker normalization* | [doi 1996](https://doi.org/10.21437/ICSLP.1996-253), [doi 1997](https://doi.org/10.1109/ICASSP.1997.596119) | 화자별 선형 변환과 화자 독립 모델을 함께 추정해 학습 데이터의 화자 간 변동을 "annihilate"하면서 표준 모델을 학습한다(원문). 시험 화자에는 적응 데이터로 변환을 추정한다. 20K/5K 어휘에서 WER을 19%/25% 줄였다(지도 적응 조건). | **같음:** 정규화된 공간에서 표준 모델을 학습하고, 배포 때 같은 정규화를 새 화자에게 추정한다. CAFT ①의 원리 그 자체이며, CAFT는 화자별 변환이 평행이동 하나뿐인 특수 경우다. **다름:** HMM-GMM 모델이고, 변환을 최대우도로 추정하며, 적응 데이터에 전사(라벨)를 쓴다. | 중간 (원리) |
| Viikki O., 1998, Speech Communication 25(1–3):133–147 — *Cepstral domain segmental feature vector normalization for noise robust speech recognition* (구간 CMVN). 보조: Strand O. M., 2004, ISCA Robust 2004 워크숍 — *Cepstral mean and variance normalization in the model domain* | [doi](https://doi.org/10.1016/S0167-6393(98)00033-8), [ISCA PDF](https://www.isca-archive.org/robust_2004/strand04_robust.pdf) | 켑스트럼 계수를 같은 구간 통계(평균 0, 분산 1)로 바꾼다. Strand & Egeberg는 기존 CMVN이 "정규화된 특징으로 모델을 학습해야 한다"는 점을 지적했다. 또 구간 통계가 주변 발화 내용(맥락)에 따라 달라지는 문제를 지적했다. Viikki 원문은 접근하지 못해 Strand & Egeberg의 서술로 확인했다. | **같음:** 학습·시험에 같은 그룹(구간·화자) 정규화를 쓰는 관행이다. '통계가 내용에 의존한다'는 문제는 CAFT가 자극 위치를 맞추는 이유와 같은 성격이다. **다름:** 입력 특징 수준의 고정 연산이다. | 중간 (원리) |
| Saon G., 2013, IEEE ASRU 2013 (55–59) — *Speaker adaptation of neural network acoustic models using i-vectors* | [doi](https://doi.org/10.1109/ASRU.2013.6707705) | 원문: "For both training and test, the i-vector for a given speaker is concatenated to every frame belonging to that speaker." 화자 독립 특징만 쓴 DNN보다 상대 WER을 10% 줄였다(Switchboard 300시간). | **같음:** 화자 단위 요약 통계를 학습과 배포에 똑같이 쓴다. **다름:** 빼기가 아닌 조건 입력이다(ARM-CML·Dubey와 같은 계열). | 낮음 |
| Miao Y., 2015, IEEE/ACM TASLP 23(11):1938–1949 — *Speaker Adaptive Training of Deep Neural Network Acoustic Models Using I-Vectors* | [doi](https://doi.org/10.1109/TASLP.2015.2457612) | i-vector로 화자 정규화 특징을 만드는 적응 망을 학습하고, 그 공간에서 DNN을 미세조정한다(SAT-DNN). 상대 WER을 13.5%/17.5% 줄였다. | **같음:** SAT를 딥러닝 미세조정으로 옮겼다. '정규화 공간에서 사전학습 모델을 미세조정'하는 구도가 CAFT와 같다. **다름:** 학습형 변환이고 음성이다. | 낮음 |
| Busso C., 2013, IEEE Trans. Affective Computing 4(4):386–397 — *Iterative Feature Normalization Scheme for Automatic Emotion Detection from Speech* (IFN) | [doi](https://doi.org/10.1109/T-AFFC.2013.26) | 감정 표현의 화자 의존성을 화자별 정규화로 줄이되, 원문대로 "the normalization scheme should not affect the acoustic differences between emotional classes". 그래서 화자마다 중립 발화를 반복 검출하고 그 부분집합으로 정규화 변수를 추정해 모든 발화에 적용한다. 정규화를 안 하거나 전역 정규화한 경우보다 낫다(초록 기준). | **같음:** 감정 인식에서 사람 단위 정규화를 학습과 시험 모두에 쓴다. 정규화 통계의 감정 편향을 정면으로 다룬다. 이는 CAFT가 감정 균형 블록을 쓰는 이유와 같은 문제다. **다름:** 음성이고, 기준이 중립 발화이며, 특징 수준이다. | 중간 |

### A-5. EEG·생체신호: 학습 중 피험자·세션별 정규화

| 논문 | 링크 | 무엇을 하나 | CAFT 와의 관계 | 위협 |
|---|---|---|---|---|
| Du Y., 2017, Sensors 17(3):458 — *Surface EMG-Based Inter-Session Gesture Recognition Enhanced by Deep Domain Adaptation* | [doi](https://doi.org/10.3390/s17030458), [PMC5375744](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5375744/fullTextXML) | sEMG의 '다중 스트림 AdaBN'이다. 학습 배치(1000)를 같은 세션 블록 10개로 나눠 세션별 BN 통계로 정규화한다. 원문: "we only need to ensure that the samples in each data batch are from the same session". 배포 때는 새 세션·사용자의 무라벨 캘리브레이션 데이터로 BN 통계를 갱신하고, 원하면 라벨 캘리브레이션으로 미세조정한다. NinaPro에서는 시험과 분리된 시행을 캘리브레이션에 썼다. 일반 AdaBN 대비 +0.2~1.2%p(표 7, 다수결 기준). | **같음:** 배포 때 무라벨 캘리브레이션 통계로 정규화하는 것을, 학습 때 세션 단위 배치로 흉내 낸다. CAFT ①과 거의 같은 구도이며 ARM보다 3년 앞선다. **다름:** 전 층 BN 평균·분산을 쓴다. 캘리브레이션 구성(클래스 균형)을 맞추지 않고("small and randomly selected subset of gestures"), 처음부터 학습한 ConvNet이며, 일관성 손실이 없다. | **높음** |
| Mazankiewicz A., 2020, Proc. ACM IMWUT 4(4):144 — *Incremental Real-Time Personalization in Human Activity Recognition Using Domain Adaptive Batch Normalization* | [doi](https://doi.org/10.1145/3432230), [arXiv 2005.12178](https://arxiv.org/abs/2005.12178) | HAR에서 학습 배치를 사용자별로 만들어 사용자 고유 평균·분산으로 정규화한다. 원문: "During training, these statistics are computed over user-specific batches". 배포 때는 새 사용자의 창이 들어올 때마다 지수이동평균으로 통계를 갱신한다(캘리브레이션 없음). 각 배치는 그 사용자의 클래스 비율대로 무작위 추출한다. | **같음:** 사용자를 그룹으로 보고 학습과 배포에서 같은 정규화를 쓴다. **다름:** 온라인 스트림 통계를 쓴다(시험 데이터 사용). 전 층 평균·분산이고 가속도계 데이터다. | 중간 |
| Fdez J., 2021, Front. Neurosci. 15:626277 — *Cross-Subject EEG-Based Emotion Recognition Through Neural Networks With Stratified Normalization* [A2·A3 인용] | [doi](https://doi.org/10.3389/fnins.2021.626277), [PMC7888301](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7888301/fullTextXML) | SEED에서 참가자×세션별 '계층화 정규화'로 학습한다. 그룹 i(세션당 15시행)의 평균 μ_ik와 분산 σ²_ik(전문 식 8–9)로 입력과 은닉 3층을 표준화하고(아핀 없음), 학습 데이터 전체를 한 배치로 넣는다. LOO 3클래스 79.6%(BN 67.1%)이고, 마지막 층의 피험자 식별은 31~33%(우연 20%)다. | **같음:** SEED 감정 인식에서 피험자(세션) 단위 정규화를 학습 그래프 안에 넣었다. 그룹이 같은 15클립이라 감정 균형과 내용 일치가 맞춰져 있다. **다름:** 시행 단위 손설계 특징과 작은 MLP를 쓰고, 평균과 분산을 모두 쓴다. 시험 정규화 통계의 출처가 논문에 없어 세션 전체로 보인다(전이적). FM·캘리브레이션 블록·일관성 손실이 없다. | **높음** |
| Bakas S., 2025, J. Neural Eng. 22(1):016047 — *Latent alignment in deep learning models for EEG decoding*. arXiv판(2023): *Latent Alignment with Deep Set EEG Decoders* [A2 인용] | [doi](https://doi.org/10.1088/1741-2552/adb336), [arXiv 2311.17968](https://arxiv.org/abs/2311.17968) | BEETL(NeurIPS 2021) 우승법이다. 원문: "a subject-wise batch normalization, applied during training and inference". 학습 배치를 피험자 단위로 짜고(MI: 4명 × 12시행), 각 정렬 층(최종 분류 공간 포함)에서 피험자 평균·표준편차로 표준화한 뒤 공유 아핀을 건다. 추론은 시험 피험자의 무라벨 시행 전체(또는 늘어나는 배치)로 한다. 원문: 학습·추론 모두 정렬하면 "eliminates the difference between training and inference behaviour". 시험 때만 정렬하는 AdaBN 대비 +0.3~1.8%p. | **같음:** 피험자 단위 다피험자 배치(K=4)를 쓰고, 학습과 배포의 연산이 같으며, 최종 특징 공간을 정렬한다. 클래스 균형 배치도 실험했다. 동기 문장까지 같다. **다름:** 평균과 표준편차를 여러 층에서 쓰고 아핀을 학습한다. 처음부터 학습한 작은 CNN이다. 통계를 시험 데이터 자체로 낸다(전이적). 자극 일치·일관성 손실이 없고, 감정 과제가 아니다. | **높음** |
| Kobler R. J., 2022, NeurIPS 2022 (35:6219–6235) — *SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG* [A1·A2 인용] | [arXiv 2206.01323](https://arxiv.org/abs/2206.01323), [doi](https://doi.org/10.52202/068431-0450) | SPD 다양체 위의 도메인별 모멘텀 BN(TSMNet)이다. 도메인은 피험자×세션이다. 학습 미니배치 50개를 도메인 5개 × 10개로 짜고, 도메인별 이동평균 통계로 정규화한다. 시험 때는 대상 도메인 전체 데이터로 프레셰 평균·분산을 계산한다. | **같음:** 도메인별 정규화를 학습 때 도메인 묶음 배치로 한다. **다름:** 공분산(SPD) 입력이다. 학습 때 이동평균으로 추정 잡음을 줄인다(CAFT는 블록 평균의 잡음을 그대로 흉내 낸다). 시험 통계가 전이적이고 MI 과제다. | 중간 |
| Shen X., 2023 (온라인 2022), IEEE TAFFC 14(3):2496–2511 — *Contrastive Learning of Subject-Invariant EEG Representations for Cross-Subject Emotion Recognition* (CLISA) [A3·A5 인용] | [doi](https://doi.org/10.1109/TAFFC.2022.3164516), [arXiv 2109.09559](https://arxiv.org/abs/2109.09559) | 대조 학습 단계에서 피험자 A의 각 시행에서 한 구간을, 피험자 B의 같은 시행·같은 시간 구간을 뽑아 '2명 × 모든 시행' 미니배치를 만든다. 같은 구간 쌍은 양성, 나머지는 음성이다(InfoNCE). 원문: "In stratified normalization, we concatenated the same channel of different samples from one subject in the minibatch together and conducted z-score normalization". 이를 인코더 입력·평균 풀링 출력·투영기 중간에 적용한다. 예측 단계에서는 DE 특징을 시험 스트림으로 온라인 정규화(감쇠 0.99)하고 LDS를 거쳐 MLP로 분류한다. SEED LOSO 86.4%. | **같음:** 자극 동기 다피험자 배치, 배치 안 피험자별 정규화, 같은 자극 위치 끌어당기기라는 CAFT 배치 설계의 세 요소가 모두 있다. **다름:** 정규화가 평균·분산이고, 배포 연산과 같게 설계되지 않았다(배포는 시험 스트림 정규화). 음성 쌍이 있고 2단계(대조 학습 → MLP)라 감정 CE와 공동 학습하지 않는다. 처음부터 학습한 작은 인코더이고 캘리브레이션 블록이 없다. | **높음** |
| Shen X., 2024, NeuroImage 301:120890 — *Contrastive learning of shared spatiotemporal EEG representations across individuals for naturalistic neuroscience* (CL-SSTER) [A3·A5 인용] | [doi](https://doi.org/10.1016/j.neuroimage.2024.120890), [arXiv 2402.14213](https://arxiv.org/abs/2402.14213) | CLISA 후속이다. 같은 자극·같은 시점의 피험자 쌍을 양성으로 코사인 대조 학습하고, 계층화 정규화(미니배치 안 피험자별로 평균을 빼고 표준편차로 나눔)를 입력과 풀링 뒤 표현에 적용한다. ISC가 올라갔다. | **같음:** ②+①과 같은 형태의 학습이다. **다름:** 신경과학 표현 학습이라 분류 배포가 없다. 음성 쌍이 있다. | 중간 |
| Kwak Y., 2023, IEEE JBHI 27(4):1801–1812 — *Subject-Invariant Deep Neural Networks Based on Baseline Correction for EEG Motor Imagery BCI* (BCM) | [doi](https://doi.org/10.1109/JBHI.2023.3238421) | (초록 기준) 깊은 특징을 피험자 불변 성분과 피험자 변이 성분으로 보고, 휴지기 기준 EEG를 쓰는 기준 보정 모듈(BCM)이 피험자 변이 성분을 지우도록 학습한다. '피험자 불변 손실'은 같은 클래스 특징을 피험자와 무관하게 모은다. 원문: "Using 1-min baseline-EEG signals of the new subject, our algorithm can eliminate subject-variant components from test data without the calibration process." | **같음:** 짧은 피험자 기준 기록으로 깊은 특징의 피험자 성분을 빼는 연산을 학습과 배포에 똑같이 쓴다(①). 피험자 간 같은 클래스 일치 손실(②와 유사)도 결합했다. **다름:** 기준이 휴지기(과제와 무관)이고, 학습형 모듈이며, 일치 단위가 자극 시점이 아니라 클래스다. MI 과제다. 원문 세부는 미확인. | **높음** (초록 기준) |
| Xu L., 2020, Front. Hum. Neurosci. 14:103 — *Cross-Dataset Variability Problem in EEG Decoding With Deep Learning* | [doi](https://doi.org/10.3389/fnhum.2020.00103) | "학습과 추론 전에" 피험자별로 EEG 분포를 정렬하는 '온라인 사전 정렬'이다. 시험 쪽은 들어오는 데이터로 리만 평균을 재귀 갱신한다(초록과 서베이 서술 기준). | **같음:** 입력 수준의 SAT다(피험자별 정렬 공간에서 학습). **다름:** 입력 공분산을 다루고 시험 스트림을 쓴다. | 낮음 |
| Chen H., 2023, IEEE TAFFC 14(3):2077–2088 — *Personal-Zscore: Eliminating Individual Difference for EEG-Based Cross-Subject Emotion Recognition* [A3 인용] | [doi](https://doi.org/10.1109/TAFFC.2021.3137857), [IEEE](https://ieeexplore.ieee.org/document/9662246/) | 피험자별 z-점수로 개인차를 없앤다(SEED, A3 기준). | **같음:** 감정 EEG의 피험자별 정규화를 학습과 시험에 쓴다. **다름:** 입력 특징 수준이고, 시험 통계 출처는 미확인(A3)이다. | 중간 |
| Li G., 2023, Knowledge-Based Systems 280:111011 — *Cross-subject EEG linear domain adaption based on batch normalization and depthwise convolutional neural network* | [doi](https://doi.org/10.1016/j.knosys.2023.111011) | (초록 기준) '실험 단위 배치 정규화'와 깊이별 합성곱을 척도·평행이동 선형 사상으로 묶어 원천과 대상 도메인의 차이를 줄인다. SEED·SEED-IV에서 평가했다. | **같음:** SEED 계열에서 실험(세션) 단위 정규화를 쓴다. **다름:** 대상 도메인으로 사상하는 방식이라 전이적으로 보인다. 원문 세부는 미확인. | 낮음 |

### A-6. 이론, 그리고 ② 와 닿는 문헌

| 논문 | 링크 | 무엇을 하나 | CAFT 와의 관계 | 위협 |
|---|---|---|---|---|
| Blanchard G., 2011, NeurIPS 24 — *Generalizing from Several Related Classification Tasks to a New Unlabeled Sample* / Blanchard G., 2021, JMLR 22 — *Domain Generalization by Marginal Transfer Learning* | [NeurIPS](https://proceedings.neurips.cc/paper/2011/hash/b571ecea16a9824023ee1af16897a582-Abstract.html), [JMLR](https://jmlr.org/papers/v22/17-679.html) | 분류기를 f(P̂_X, x)로 둔다. 즉 새 과제의 무라벨 표본에서 얻은 경험 주변분포를 함께 입력한다. 두 단계 생성 모형에서 일반화 한계를 증명했고, 시험 표본 크기 n_T에 조건화한 위험 E(f\|n_T)와 n_T→∞의 이상 위험을 구분했다. ARM 이론의 출발점이다. | **같음:** CAFT는 f(P̂, x)=g(φ(x)−E_P̂[φ]), 곧 주변분포를 평균 하나로만 요약하는 특수 경우다. 캘리브레이션 길이가 n_T에 해당한다. **다름:** 커널 방법이다. | 낮음 (이론) |
| Kaba S.-O., 2023, ICML 2023 (PMLR 202:15546–15566) — *Equivariance with Learned Canonicalization Functions* | [PMLR](https://proceedings.mlr.press/v202/kaba23a.html) | 입력을 정규형(canonical form)으로 보낸 뒤 임의의 망을 써서, 불변성·등변성을 아키텍처 제약 없이 '구조로' 얻는 틀이다. | **같음:** 피험자 평균 빼기는 '피험자 전체에 공통인 평행이동' 군에 대한 정규형 사상이다(평균 0인 대표 원소를 고름). 불변성을 학습이 아니라 구조로 얻는다고 서술할 때의 이론 인용이 된다. **다름:** 집합 단위 군 작용은 다루지 않는다(이 해석은 우리 것이다). | 낮음 (이론) |
| Mahajan D., 2021, ICML 2021 (PMLR 139) — *Domain Generalization using Causal Matching* (MatchDG) | [arXiv 2006.07500](https://arxiv.org/abs/2006.07500) | 원문: "inputs across domains should have the same representation if they are derived from the same object". 대상 짝이 관측되면(완전 일치) 짝 사이 표현 거리를 ERM과 함께 줄인다. | **같음:** ②의 원리와 같다. 같은 자극 위치를 같은 대상으로, 피험자를 도메인으로 보면 된다. **다름:** 이미지 데이터이고 정규화가 없다. B2 갈래와 겹칠 수 있다. | 중간 (② 관련) |

### 표에 넣지 않은 보조 문헌 (서지 확인)

**시험 시점 BN 변형·추정 잡음**
- **TTN** — Lim, ICLR 2023, [arXiv 2302.05155](https://arxiv.org/abs/2302.05155). 시험 시점 BN의 원천·대상 통계 보간.
- **배치 통계 보정** — You, 2021, [arXiv 2110.04065](https://arxiv.org/abs/2110.04065).
- **MABN** — Wu, AAAI 2024, 38(14):15961–15969, [doi](https://doi.org/10.1609/aaai.v38i14.29527). 원문: "the normalization step in BN is intrinsically unstable when the statistics are re-estimated from a few samples". 원천 통계는 두고 아핀만 메타학습으로 갱신한다.
- **MetaNorm** — Du, ICLR 2021, [OpenReview](https://openreview.net/pdf?id=9z_dNsC4B5t). 소수샷 배치의 정규화 통계를 메타학습한 하이퍼망으로 추론한다.
- **EvalNorm** — Singh & Shrivastava, ICCV 2019, 3632–3640, [doi](https://doi.org/10.1109/ICCV.2019.00373). 평가용 BN 통계 추정.
- **BN 임베딩** — Segu, Pattern Recognit. 135:109115, 2023, [doi](https://doi.org/10.1016/j.patcog.2022.109115). 도메인별 BN 통계를 도메인 임베딩으로 쓴다.
- **도메인별 정규화 최적화** — Seo, [arXiv 1907.04275](https://arxiv.org/abs/1907.04275).
- **AutoDIAL** — Carlucci, [arXiv 1704.08082](https://arxiv.org/abs/1704.08082).
- **"Be Like Water"** — Kaku, 2020, [arXiv 2002.04019](https://arxiv.org/abs/2002.04019). 뇌졸중 환자 상지 동작 등에서 외부 변수 때문에 BN 통계가 어긋나는 문제를 다룬다. 추론 때 인스턴스 정규화식 적응 통계로 해결한다. ARM이 인용했다.
- **BN 원전** — Ioffe & Szegedy, ICML 2015, PMLR 37:448–456, [PMLR](https://proceedings.mlr.press/v37/ioffe15.html).

**메타학습·도메인 일반화**
- **MLDG** — Li D., AAAI 2018, [doi](https://doi.org/10.1609/aaai.v32i1.11596). 학습 중 가짜 도메인 이동을 에피소드로 흉내 낸다.
- **Kumagai & Iwata** — 2018, [arXiv 1807.02927](https://arxiv.org/abs/1807.02927). ARM 논문이 ARM-CML과 비슷하다고 언급한 선행.
- **MT3** — Bartler, [arXiv 2103.16201](https://arxiv.org/abs/2103.16201).
- **Meta-DMoE** — Zhong, NeurIPS 2022, [arXiv 2210.03885](https://arxiv.org/abs/2210.03885).

**음성**
- **Gales 1998** — Computer Speech & Language 12(2):75–98, [doi](https://doi.org/10.1006/csla.1998.0043). 제약 최대우도 선형 변환. 서지만 확인했다.
- **VTLN** — Lee & Rose, IEEE TSAP 6(1):49–60, 1998, [doi](https://doi.org/10.1109/89.650310). 주파수 와핑 화자 정규화, WER 약 20% 감소.
- **Furui 1981** — IEEE TASSP 29(2):254–272, [doi](https://doi.org/10.1109/TASSP.1981.1163530). 켑스트럼 평균 빼기의 고전.
- **LHUC** — Swietojanski, IEEE/ACM TASLP 24(8), 2016, [arXiv 1601.02828](https://arxiv.org/abs/1601.02828).
- **Sethu 2007** — DSP 2007, 611–614, [doi](https://doi.org/10.1109/ICDSP.2007.4288656). 화자별 특징 워핑으로 음성 감정 검출을 상대 최대 20% 개선.

**EEG·생체신호**
- **Bakas 2022** — [arXiv 2202.03267](https://arxiv.org/abs/2202.03267). BEETL 대회 보고서이며 잠재 정렬의 예비판.
- **Jiménez-Guarneros & Gómez-Gil** — IEEE SPL 27:750–754, 2020, [doi](https://doi.org/10.1109/LSP.2020.2989663). AdaBN + MMD, 전이적 적응.
- **ResTL** — An, MICCAI 2024, LNCS 678–688, [doi](https://doi.org/10.1007/978-3-031-72120-5_63). 휴지기 EEG로 새 피험자에 적응하며, BCM보다 3개 데이터셋에서 +2%p 이상 높다.
- **STEM** — An, MICCAI 2026, [페이지](https://papers.miccai.org/miccai-2026/1006-Paper2756.html). 피험자 단위 에피소드의 메트릭 메타학습과 대조 자기지도를 결합한다. 피험자별 중심화는 명시되지 않았다.
- **Bhosale 2022** — BSPC 72:103289, [doi](https://doi.org/10.1016/j.bspc.2021.103289). 캘리브레이션 없는 메타학습 감정 인식.
- **FACE** — Liu, 2025, [arXiv 2503.18998](https://arxiv.org/abs/2503.18998). MAML 계열 소수샷 감정 어댑터.
- **Yang 2018** — IJCNN 2018, [doi](https://doi.org/10.1109/IJCNN.2018.8489331). DEAP에서 시행 전 기준 신호를 빼는 전처리로 정확도 약 +32%(구간 수준 평가).
- **TA2CL** — Xie, 2026, [arXiv 2605.22379](https://arxiv.org/abs/2605.22379). A3·A5 인용. 같은 자극·같은 시간 쌍을 쓰지만, 원문에서 배치 안 피험자별 정규화는 확인되지 않았다.

**이론**
- **조건부 이동 모형** — Zhang K., ICML 2013, PMLR 28(3):819–827, [PMLR](https://proceedings.mlr.press/v28/zhang13d.html). 본문에서 P(X\|Y)가 위치-척도 변환으로 바뀌는 'LS-ConS' 모형을 다룬다.
- **Deep Sets** — Zaheer, NeurIPS 2017, [arXiv 1703.06114](https://arxiv.org/abs/1703.06114). 잠재 정렬이 쓴 집합 함수 틀.
- **집단 평균 중심화** — Enders & Tofighi, Psychol. Methods 12(2):121–138, 2007, [doi](https://doi.org/10.1037/1082-989X.12.2.121). 다수준 모형의 집단 평균 중심화 대 전체 평균 중심화.
- **패널 고정효과** — Mundlak, Econometrica 46(1), 1978, [doi](https://doi.org/10.2307/1913646). 서지만 확인했다.
- **ComBat** — Johnson, Biostatistics 8(1):118–127, 2007, [doi](https://doi.org/10.1093/biostatistics/kxj037). 작은 배치에서 경험적 베이즈로 위치·척도 배치 효과를 보정한다. Fortin, NeuroImage 167:104–120, 2018, [doi](https://doi.org/10.1016/j.neuroimage.2017.11.024)는 이를 다기관 MRI에 적용했다.

**이름 충돌**
- Xiao, ICML 2025, *Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach*, [arXiv 2505.01997](https://arxiv.org/abs/2505.01997)
- Chen, 2025, *Improving Large Language Models with Concept-Aware Fine-Tuning* (CAFT), [arXiv 2506.07833](https://arxiv.org/abs/2506.07833)
- Casademunt, 2025, *Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning* (CAFT), [arXiv 2507.16795](https://arxiv.org/abs/2507.16795)

---

## (B) 가장 가까운 선행 연구 정밀 비교

### B-0. 한눈에 보는 비교 (원문 확인 기준)

| 항목 | **CAFT ① (+②)** | ARM-BN (Zhang 2021) | 잠재 정렬 (Bakas 2025) | 계층화 정규화 (Fdez 2021) | CLISA 대조 단계 (Shen 2023) | 다중 스트림 AdaBN (Du 2017) |
|---|---|---|---|---|---|---|
| 그룹 단위 | 피험자 (같은 세션) | 도메인 (사용자·회전각·손상 종류) | 피험자, ERP·수면은 피험자×세션 | 참가자×세션 | 피험자 | 세션 (또는 피험자) |
| 학습 배치 | 같은 세션 K=4명 × 같은 (클립, 초) M=15곳, 감정마다 3곳 | 도메인 2~6개 × 도메인당 50~100개를 도메인 안에서 무작위로 뽑음 (클래스 균형 없음) | 피험자 4명 × 12시행(MI), 클래스 균형 또는 무작위 | 학습 데이터 전체를 한 배치로 (참가자 × 3세션 × 15시행) | 피험자 2명 × 모든 시행에서 같은 시간 구간 1개씩 | 1000개를 같은 세션 블록 10개(블록당 100개)로 |
| 정규화 연산·위치 | 풀링 임베딩 z(200차원) 한 곳, **평균만** 빼기 | 모든 BN 층, 평균·분산 | 여러 정렬 층(최종 분류 공간 포함), 평균·표준편차 + 공유 아핀 | 입력·은닉 3층, 평균·분산 (아핀 없음) + 입력 min-max | 인코더 입력·풀링 출력·투영기 중간, z-점수 | 모든 BN 층, 평균·분산 |
| 새 파라미터 | 0 | 0 (기존 BN 아핀) | 공유 α, ζ | 0 | 0 | 0 |
| 통계를 통한 기울기 | 흐름 | 흐름 (표준 BN) | 흐름 | 명시 없음 | 명시 없음 | 흐름 (표준 BN) |
| 배포 때 통계 출처 | 시험과 **분리된** 짧은 캘리브레이션 블록(감정마다 영상 1편)의 평균을 고정해 이후 시험 창에 적용 (비전이적) | **시험 배치 자체**(같은 도메인 무라벨 50~100개), 스트리밍이면 누적 | 시험 피험자의 무라벨 시행 **전체**, 또는 늘어나는 배치 | 시험 참가자 세션 (명시 없음, 전이적으로 보임) | 대조 단계 정규화와 별개로, DE 특징을 **시험 스트림**으로 온라인 정규화 | 시험과 **분리된** 무라벨 캘리브레이션 데이터 |
| 클래스(감정) 균형 | 설계로 보장 | 통제 안 함 | 실험 변수로 다룸 | SEED 구조상 균형 | 모든 시행을 포함하므로 균형 | 무작위 |
| 내용(자극) 일치 | 그룹 사이에 같은 클립·같은 초 | 없음 | 없음 | 같은 15클립 (시행 단위) | 같은 시행·같은 시간 구간 | 없음 |
| 모델 | 사전학습 LaBraM-base 전체 파인튜닝 (LayerNorm 트랜스포머) | 작은 CNN, ImageNet 사전학습 ResNet-50 | EEGNet·DeepSleep·EEGInception, 처음부터 학습 | 작은 MLP (손설계 특징) | 작은 합성곱 인코더 + MLP | ConvNet |
| 추가 손실 | ② 같은 자극 위치 일관성 (양성만, λ=0.5) | 없음 | 없음 | 없음 | InfoNCE (음성 쌍 있음) | 없음 |
| 과제·평가 | 감정 (SEED-V 등), LOSO | 이미지 분류 | MI·ME·수면·P300, 피험자 10겹 | 감정 (SEED), LOO | 감정 (SEED, THU-EP) | sEMG 제스처 |

### B-1. ARM / ARM-BN (Zhang 외, NeurIPS 2021)

**정확히 무엇을 하나.**

- **문제 설정.**
  - 학습 데이터가 도메인 S개로 나뉜다. 시험 때는 새 도메인의 무라벨 배치 x₁…x_K를 받는다.
  - 목표는 적응한 뒤의 기대 손실(적응 위험)을 최소화하는 것이다.
  - 이론 근거로 Blanchard 외의 보조정리 9를 인용한다. p(y\|x)가 p(x)로 결정되는 조건이면, 적응 위험을 최소화한 모델이 거의 모든 도메인에서 베이즈 최적과 일치한다.
- **알고리즘 1.**
  - 도메인을 하나 고르고 거기서 K개를 뽑는다. 적응 모델 h가 θ′를 만들고, 같은 배치에서 라벨 손실로 θ와 φ를 갱신한다.
  - 원문: "This mimics the adaptation procedure at test time." 실제로는 여러 도메인을 묶어 메타 배치로 쓴다.
- **ARM-BN의 정의**(원문).
  - "first, the training batches are sampled from a single domain, rather than from the entire dataset, and second, the normalization statistics are recomputed at test time rather than using a training running average."
  - 저자들은 두 번째는 기존 시험 시점 적응(AdaBN, Schneider, Nado 등)이고 첫 번째가 ARM-BN의 새 부분이라고 썼다. PyTorch에서는 코드 한 줄로 구현된다.
- **설정.**
  - Rotated MNIST는 도메인 6개 × 50개다.
  - FEMNIST는 메타 배치 2개 × 50개다. 100개 미만인 사용자를 빼고, 학습 사용자 262명과 시험 사용자 35명이 겹치지 않는다.
  - CIFAR-10-C와 Tiny ImageNet-C는 지지 집합 100개 × 메타 배치 3개다. Tiny ImageNet-C에서는 ImageNet 사전학습 ResNet-50을 미세조정했다.
  - 시험은 사용자 데이터를 50개씩 배치로 나눠 평가하며, 마지막 배치는 작을 수 있다.
- **결과(최악 도메인 / 평균, %).**

  | 데이터 | BN 적응 (시험 때만) | ARM-BN |
  |---|---|---|
  | Rotated MNIST | 78.0 / 94.4 | 83.3 / 95.6 |
  | FEMNIST | 65.7 / 80.0 | 64.5 / 83.2 |
  | CIFAR-10-C | 60.6 / 70.9 | 61.7 / 72.4 |
  | Tiny ImageNet-C | 26.5 / 42.8 | 28.3 / 43.3 |

  - WILDS에서는 엇갈린다. RxRx1은 20.0 → 31.2(ERM 29.9)로 크게 올랐다. 반면 FMoW 평균은 51.6 → 42.0(ERM 53.0)으로 크게 나빠졌고, iWildCam은 46.4 → 70.3으로 올랐지만 ERM(71.6)보다 낮다. 저자들도 "ARM-BN performs particularly poorly on the FMoW problem"이라고만 쓰고 원인은 분석하지 않았다.
  - 스트리밍: 학습 배치가 100개여도 50개 이하에서 원래 성능 근처에 도달했다.

**학습 단계 비교.**
- 같은 점:
  1. 그룹(사용자 = 피험자) 단위 배치를 쓴다.
  2. 배치 안 그룹 통계로 정규화한 표현 위에서 라벨 손실을 건다.
  3. 통계를 통해 기울기가 흐른다.
  4. 새 파라미터가 없다.
  5. 사전학습 모델 미세조정 사례가 있다.
- 다른 점:
  1. ARM-BN은 모든 BN 층에서 평균과 분산을 쓰고, CAFT는 최종 임베딩 한 곳에서 평균만 쓴다.
  2. ARM-BN은 도메인 안에서 무작위로 뽑아 클래스 비율이 그대로 따라간다. CAFT는 감정 균형, 같은 세션, K명 사이 같은 자극 위치로 구성을 맞춘다.
  3. ARM-BN에는 일관성 손실이 없다.
  4. ARM-BN은 BN 모델을 전제한다. LaBraM 같은 LayerNorm 모델에서는 같은 일을 하려면 CAFT처럼 특징 수준에서 그룹 연산을 따로 넣어야 한다.

**배포 단계 비교.**
- 같은 점: 새 그룹의 무라벨 통계로 정규화하고 나머지 파라미터는 고정한다.
- 다른 점:
  - ARM-BN은 **예측할 시험 배치 자체**로 통계를 낸다(전이적). CAFT는 분리된 캘리브레이션 블록의 평균을 고정해 이후 창에 적용한다. TaskNorm의 MetaBN 구조이며 비전이적이다.
  - ARM은 시험 배치의 클래스 구성을 통제하지 않는다(FEMNIST는 숫자가 거의 절반이다). CAFT 블록은 감정 균형이다.
  - 분류기가 다르다. ARM은 같은 head를 쓰고, CAFT는 학습 피험자 프로토타입과의 코사인(+SLA)을 쓴다.

**위치 잡기.**
- 먼저 인정한다: "CAFT ①은 ARM-BN을 (i) 풀링 임베딩 한 곳의 평균 정규화로 줄이고, (ii) 적응에 쓰는 무라벨 집합을 '시험 배치'에서 '배포 캘리브레이션 블록 모양'으로 바꾼 사례로 볼 수 있다."
- 그다음 새로움을 (ii)의 프로토콜 일치, FM·감정 설정, ②와의 결합에 둔다.
- ARM 논문은 'BN 적응'(시험 때만) 대 'ARM-BN'을 핵심 대조로 썼다. 우리도 '배포 때만 중심화' 대 'CAFT ①'을 같은 형식(최악 피험자와 평균을 함께)으로 보고하면 리뷰어가 바로 이해한다.

### B-2. 잠재 정렬 (Bakas 외, JNE 2025; arXiv 2023)

**정확히 무엇을 하나.**
- 원문: "Intuitively, Latent Alignment can be understood as a subject-wise batch normalization, applied during training and inference."
- **학습 배치 구성.** "we carefully compose each batch of training data to include a fixed number of trials from each included subject session"
  - MI·ME: EEGNet, 피험자 4명 × 12시행 = 48
  - 수면: 피험자 세션당 64시행 × 4 = 256
  - P300: 피험자 세션당 12시행, 비표적 5 : 표적 1 비율
- **정규화.** 각 정렬 층에서 피험자의 차원별 평균·표준편차로 표준화하고, 피험자 공통 아핀(α, ζ)을 건다. 이를 "up to and including the final classification space" 반복한다. 입력에도 전극별 BN(아핀 없음)을 넣는다.
- **추론.** "all unlabeled trials of individual subjects are decoded simultaneously, using the full statistics" 한다. 시행이 하나씩 들어오면 이전 시행과 이어 붙인 '늘어나는 배치'로 계산한다.
- **동기.** 원문: "applying alignment both during training and inference proactively addresses inter-subject variability in the training set, and eliminates the difference between training and inference behaviour". 이 문장은 CAFT ①의 동기와 거의 같다.
- **대조군(AdaBN).** 같은 논문의 AdaBN은 학습 때 보통 BN을 쓰고 추론 때만 피험자 통계를 쓴다. 즉 우리의 '배포 때만 중심화'와 같은 위치다.
- **결과(균형 정확도).**

  | 과제 | AdaBN | 잠재 정렬 | 차이 |
  |---|---|---|---|
  | ME | 0.630 | 0.641 | +0.011 |
  | MI | 0.508 | 0.517 | +0.009 |
  | 수면 | 0.731 | 0.749 | +0.018 |
  | P300 | 0.866 | 0.869 | +0.003 |

  둘 사이의 직접 유의성 검정은 없다. 둘 다 기준선 대비로만 검정했다.
- **클래스 불균형.** 늦은 층에서 정렬할수록 정확도는 높지만, 통계를 내는 시행 집합의 클래스 불균형에 약해진다(A2: 극단적인 경우 우연 수준). 학습 배치를 무작위 클래스 비율로 짜도 가중 정확도가 유의하게 달라지지 않았다(10겹 짝 t-검정).

**학습 단계 비교.**
- 같은 점: 피험자 단위 다피험자 배치를 쓴다(K=4까지 같다). 배치 안 피험자 통계로 정규화한 표현에 분류 손실을 걸고, 최종 분류 공간까지 정렬하며, 동기도 같다.
- 다른 점:
  - 평균과 표준편차를 여러 층에서 쓰고 아핀을 학습한다. CAFT는 평균만, 한 곳에서, 파라미터 없이 한다.
  - 처음부터 학습한 작은 CNN이다. CAFT는 사전학습 FM을 전체 파인튜닝한다.
  - MI·ERP에는 공유 자극 시간축이 없어 자극 일치 배치가 없다.
  - 일관성 손실이 없다.

**배포 단계 비교.** 잠재 정렬은 시험 데이터 자체로 통계를 낸다(전체 또는 늘어나는 배치). CAFT는 시험 전에 받은 캘리브레이션 블록만 쓴다.

**위치 잡기.**
- "잠재 정렬의 평균 전용·단일 위치·FM 판이며, 통계를 시험 데이터가 아닌 별도 캘리브레이션 블록에서 낸다"고 쓴다.
- CAFT는 가장 늦은 층(최종 임베딩)에서 중심화한다. 잠재 정렬의 결과대로라면 클래스 불균형에 가장 약한 위치다. 이 점이 감정 균형 블록 설계를 정당화하는 근거이자, 불균형 스트레스 실험((C) C-7)이 필요한 이유다.
- 잠재 정렬은 공개 코드가 있다([GitHub](https://github.com/StylianosBakas/LatentAlignment)). LaBraM 몇 깊이에 피험자별 표준화를 넣은 'LA형' 기준선을 만들 수 있다((C) C-3).

### B-3. SEED 계보: 계층화 정규화(Fdez 2021) → CLISA(2022/2023) → CL-SSTER(2024)

**정확히 무엇을 하나.**
- **계층화 정규화.**
  - 참가자×세션 그룹 i의 15시행으로 μ_ik와 σ²_ik를 내 입력과 은닉 3층을 표준화한다(아핀 없음). 입력 특징은 먼저 그룹별 min-max로 정규화한다.
  - 학습 데이터 전체를 한 배치로 넣는다. 그래서 그룹 통계가 그 참가자·세션의 15클립 전체로 계산되며, 감정 균형과 내용 일치가 자동으로 맞는다.
  - 시험 참가자를 어떻게 정규화하는지는 논문에 없다(A2와 같은 판단). 같은 연산이라면 시험 세션 15시행 전체를 쓰는 전이적 방식이다.
  - 마지막 층 피험자 식별 정확도가 31~33%(우연 20%)로 BN보다 낮다. 이것이 '피험자 정보 제거'의 근거로 제시됐다.
- **CLISA.**
  - 대조 학습 단계에서 '피험자 2명 × 모든 시행의 같은 시간 구간' 미니배치를 쓴다.
  - 계층화 정규화(배치 안 피험자별 z-점수)를 입력, 평균 풀링 출력, 투영기의 시간 합성곱 출력에 적용한다.
  - 같은 구간 쌍을 InfoNCE로 끌어당긴다.
  - 예측 단계에서는 인코더 뒤 DE 특징을 학습 통계로 시작해 시험 데이터로 지수 가중 갱신(감쇠 0.99)하고, LDS 평활 뒤 MLP로 분류한다.
- **CL-SSTER.** 같은 계층화 정규화(입력과 풀링 뒤)와 같은 자극·같은 시점 양성 쌍 코사인 대조 학습을 쓴다. 목적은 ISC를 높인 공유 표현이다.

**학습 단계 비교.**
- 같은 점:
  - CLISA는 (가) 자극 동기 다피험자 배치, (나) 배치 안 피험자별 정규화, (다) 같은 자극 위치 끌어당기기를 모두 갖췄다.
  - 계층화 정규화는 SEED에서 감정 균형·내용 일치 그룹 통계로 학습했다.
- 다른 점:
  1. 정규화가 평균·분산이고 여러 층에 있다.
  2. 데이터와 모델이 다르다. 손설계 DE·PSD 특징이나 원신호를 작은 망에 넣고, 사전학습 FM이 아니다.
  3. CLISA는 2명이고 시행마다 1구간이다. CAFT는 4명이고 감정마다 3위치다.
  4. CLISA는 음성 쌍이 있는 InfoNCE이고, 대조 학습 → MLP의 2단계다. CAFT는 감정 CE와 양성만 일관성을 함께 학습한다.
  5. 가장 중요한 차이: 학습 중 정규화가 배포 연산의 복제로 설계되지 않았다. CLISA는 배포 때 시험 스트림 정규화를 따로 쓴다.

**배포 단계 비교.** 세 논문 모두 캘리브레이션 블록 개념이 없다. 계층화 정규화는 시험 세션 통계를 쓰는 것으로 보이고, CLISA는 시험 스트림으로 온라인 정규화한다. CAFT는 분리된 짧은 블록만 쓴다.

**위치 잡기.**
- "SEED 계열에서 피험자별 배치 정규화(계층화 정규화)는 2021년부터 쓰인 도구다. CLISA와 CL-SSTER는 이를 자극 동기 배치와 함께 썼다."
- "CAFT는 이 정규화를 배포 캘리브레이션 연산과 같은 연산(블록 평균 빼기)으로 다시 정의하고, 블록 모양을 흉내 낸 배치로 FM 파인튜닝에 넣었다."
- TAFFC 같은 감정 컴퓨팅 저널에서는 CLISA와의 차이를 표로 보여 주는 것이 사실상 필수다.

### B-4. 생체신호의 '캘리브레이션 배포' 선례: 다중 스트림 AdaBN(Du 2017), 사용자별 DA-BN(Mazankiewicz 2020), BCM(Kwak 2023)

**정확히 무엇을 하나.**
- **Du 2017.**
  - "In the training phase, the statistics μi and σi for each source domain are calculated independently." 배치를 같은 세션 블록 M=10개로 나눈다.
  - 배포 때는 "the statistics of BatchNorm are updated with the unlabeled calibration data". 저자들은 캘리브레이션에 "only require a small and randomly selected subset of gestures"라고 썼다.
  - NinaPro LOSO에서는 26명으로 학습하고, 시험 피험자의 별도 시행(1, 3, 4, 5, 9번)으로 적응한 뒤 다른 시행(2, 6, 7, 8, 10번)으로 평가했다.
  - 다중 스트림의 이득은 일반 AdaBN 대비 +0.2~1.2%p로 작다. 세션이 많을수록 이득이 컸다.
- **Mazankiewicz 2020.** 학습 때 사용자별 배치로 통계를 내고, 배포 때는 창마다 지수이동평균으로 통계를 갱신한다. 캘리브레이션은 없다.
- **Kwak 2023 (초록 기준).** 1분 휴지기 기준 EEG로 깊은 특징의 피험자 변이 성분을 지우는 모듈(BCM)과 피험자 불변 손실을 함께 학습한다. 새 피험자는 1분 기준 신호만 있으면 된다. An 2024(MICCAI)는 BCM을 "refines the features extracted from TS EEG signals using subject characteristics obtained from RS EEG signals"라고 요약했다.

**같은 점 / 다른 점.**
- **Du 2017**
  - 같은 점: 배포 = 분리된 무라벨 캘리브레이션 데이터 통계로 정규화. 학습 = 같은 정규화를 그룹 단위 배치에서 수행. CAFT ①과 구도가 가장 같다.
  - 다른 점: 전 층 BN 평균·분산을 쓰고, 캘리브레이션 구성(클래스 균형)을 맞추지 않으며, 일관성 손실이 없고, sEMG다.
- **Kwak 2023**
  - 같은 점: 짧은 개인 기준 기록으로 깊은 특징에서 피험자 성분을 빼는 연산을 학습과 배포에 똑같이 쓴다. 피험자 간 일치 손실도 결합했다. ①+② 조합의 MI판이라 할 만하다.
  - 다른 점: 기준이 과제와 무관한 휴지기이고, 학습형 모듈이며, 일치가 클래스 수준이다.

**위치 잡기.**
- ARM 논문은 "the first difference is novel to ARM-BN"이라고 썼다. 하지만 생체신호에서는 Du 2017이 이미 같은 일을 했다. 두 논문을 함께 인용하면 계보를 정확히 아는 것으로 보인다.
- BCM은 원문을 확인한 뒤 ②와 함께 비교해야 한다. 차이는 세 가지다.
  1. 기준이 휴지기가 아니라 감정 자극 블록이다.
  2. 빼기가 학습 모듈이 아니라 닫힌 형태의 평균이다.
  3. 일치 단위가 클래스가 아니라 자극 시점이다.

### B-5. 원리 수준: SAT·CMVN, 그리고 TaskNorm/MetaBN

- **SAT(1996).**
  - 화자 변환과 표준 모델을 함께 추정해 "jointly annihilates the inter-speaker variation and estimates the HMM parameters of the SI acoustic models"한다. 그 결과 시험 화자에 "more efficiently adapted"된다.
  - CAFT ①은 화자 변환을 평행이동 하나(닫힌 형태: 블록 평균)로 줄인 SAT로 서술할 수 있다.
- **CMVN 관행.** 학습과 시험의 특징을 같은 방식으로 정규화하는 관행이다. Strand & Egeberg 2004는 이를 "train models on normalized features"라고 부르고, 구간 통계가 내용 맥락에 의존하는 문제를 지적했다.
- **Busso IFN(2013).** 감정 음성에서 정규화가 감정 차이를 지우지 않도록 중립 발화만으로 통계를 낸다.
- **세 문헌이 CAFT 설계에 주는 근거.** 각각 CAFT의 두 장치가 왜 필요한지를 뒷받침한다.
  - 내용 맥락 의존(CMVN) → 같은 자극 위치로 맞추기
  - 감정 편향(IFN) → 감정 균형 블록
- **TaskNorm/MetaBN.**
  - CAFT의 배포는 MetaBN과 같은 구조다. 컨텍스트 = 캘리브레이션 블록, 대상 = 시험 창이며 비전이적이다.
  - 반면 CAFT의 학습은 평균을 낸 창에 바로 손실을 걸어 '전이적 BN'에 가깝다.
  - TaskNorm은 학습과 시험 사이 전이성·클래스 균형이 어긋나면 실패한다고 보였다. 그러므로 MetaBN식 학습(평균용 창과 손실용 창 분리)을 대조 실험으로 넣으면 '같은 연산' 주장이 더 단단해진다((C) C-5).
  - 작은 컨텍스트에서는 인스턴스 통계와 섞는 TaskNorm이 낫다고 했다. 20초 캘리브레이션처럼 짧은 블록에서 수축 추정을 검토할 근거다.

### B-6. 이론: '그룹 평행이동 불변성을 구조로 얻는' 그룹 중심화

이번 범위에서는 이 주장을 딥러닝 설정에서 직접 정리한 논문을 찾지 못했다. 가까운 이론 조각은 다음과 같다.

- **(i) 정규형(canonicalization).** Kaba 2023은 정규형 사상 뒤 임의의 망을 써서 불변성을 구조로 얻는다.
- **(ii) 집단 평균 중심화.** 다수준 모형에서는 집단 평균 중심화로 집단 간 변동을 분리한다(Enders & Tofighi 2007). 계량경제학의 패널 고정효과 '집단 내 변환'도 같은 생각이다(Mundlak 1978, 서지만).
- **(iii) 위치-척도 조건부 이동 모형.** Zhang 2013의 LS-ConS가 이 모형을 다룬다.
- **(iv) 주변분포 전이 학습.** Blanchard 2011/2021은 f(P̂_X, x), 시험 표본 크기 n_T에 조건화한 위험을 다룬다. ARM의 이론 근거이기도 하다.

**우리가 쓸 수 있는 정리 (우리 유도이며 문헌 인용이 아님).**
- **모형.** 피험자 s의 특징을 z = φ(x) + b_s로 둔다. b_s는 그 피험자의 모든 창에 공통인 평행이동이다.
- **정리.** 같은 자극·같은 감정 구성의 블록 평균을 빼면 b_s가 정확히 사라진다. 즉 표현이 피험자 평행이동에 대해 구조적으로 불변이다.
- **조건.**
  1. 그룹 사이에 블록의 내용·감정 구성이 같아야 한다. 다르면 E_블록[φ]의 차이만큼 편향이 생긴다. 라벨 이동에서 재중심화가 해롭다는 SPDIM(A1·A2), Bakas의 불균형 붕괴, Busso의 중립 기준과 같은 문제다.
  2. 추정 분산은 블록 창 수 n에 대해 σ²/n으로 줄어든다(Blanchard의 n_T 의존과 같은 모양).
- **CAFT 설계와의 대응.**
  - 자극 위치 일치와 감정 균형은 조건 1을 학습과 배포 모두에서 맞추는 장치다.
  - ① 학습은 φ가 평행이동이 아닌 피험자 차이(회전·척도)를 덜 만들도록 미는 장치로 해석할 수 있다.
- **A4와의 연결.** Identity Trap은 미세조정이 피험자 분산을 +10~+63%p 키운다고 보고했다. 그래서 FM 전체 파인튜닝에서 학습 시점 중심화가 특히 필요하다는 동기로 쓸 수 있다.
- 이 정리는 짧은 명제로 논문에 넣을 수 있다. 다만 '새 이론'이라고 하기보다 위 문헌의 특수 경우로 서술하는 것이 안전하다.

---

## (C) 리뷰어가 요구할 기준선·대조 실험

필수(★)부터 적었다. 하이퍼파라미터(λ, M, K, 학습률)는 시험 피험자를 쓰지 않는 중첩 LOSO로 고르고, 16명 전체와 4개 데이터셋에서 시드를 3개 이상 돌린다. 창 정확도와 클립 정확도를 함께 보고한다(A2 권고와 같다).

1. **★ C-1. 'BN 적응 대 ARM-BN' 형식의 핵심 대조.**
   - 기준선: 학습은 보통(무작위 배치, 중심화 없음), 배포는 캘리브레이션 중심화.
   - CAFT ①: 학습 중심화 + 같은 배포.
   - 피험자 단위 짝 검정을 하고, 평균과 최악 피험자를 함께 보고한다.
   - 선행의 추가 이득은 +0.2~3%p였고 ARM-BN은 일부 데이터(FMoW)에서 악화했다. 그러므로 효과가 없거나 음수인 데이터셋도 그대로 보고한다.
2. **★ C-2. 무엇이 이득을 내는지 분해.** 그룹 크기와 손실 가중을 같게 두고 다음 세 가지를 비교한다.
   - (a) ARM-BN형: 피험자 단위 배치이지만 창을 무작위로 뽑음(감정 불균형 허용, 클립·초 무작위)
   - (b) 감정 균형이지만 자극 위치는 피험자마다 다름(내용 불일치 그룹 통계)
   - (c) CAFT: 균형 + 자극 일치
   - 이렇게 '그룹화 자체 / 감정 균형 / 자극 일치'를 나눈다. CLISA·계층화 정규화와의 차별점이 실제로 성능에 기여하는지 보이는 유일한 방법이다.
3. **★ C-3. 정규화의 형태와 위치.**
   - 비교할 형태:
     - 평균만
     - 평균+분산(차원별 z-점수: 계층화 정규화·잠재 정렬·CLISA형)
     - 여러 깊이(LA형: LaBraM 중간 블록 출력에도 피험자별 표준화를 넣음)
     - 입력 수준 EA를 학습·시험 모두에 적용(He & Wu 2020, Xu 2020)
   - 배포 연산도 똑같이 바꿔 '같은 연산' 원칙을 지킨 채 비교한다.
4. **★ C-4. 전이적 상한(오라클).** 시험 세션 전체 평균이나 시험 창 자체의 평균으로 중심화한다. 이것이 잠재 정렬·계층화 정규화·CLISA식이다. 그 차이를 '비전이적 캘리브레이션의 비용'으로 보고하고, 선행 수치와 비교할 때 프로토콜 차이를 명시한다.
5. **C-5. 문맥·대상 분리와 기울기 차단.**
   - (a) MetaBN식: 15곳 중 일부(예: 감정당 1곳)로 평균을 내고 나머지 창에만 CE를 건다.
   - (b) 자기 제외(leave-one-out) 평균
   - (c) 평균에 기울기 차단(stop-gradient)
   - Wu & Johnson의 '배치 안 정보 누설'과 TaskNorm의 전이적 학습 문제를 점검하는 대조다.
6. **C-6. 추정 잡음 일치.**
   - 학습 그룹 크기 M ∈ {5, 15, 30, 세션 전체}와 배포 캘리브레이션 길이 {20초, 40초, 전체}의 교차표를 만든다.
   - 수축 평균과 비교한다: Schneider의 의사표본 수 N, TaskNorm의 α, ComBat식 경험적 베이즈.
   - 'M을 배포 블록에 맞추는 것'이 실제로 중요한지 보인다(Wu & Johnson: 정규화 배치 크기가 학습 잡음과 학습-추론 불일치를 함께 좌우한다).
7. **★ C-7. 감정 불균형 스트레스.**
   - 캘리브레이션에서 감정 하나를 빼거나 감정별 길이를 다르게 한다. 학습 배치를 균형으로 했을 때와 무작위로 했을 때의 견고성을 비교한다. Bakas는 무작위 균형 학습의 이득이 유의하지 않았다고 보고했다.
   - 대안과도 비교한다: SPDIM류 라벨 이동 보정, Busso IFN식 '중립 클립만으로 중심화'(SEED-V에는 중립 감정이 있다).
8. **C-8. 빼기 대 문맥 조건화.** 캘리브레이션 평균을 빼는 대신 head 입력에 이어 붙인다(ARM-CML, Dubey 2021, i-vector식). 그러면 '빼기라는 귀납 편향'의 가치를 보일 수 있다.
9. **★ C-9. ② 관련.**
   - ①·② 2×2 표를 만든다.
   - ②의 형태를 비교한다: 양성만, InfoNCE(CLISA·CL-SSTER형), 클래스 수준 일치(BCM형).
   - λ 민감도, 중심화 전과 후 특징에 일관성을 거는 경우, 시간 허용 창(TA2CL)도 본다.
   - ②가 방향 일치도는 올리지만 정확도 이득이 들쭉날쭉하다는 중간 결과도 그대로 보고한다.
10. **C-10. 학습 시점 불변성 기준선.** ARM이 비교한 범주들이다.
    - 피험자 적대 학습(DANN)
    - 피험자 쌍 MMD·CORAL
    - 계층화 정규화 원판(입력+층별)
11. **C-11. 불변성 진단.**
    - 중심화 임베딩에서 피험자 식별 정확도(Fdez 방식)
    - 피험자 분산 비율(Identity Trap, A4)
    - ①이 파인튜닝으로 커지는 피험자 성분을 줄이는지 보인다.
12. **C-12. 일반성.**
    - 같은 ①을 다른 FM 하나(CBraMod 등)와 처음부터 학습한 작은 모델(EEGNet 등)에 적용한다.
    - 그러면 'FM 설정 특유의 효과'인지 '일반 원리의 재현'인지 구분된다. 잠재 정렬·계층화 정규화는 작은 모델이었다.
13. **C-13. DEAP 관련.**
    - 시행 전 기준 신호 빼기(Yang 2018) 관행과의 관계를 정리한다.
    - DEAP에서 같은 자극 위치 배치를 만들 수 있는지 확인한다. 영상 제시 순서가 피험자마다 같은지는 이번 조사에서 확인하지 못했다.

---

## (D) 확인하지 못한 항목

**본문 세부를 확인하지 못한 항목**
- **Kwak 2023 (BCM)**: 공개 원문을 찾지 못했다. 다음 세 가지는 초록과 An 2024의 한 줄 요약으로만 확인했다.
  - 빼기의 정확한 형태(학습형인지, 차감인지)
  - 피험자 불변 손실의 형태
  - 데이터셋과 수치
- **Li G. 2023 (KBS)**: 공개 원고 링크가 PDF 대신 HTML을 돌려줬다. '실험 단위 BN'의 학습·시험 통계 출처는 미확인이다.
- **Viikki & Laurila 1998**: 출판사 접근이 막혔다(403). 내용은 Strand & Egeberg 2004의 서술과 2차 검색 결과로만 확인했다.
- **Gales 1998, Mundlak 1978**: 서지만 확인했다. 내용 서술은 피했다.
- **Busso 2013, Saon 2013, Miao 2015, BEN, MetaNorm, MABN, FedBN, SiloBN, Sethu 2007, STEM(MICCAI 2026)**: 초록이나 학회 페이지 수준으로만 확인했다.
- **Fdez 2021**: 시험 참가자 정규화 통계의 출처는 논문에 없다(A2와 같은 판단). 표에는 '세션 전체로 보임'으로 적었다.
- **CLISA**: 예측 단계에서 인코더 안 계층화 정규화를 어떻게 처리하는지 본문에 없다. 학습 단계 사용만 확인했다.
- **Du 2017**: 표 7 열 머리의 대응(데이터셋·평가 방식)이 텍스트 추출에서 모호하다. 그래서 '+0.2~1.2%p'라는 범위로만 썼다.

**게재처를 확인하지 못한 항목**
- Nado 2020, Kaku 2020, You 2021, Wu & Johnson 2021: arXiv에서만 확인했다.

**커버리지 한계**
- 조사 중 arXiv API가 막혀(429), 2026년 하반기 arXiv 전용 EEG 논문을 체계적으로 훑지 못했다. 특히 다음 주제가 빠졌을 수 있다.
  - 피험자별 정규화로 학습한 EEG FM 파인튜닝
  - 캘리브레이션을 흉내 내는 에피소드 학습
- 웹 검색·OpenAlex로 보완했지만, 피험자별 배치 정규화는 논문마다 부르는 이름이 달라('stratified', 'subject-wise', 'domain-specific', 'latent alignment', 'personalized BN') 검색으로 놓쳤을 가능성이 남는다.
- 교차 피험자 EEG 서베이(Li T. 외 2026, arXiv 2604.27033)도 잠재 정렬과 계층화 정규화를 다루지 않았다. 이 계열이 서베이에 잘 정리되지 않은 상태로 보인다.
