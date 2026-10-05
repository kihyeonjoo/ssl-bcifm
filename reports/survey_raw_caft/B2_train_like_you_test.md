# B2. "시험(배포) 조건을 흉내 내며 학습한다" — 원리와 EEG/BCI 적용, CAFT 새로움 점검 (기준일 2026-10-05)

**검증 방법.**
- 모든 논문의 제목·저자·연도·게재처는 1차 출처로 확인했다. arXiv API(초록·주석), 학회 페이지(NeurIPS 논문집, PMLR, OpenReview, ISCA 아카이브, MICCAI 논문 페이지), Crossref·OpenAlex·PubMed(DOI·권호)를 썼다.
- 핵심 주장(인용문, 배치 구성, 수치)은 원문 PDF에서 직접 확인했다: Matching Networks, Prototypical Networks, ARM, TaskNorm, CBN, Kobler 2022, CLISA, mdJPT, GE2E, Meta-Baseline, QAT, SAT, Tao & Chen 2026, An 2024(ResTL — Kwak 2023의 기저선 보정 모듈 서술을 재확인하는 데 씀).
- 초록만 본 것은 표에 "(초록 기준)"으로 적었다. 2차 서술만 본 것은 그렇게 밝혔다.
- 웹 검색은 약 25회 썼다. arXiv API가 한때 429(요청 한도)를 내서 재시도했고, DBLP는 봇 차단으로 쓰지 못했다.
- 기존 조사 A1~A5(`reports/survey_raw/`)와 겹치는 논문은 거기서 확인한 서지를 재사용했고, 행에 "(A1 재사용)"처럼 표시했다.

**새로움 위협 판정 기준.** CAFT ①의 "학습-배포 일치" 설계와 ②에 대한 위협을 셋으로 나눴다.
- **높음**: 리뷰어가 "CAFT ①은 X를 Y에 옮긴 것"이라고 말할 수 있는 경우다. 학습 때 도메인(사람·카메라·과제) 단위 묶음으로 계산한 고정 통계로 정규화하고, 배포 때 새 도메인의 표본 통계로 같은 연산을 한다(새 파라미터 없음, 배포 경사하강 없음). 또는 CAFT의 배치 구성과 배치 안 피험자별 정규화를 이미 함께 썼다.
- **중간**: 원리는 같지만 다음 중 하나에 해당한다. 적응이 학습되는 모듈·파라미터다. 사전학습 같은 다른 단계에서 한다. 시험 데이터로 통계를 낸다. 분야가 멀어 직접 비교는 어렵지만 인용을 요구받을 만하다.
- **낮음**: 원리의 원전·명명 선례이거나, 배포 때 경사하강을 쓰는 메타학습이다. 인용은 필요하지만 CAFT의 방법 새로움을 직접 위협하지는 않는다.

**약어** (본문에서도 처음 나올 때 풀어 쓴다)
- 일반 기계학습: 배치 정규화(BN), 도메인 일반화(DG), 시험 시점 학습(TTT), 모델 무관 메타학습(MAML), 적응 위험 최소화(ARM), 메타 배치 정규화(MetaBN), 전이적 배치 정규화(TBN), 맥락 메타학습(CML), 양자화 인지 학습(QAT), 특징별 선형 변조(FiLM), 저랭크 적응(LoRA)
- EEG·과제: 파운데이션 모델(FM), 운동 상상(MI), 사건 관련 전위(ERP), 정상 상태 시각 유발 전위(SSVEP), 피험자 하나 빼기 교차검증(LOSO), 미분 엔트로피(DE), 유클리드 정렬(EA), 대칭 양의 정부호 행렬(SPD), 기저선 보정 모듈(BCM), 자극 시점 정렬(SLA)
- 분야 밖: 카메라 기반 배치 정규화(CBN), 인간 활동 인식(HAR), 화자 적응 학습(SAT), 반복 특징 정규화(IFN), 일반화 종단 간 손실(GE2E), 튜플 기반 종단 간 손실(TE2E)

---

## 0. 핵심 결론

- **원리 자체는 오래됐다.** 다음 연구들이 모두 같은 원리를 각자의 단계에 적용한다. 그래서 CAFT는 원리의 새로움을 주장할 수 없고, 이들을 원전으로 인용해야 한다.
  - Matching Networks(2016)는 학습 절차를 *"test and train conditions must match"*라는 원칙 위에 세웠다.
  - Prototypical Networks(2017)는 학습 shot을 시험 shot에 맞추는 것이 대체로 가장 좋다고 보고했다.
  - 모델 무관 메타학습(MAML), MLDG·에피소드 도메인 일반화(DG), 시험 시점 학습(TTT) 계열
  - "사전학습처럼 파인튜닝"하는 FLYP, 양자화 인지 학습(QAT)
- **① 의 메커니즘은 일반 기계학습에 이미 있다.** 메커니즘이란 "학습 때 도메인별 배치 통계로 고정 정규화하고, 배포 때 새 도메인의 소수 표본으로 같은 연산을 한다(새 파라미터 없음, 배포 경사하강 없음)"는 것이다. 예는 셋이다.
  - **ARM-BN**(적응 위험 최소화(ARM), NeurIPS 2021): 한 도메인에서 뽑은 배치로 학습하고, 시험 배치로 통계를 다시 계산한다.
  - **TaskNorm의 MetaBN**(메타 배치 정규화(MetaBN), ICML 2020): 작은 컨텍스트 집합의 통계로 컨텍스트와 대상을 모두 정규화한다. 메타 학습과 메타 시험에 똑같이 적용하고, 비전이적이다.
  - **카메라 기반 BN**(CBN, ECCV 2020): 새 카메라의 라벨 없는 소수 표본으로 통계를 추정한다.

  "이전 연구의 사후 중심화 → CAFT"의 관계는 ARM 논문의 "BN 적응 → ARM-BN"과 정확히 같다.
- **EEG 안에서도 요소별 선행이 있다.**
  - 층화 정규화(Fdez 2021, SEED 감정): 참가자·세션별 z-점수로 정규화하며 학습한다.
  - **CLISA(TAFFC 2023)**: 미니배치를 "피험자 2명 × 시행마다 같은 시점"으로 짜고, 배치 안에서 피험자별 z-점수 정규화를 하며, 같은 자극 구간을 양성쌍으로 대조학습한다. CAFT의 배치 구성·①·②가 한 논문에 다 있다.
  - Kobler(NeurIPS 2022): 학습 중 변하는 잠재 공간의 도메인 평균을 추적하는 도메인별 BN이다.
  - **Kwak(JBHI 2023)**: 새 피험자의 1분 휴지기 EEG로 깊은 특징의 피험자 성분을 빼고, 학습도 같은 연산으로 한다. 같은 클래스를 피험자와 무관하게 모으는 손실도 더한다.
- **"CAFT와 거의 같은 일"을 한 단일 논문은 찾지 못했다.** 찾지 못한 조합은 다음 다섯 가지를 모두 갖춘 것이다. 가장 가까운 것은 EEG 안에서 CLISA와 Kwak 2023, 원리·연산 면에서 ARM-BN과 MetaBN이다. 따라서 주장은 "새 원리"가 아니라 **"ARM-BN·MetaBN·층화 정규화 계열의 원리를, EEG FM의 비전이적 캘리브레이션 블록 배포에 맞춰 설계하고 감정 LOSO로 검증"**으로 한정해야 한다.
  - EEG 파운데이션 모델(FM) 전체 파인튜닝
  - 분류 토큰 평균 빼기 하나만 쓰는 적응
  - 배포 캘리브레이션 블록과 같은 구성의 에피소드(같은 세션, 감정 균형, 피험자 간 같은 (클립, 초))
  - 테스트와 분리된 짧은 블록으로 하는 비전이적 배포
  - 감정 LOSO(피험자 하나 빼기 교차검증)
- **EEG FM에서 "캘리브레이션 문맥을 쓰도록 학습"한 가장 가까운 시험은 무효과였다.**
  - Tao & Chen(arXiv, 2026-09)은 CBraMod에 "문맥 모델"을 붙였다. 휴지기 또는 앞쪽 시행 특징으로 FiLM·LoRA를 생성하고, 학습 피험자 에피소드로 학습하며, 배포 때 경사하강이 없다.
  - 결과(운동 상상): 진짜 문맥과 섞은 문맥의 차이가 중앙값 0.00%p였고, 메타학습 초기화도 뚜렷한 이점이 없었다.
  - 그래서 "EEG FM을 캘리브레이션 모양 에피소드로 학습한 첫 사례"라고는 쓸 수 없다.
  - 대신 "학습되는 문맥 조건화가 이득을 못 낸 곳에서, 파라미터 없는 고정 연산이 이득을 냈다"는 대비가 기여가 될 수 있다. 다만 같은 대조군(섞은·교환 캘리브레이션, 모집단 학습량 맞춤)을 갖춰야 한다.
- **이름 충돌이 있다.**
  - "Calibration-Aware Fine-Tuning"은 ICML 2025 LLM 확률 보정 논문(Xiao 외, 약칭 CFT)이 이미 쓴다.
  - "CAFT" 약칭은 2025년 LLM 논문 2편(Concept-Aware Fine-Tuning, Concept Ablation Fine-Tuning)이 쓴다.
  - "Fine-Tune Like You Calibrate"라는 제목은 찾지 못했다. FLYP를 본뜬 것으로 읽힌다.
  - 권장: 초록 첫 문장에서 calibration이 "새 사용자 캘리브레이션 데이터"임을 밝히고, 방법 이름·약칭은 바꾼다.

---

## (A) 논문 표

본표는 35편(A-1~A-5)이다. 이름 충돌 5편은 A-6에 따로 적었다.

### A-1. 원리: 학습 조건을 시험 조건에 맞춘다 (메타학습·에피소드 학습·시험 시점 적응)

| 논문 (첫 저자, 연도, 학회/저널) | 링크 | 무엇을 하나 (1~2문장) | CAFT 와의 관계 (같은 점 / 다른 점) | 새로움 위협 |
|---|---|---|---|---|
| 1. Vinyals, 2016, NeurIPS — "Matching Networks for One Shot Learning" | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2016/hash/90e1357833654983612fb05e3ec9148c-Abstract.html), [arXiv 1606.04080](https://arxiv.org/abs/1606.04080) | 지원 집합에 대한 주의(attention) 기반 최근접 분류다. 학습 절차를 *"a simple machine learning principle: test and train conditions must match"*에 두고, 미니배치마다 과제를 바꿔 클래스당 몇 개 예시만 보여 준다(원문 확인). | **같은 점**: CAFT 원리의 원전이므로 인용이 필수다. **다른 점**: 새 클래스 소수샷이 목표이고, 피험자 이동이나 중심화가 없다. | 낮음 |
| 2. Snell, 2017, NeurIPS — "Prototypical Networks for Few-shot Learning" | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html), [arXiv 1703.05175](https://arxiv.org/abs/1703.05175) | 지원 임베딩 평균을 클래스 원형으로 삼아 거리로 분류하며, 에피소드로 학습한다. *"advantageous to match the training-shot with the test-shot"*, 그리고 학습 way는 시험보다 크게 하라고 보고했다(원문 확인). | **같은 점**: 에피소드 모양을 배포에 맞추고 평균 임베딩을 쓴다. CAFT 배포의 "학습 피험자 원형 + 코사인"도 이 계열이다. **다른 점**: 평균은 클래스 원형이지 도메인 평균 빼기가 아니다. CAFT는 학습 때 원형 손실이 아니라 선형 head 교차 엔트로피를 쓴다. | 낮음 |
| 3. Finn, 2017, ICML — "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (1차 근사 변형 Reptile: Nichol 2018, [arXiv 1803.02999](https://arxiv.org/abs/1803.02999)) | [PMLR v70](https://proceedings.mlr.press/v70/finn17a.html) | 몇 번의 경사 갱신 뒤 성능이 좋도록 초기값을 메타학습한다. 내부 루프가 곧 "배포 적응을 학습에서 흉내 낸 것"이다. | **같은 점**: 배포 적응 절차를 학습 루프 안에 넣는다. **다른 점**: 적응이 경사하강이라 배포 때도 갱신이 필요하다. CAFT는 학습되지 않는 평균 빼기이고 배포 경사하강이 없다. | 낮음 |
| 4. Li D., 2018, AAAI — "Learning to Generalize: Meta-Learning for Domain Generalization" (MLDG) | [DOI 10.1609/aaai.v32i1.11596](https://doi.org/10.1609/aaai.v32i1.11596) | 각 미니배치 안에 가상 시험 도메인을 만들어 학습·시험 도메인 이동을 흉내 낸다. 학습 도메인을 개선하는 단계가 가상 시험 도메인도 개선하도록 메타 최적화한다(초록 기준). | **같은 점**: 배치를 도메인 구조로 짜서 "새 도메인"을 흉내 낸다. **다른 점**: 새 도메인 데이터를 배포 때 전혀 쓰지 않는 DG이고, 적응 연산이 없다. | 낮음 |
| 5. Li D., 2019, ICCV — "Episodic Training for Domain Generalization" | [DOI 10.1109/ICCV.2019.00153](https://doi.org/10.1109/ICCV.2019.00153), [arXiv 1902.00113](https://arxiv.org/abs/1902.00113) | 특징 추출기와 분류기를 "현재 도메인에 맞지 않게 학습된 짝"과 상호작용시켜, 실행 시 새 도메인 이동에 노출되도록 에피소드로 학습한다(초록 기준). | **같은 점**: 에피소드로 시험 조건을 흉내 낸다. **다른 점**: 배포 캘리브레이션이 없고, 학습 때만 쓰는 도메인별 보조 모듈이 있다. | 낮음 |
| 6. Chen Y., 2021, ICCV — "Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning" | [DOI 10.1109/ICCV48922.2021.00893](https://doi.org/10.1109/ICCV48922.2021.00893), [arXiv 2003.04390](https://arxiv.org/abs/2003.04390) | 전체 분류로 수렴한 모델(시험 때 지원 평균 원형 + 코사인 최근접)을 바로 그 평가 방식으로 에피소드 메타학습하면 좋아진다. 두 목적 사이의 상충도 보고했다(원문 확인). | **같은 점**: "표준 학습 후 배포 방식으로 다시 파인튜닝"이라는 구조가 CAFT(표준 LOSO 파인튜닝 → 배포 중심화를 흉내)와 같다. **다른 점**: 도메인 중심화가 없고 새 클래스가 대상이다. **시사점**: CAFT도 학습 손실을 배포 분류기(원형 코사인)에 맞춘 변형을 시험해야 한다. | 중간 |
| 7. Bronskill, 2020, ICML — "TaskNorm: Rethinking Batch Normalization for Meta-Learning" | [PMLR v119](https://proceedings.mlr.press/v119/bronskill20a.html), [arXiv 2003.03284](https://arxiv.org/abs/2003.03284) | 정규화 통계를 과제 수준 변수로 본다. MetaBN은 *"the context set alone is used to compute the normalization statistics for both the context and target sets, both at meta-train and meta-test time"*이다(원문). 대상 표본끼리 통계를 섞는 전이적 BN(TBN)을 비판하고, 작은 컨텍스트에서는 표본 자체 통계와 섞는 TaskNorm을 제안한다. | **같은 점**: "작고 클래스 균형 잡힌 보정용 집합의 통계로 고정 정규화, 학습·배포 동일, 새 파라미터 없음, 배포 경사하강 없음" 조합이 이미 있다. **다른 점**: 도메인이 아니라 새 클래스 소수샷 과제이고, 모든 BN 층의 평균·분산을 쓴다. CAFT는 분류 토큰 평균만 빼고, 클래스는 고정이며 피험자 이동을 다룬다. 또 CAFT 학습은 분류 대상 15창이 평균에 들어가 TBN 쪽이고, 배포는 MetaBN 쪽(비전이적)이라 둘이 완전히 같지 않다. | **높음** |
| 8. Zhang M., 2021, NeurIPS — "Adaptive Risk Minimization: Learning to Adapt to Domain Shift" (ARM) | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html), [arXiv 2007.02931](https://arxiv.org/abs/2007.02931) | 도메인별 묶음으로 "라벨 없는 배치로 적응 → 예측"을 메타학습한다. ARM-BN은 *"the training batches are sampled from a single domain … the normalization statistics are recomputed at test time"*이다. ARM-CML(맥락 메타학습)은 배치 평균 문맥 벡터를 입력에 붙인다(원문). ARM-BN은 이미지 4개 과제 평균에서 BN 적응만 한 경우보다 좋았지만(예: FEMNIST 80.0→83.2), WILDS FMoW에서는 오히려 나빴다(평균 51.6→42.0, 표준 학습 53.0). | **같은 점**: "사후 BN 적응 → ARM-BN"이 "사후 중심화 → CAFT"와 같은 관계다. 새 파라미터가 없고, 배포 경사하강이 없으며, 평균을 통해 기울기가 흐른다. **다른 점**: 배포 통계가 시험 배치 자체라 전이적이다. BN 전 층의 평균·분산을 쓰고, 배치는 "한 도메인의 무작위 표본"이지 감정 균형·자극 정렬이 아니다. 이미지 과제다. | **높음** |
| 9. Sun Y., 2020, ICML — "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts" (TTT) | [PMLR v119](https://proceedings.mlr.press/v119/sun20b.html), [arXiv 1909.13231](https://arxiv.org/abs/1909.13231) | 라벨 없는 시험 표본 하나를 자기지도 문제로 바꿔, 예측 전에 모델 파라미터를 갱신한다(초록 기준). | **같은 점**: 시험 시점 적응을 전제로 학습을 설계한다. **다른 점**: 시험 데이터를 쓰고 시험 때 경사하강을 한다. CAFT는 둘 다 없다. | 낮음 |
| 10. Bartler, 2022, AISTATS — "MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption" (같은 계열: Meta-TTT, Tao C. 2024, [arXiv 2410.01709](https://arxiv.org/abs/2410.01709) / Learning to (Learn at Test Time), Sun 2023, [arXiv 2310.13807](https://arxiv.org/abs/2310.13807) / MABN, Wu 2024 AAAI, [DOI](https://doi.org/10.1609/aaai.v38i14.29527)) | [PMLR v151](https://proceedings.mlr.press/v151/bartler22a.html), [arXiv 2103.16201](https://arxiv.org/abs/2103.16201) | 메타학습, 자기지도, TTT를 결합한다. 자기지도 손실로 이미지 한 장에 적응한 뒤의 성능이 좋도록 메타 모델을 학습한다(초록 기준). Meta-TTT는 BN 층의 TTT를 최소최대 메타학습으로 하고, 시험 배치와 원천 통계를 섞는다. | **같은 점**: "시험 시점 적응을 학습에서 흉내 낸다". 사용자가 물은 "적응을 고려한 사전학습"의 실제 이름들이 이것이다. **다른 점**: 배포 경사하강과 학습된 적응을 쓰고, 시험 데이터로 적응한다. | 낮음 |

### A-2. 파인튜닝을 다른 단계의 조건에 맞추기 (명명·원리 선례)

| 논문 (첫 저자, 연도, 학회/저널) | 링크 | 무엇을 하나 (1~2문장) | CAFT 와의 관계 (같은 점 / 다른 점) | 새로움 위협 |
|---|---|---|---|---|
| 11. Goyal, 2023, CVPR — "Finetune like you pretrain: Improved finetuning of zero-shot vision models" (FLYP) | [DOI 10.1109/CVPR52729.2023.01853](https://doi.org/10.1109/CVPR52729.2023.01853), [arXiv 2212.00638](https://arxiv.org/abs/2212.00638) | CLIP 파인튜닝을 사전학습과 같은 대조 손실(클래스 이름 프롬프트와 이미지)로 계속하면, 분포 안(ID)·분포 밖(OOD) 모두 표준 파인튜닝보다 좋다(초록 기준). | **같은 점**: 가제 "Fine-Tune Like You Calibrate"의 명명 원전이다. "파인튜닝 목적을 다른 단계와 맞춘다"는 점도 같다. **다른 점**: FLYP는 앞 단계(사전학습)에 맞추고, CAFT는 뒤 단계(배포 캘리브레이션)에 맞춘다. | 낮음 |
| 12. Jacob, 2018, CVPR — "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (QAT) | [DOI 10.1109/CVPR.2018.00286](https://doi.org/10.1109/CVPR.2018.00286), [arXiv 1712.05877](https://arxiv.org/abs/1712.05877) | 추론 때의 정수 양자화를 학습 순전파 안에서 모사하고, 역전파는 그대로 한다(*"simulate quantization effects in the forward pass of training"*, 원문). | **같은 점**: "X-aware training"이라는 이름과 "배포 연산을 순전파에 넣고 그대로 역전파"하는 구조가 CAFT ①과 같다. **다른 점**: 모델 압축 분야이고 개인 적응이 아니다. | 낮음 |
| 13. Touvron, 2019, NeurIPS — "Fixing the train-test resolution discrepancy" (FixRes) (관련: Batch Renormalization, Ioffe 2017 [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/hash/c54e7837e0cd0ced286cb5995327d1ab-Abstract.html) / Scheduled Sampling, Bengio 2015 [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)) | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/hash/d03a857a23b5285736c4d55e0bb067c8-Abstract.html), [arXiv 1906.06423](https://arxiv.org/abs/1906.06423) | 데이터 증강 때문에 생긴 학습·시험 해상도 불일치를, 시험 해상도에서 짧게 파인튜닝해 없앤다(초록 기준). Batch Renorm은 배치 통계에 의존하는 정규화의 학습·추론 불일치를 다룬다. | **같은 점**: 학습·시험 불일치를 파인튜닝으로 없앤다. 배치 통계 의존 연산의 학습·추론 차이라는 쟁점도 같다. **다른 점**: 불일치의 원인이 전처리·배치 크기이지 사용자 캘리브레이션이 아니다. | 낮음 |

### A-3. 도메인별 정규화로 "배포 적응"을 학습에서 흉내 (분야 밖)

| 논문 (첫 저자, 연도, 학회/저널) | 링크 | 무엇을 하나 (1~2문장) | CAFT 와의 관계 (같은 점 / 다른 점) | 새로움 위협 |
|---|---|---|---|---|
| 14. Chang, 2019, CVPR — "Domain-Specific Batch Normalization for Unsupervised Domain Adaptation" (DSBN) | [DOI 10.1109/CVPR.2019.00753](https://doi.org/10.1109/CVPR.2019.00753), [arXiv 1906.03950](https://arxiv.org/abs/1906.03950) | 도메인마다 별도 BN 층을 두고 나머지 가중치는 공유한다. 대상 도메인 의사 라벨을 쓰는 2단계 비지도 도메인 적응이다(초록 기준). | **같은 점**: 도메인별 통계로 정규화하며 학습한다. Kobler 2022가 이를 EEG로 확장했다. **다른 점**: 대상 도메인 학습 데이터 전체를 쓰고(전이적), 도메인별 affine 파라미터가 있으며, 새 사용자의 짧은 보정 시나리오가 아니다. | 중간 |
| 15. Zhuang, 2020, ECCV — "Rethinking the Distribution Gap of Person Re-identification with Camera-based Batch Normalization" (CBN) | [DOI 10.1007/978-3-030-58610-2_9](https://doi.org/10.1007/978-3-030-58610-2_9), [arXiv 2001.08680](https://arxiv.org/abs/2001.08680) | *"In training, CBN disassembles each mini-batch and standardizes the corresponding input according to its camera labels. In testing, CBN utilizes few samples to approximate the BN statistics of every testing camera"*(원문). 미니배치 10개 정도로도 충분하다고 보고했다. | **같은 점**: "도메인 = 카메라"를 "피험자"로 바꾸면 CAFT ①과 거의 같은 설계다. 배치 안 도메인별 통계, 고정 연산, 새 파라미터 없음, 배포 때 새 도메인의 라벨 없는 소수 보정 표본, 경사하강 없음. **다른 점**: 모든 BN 층의 평균·분산을 쓰고, 에피소드가 클래스 균형·자극 정렬이 아니며, 사람 재식별 과제다. | **높음** |
| 16. Mazankiewicz, 2020, Proc. ACM IMWUT — "Incremental Real-Time Personalization in Human Activity Recognition Using Domain Adaptive Batch Normalization" | [DOI 10.1145/3432230](https://doi.org/10.1145/3432230), [arXiv 2005.12178](https://arxiv.org/abs/2005.12178) | 층 입력을 사용자별 평균·분산으로 정규화한다. *"During training, these statistics are computed over user-specific batches. In the online phase, they are estimated incrementally for any new target user"*(초록). | **같은 점**: 사용자를 도메인으로 보고, 학습 배치를 사용자별로 짜서 배포 정규화를 흉내 낸다. 라벨이 필요 없다. **다른 점**: 배포 통계를 시험 스트림에서 온라인으로 추정하고(별도 보정 블록 아님), 분산까지 정규화하며, 웨어러블 인간 활동 인식(HAR)이다. | 중간 |

### A-4. 분야 밖 "사용자 캘리브레이션(등록)을 학습에서 흉내" (② 원형 포함)

| 논문 (첫 저자, 연도, 학회/저널) | 링크 | 무엇을 하나 (1~2문장) | CAFT 와의 관계 (같은 점 / 다른 점) | 새로움 위협 |
|---|---|---|---|---|
| 17. Anastasakos, 1996, ICSLP — "A compact model for speaker-adaptive training" (SAT) | [DOI 10.21437/ICSLP.1996-253](https://doi.org/10.21437/ICSLP.1996-253) | 학습 화자마다 화자 변환(MLLR)을 공통("compact") 모델과 함께 추정한다. 이 변환은 시험 화자를 적응시키는 방식과 같은 방식으로 화자 차이를 떼어낸다(원문 확인). | **같은 점**: "배포 때 할 화자(피험자) 적응을 학습 화자에게도 똑같이 적용하며 공통 모델을 학습"하는 원리의 고전이다. **다른 점**: 화자별 변환이 추정되는 파라미터이고, 새 화자는 적응 데이터로 변환을 최적화해야 한다. | 중간 |
| 18. Wan, 2018, ICASSP — "Generalized End-to-End Loss for Speaker Verification" (GE2E) (선행 TE2E: Heigold 2016 ICASSP, [DOI 10.1109/ICASSP.2016.7472652](https://doi.org/10.1109/ICASSP.2016.7472652)) | [DOI 10.1109/ICASSP.2018.8462665](https://doi.org/10.1109/ICASSP.2018.8462665), [arXiv 1710.10467](https://arxiv.org/abs/1710.10467) | 배치를 "N명 화자 × M 발화"로 짜서 배치 안 화자 중심(등록 음성 지문)을 계산하고 코사인 유사도로 학습한다. 선행 TE2E는 *"simulates the two-stage process of runtime enrollment and verification during training"*이다. 참 화자 중심을 계산할 때 자기 발화를 빼야 학습이 안정되고 자명해를 피한다고 보고했다(원문). | **같은 점**: "K명 × M개" 배치, 현재 모델로 계산한 배치 안 사람별 평균, 배포 등록 절차 흉내까지 CAFT의 배치 설계와 구조가 같다. **다른 점**: 사람별 평균이 신호(화자 식별)이고, CAFT에서는 빼야 할 잡음이다. **시사점**: 자기 제외 중심은 CAFT의 자기 포함 평균 문제(C절)에 그대로 적용된다. | 중간 |
| 19. Busso, 2013, IEEE TAFFC — "Iterative Feature Normalization Scheme for Automatic Emotion Detection from Speech" (IFN) | [DOI 10.1109/T-AFFC.2013.26](https://doi.org/10.1109/T-AFFC.2013.26) | 화자마다 중립 발화를 반복 검출하고, 그 부분으로 정규화 파라미터를 추정해 중립·감정 발화 모두에 같은 아핀 변환을 적용한다. 그렇게 정규화한 특징으로 학습·시험한다(초록 기준). | **같은 점**: 감정 인식에서 "사람별 기준 통계로 정규화한 특징으로 학습하고 시험도 똑같이 한다". **다른 점**: 음성이고, 기준이 중립 발화뿐(감정 균형 아님)이며, 반복 검출을 쓰고, 딥러닝 파인튜닝이 아니다. | 중간 |
| 20. Liu G., 2021, IEEE TPAMI — "A Differential Approach for Gaze Estimation" (같은 문제의 메타학습 접근: FAZE, Park 2019 ICCV, [DOI 10.1109/ICCV.2019.00946](https://doi.org/10.1109/ICCV.2019.00946)) | [DOI 10.1109/TPAMI.2019.2957373](https://doi.org/10.1109/TPAMI.2019.2957373), [arXiv 1904.09459](https://arxiv.org/abs/1904.09459) | 같은 피험자의 두 눈 영상 사이 시선 차이를 예측하도록 학습한다. 배포 때 피험자별 캘리브레이션 영상을 기준으로 새 영상의 시선을 추정하며, 캘리브레이션 1장으로도 기존 방법에 개인 적응을 더한 것보다 좋다고 보고했다(초록). FAZE는 메타학습으로 학습한 시선 추정기를 9장 이하(3장부터 효과) 보정 표본으로 사람별 적응시킨다(초록). | **같은 점**: "사용자 캘리브레이션 기준과의 차이"를 학습 단계부터 다룬다. 배포 경사하강이 없다. **다른 점**: 평균 빼기가 아니라 쌍 차분이고, 회귀이며, 시선 추정이다. | 중간 |
| 21. Mahajan, 2021, ICML — "Domain Generalization using Causal Matching" (MatchDG) | [PMLR v139](https://proceedings.mlr.press/v139/mahajan21b.html), [arXiv 2006.07500](https://arxiv.org/abs/2006.07500) | 서로 다른 도메인에서 같은 객체로 대응되는 표본 쌍의 표현을 가깝게 하는 정합 손실로 DG를 한다. | **같은 점**: CAFT ②(같은 (클립, 초)에서 다른 피험자의 표현을 가깝게)의 분야 밖 원형이다. **다른 점**: 배포 캘리브레이션이 없고, 대응 쌍을 학습으로 추정한다. | 낮음 |

### A-5. EEG/BCI: 피험자별 정규화로 학습, 그리고 메타학습·에피소드로 캘리브레이션 준비

| 논문 (첫 저자, 연도, 학회/저널) | 링크 | 무엇을 하나 (1~2문장) | CAFT 와의 관계 (같은 점 / 다른 점) | 새로움 위협 |
|---|---|---|---|---|
| 22. He H., 2020, IEEE TBME — "Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach" (유클리드 정렬(EA)) (리만 재중심화: Zanini 2018 TBME, [DOI](https://doi.org/10.1109/TBME.2017.2742541) / "EA 이득은 곧 재중심화": Lopes 2026, [arXiv 2606.16462](https://arxiv.org/abs/2606.16462) — A1 재사용) | [DOI 10.1109/TBME.2019.2913914](https://doi.org/10.1109/TBME.2019.2913914) | 피험자(세션)마다 평균 공분산으로 시행을 백색화해, 학습 피험자 전원과 새 피험자를 같은 기준으로 맞춘다. | **같은 점**: BCI에서 "배포 때 할 피험자별 재중심화를 학습 피험자 전원에게도 미리 적용하고 학습"하는 원리의 표준 사례다. **다른 점**: 입력(공분산) 수준의 고정 전처리이고, 피험자 데이터 전체로 통계를 한 번 계산한다. 파인튜닝 중 특징이 변해서 배치 안에서 다시 계산해야 하는 상황이 아니다. | 중간 |
| 23. Fdez, 2021, Front. Neurosci. — "Cross-Subject EEG-Based Emotion Recognition Through Neural Networks With Stratified Normalization" (A3 재사용) | [DOI 10.3389/fnins.2021.626277](https://doi.org/10.3389/fnins.2021.626277) | 신경망 처음 세 층에서 특징을 참가자·세션별 z-점수(세션 15시행의 평균·표준편차)로 정규화하며 학습한다. 학습 데이터 전체를 한 배치로 넣는다. SEED LOSO 3클래스 79.6%로 BN보다 크게 높았다(본문 확인). | **같은 점**: 감정 EEG(SEED)에서 "피험자(세션)별 통계로 정규화하며 학습"한다. 통계 단위(감정이 균형 잡힌 한 세션)도 CAFT 에피소드와 비슷하다. **다른 점**: 여러 층 z-점수이고 소형 망이다. 시험 피험자의 통계를 어떻게 계산했는지 본문에 명시가 없다(A3는 시험 세션 자체 통계, 즉 전이적이라고 판단). 배포 캘리브레이션 블록 개념이 없다. | **높음** |
| 24. Shen X., 2023, IEEE TAFFC — "Contrastive Learning of Subject-Invariant EEG Representations for Cross-Subject Emotion Recognition" (CLISA) (A3·A5 재사용) | [DOI 10.1109/TAFFC.2022.3164516](https://doi.org/10.1109/TAFFC.2022.3164516), [arXiv 2109.09559](https://arxiv.org/abs/2109.09559) | 미니배치를 "피험자 A·B 2명 × 시행마다 같은 시점 1구간"으로 짜서 같은 자극 구간을 양성쌍으로 대조학습한다. 이때 *"we concatenated the same channel of different samples from one subject in the minibatch together and conducted z-score normalization"*(층화 정규화)을 입력·중간 층에 적용한다(원문). 예측 단계의 미분 엔트로피(DE) 특징은 학습 통계로 시작해 시험 데이터로 갱신하는 온라인 정규화를 쓴다. | **같은 점**: CAFT의 배치 구성(여러 피험자 × 정렬된 자극 위치), ① 배치 안 피험자별 정규화, ② 사람 간 같은 자극 정렬이 한 논문에 다 있다. **다른 점**: 감정 라벨 교차 엔트로피가 아니라 자기지도 사전학습 단계이고, FM이 아닌 소형 CNN이며, 피험자는 2명이다. 여러 층 z-점수(CAFT는 분류 토큰 평균만)를 쓰고, 음성쌍이 있는 InfoNCE를 쓰며, 배포 정규화를 시험 스트림으로 갱신한다(전이적). "배포 캘리브레이션을 흉내 낸다"는 동기도 없다. | **높음** |
| 25. Kobler, 2022, NeurIPS — "SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG" (A1 재사용, 학회 확인) | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/28ef7ee7cd3e03093acc39e1272411b7-Abstract-Conference.html), [arXiv 2206.01323](https://arxiv.org/abs/2206.01323) | 대칭 양의 정부호(SPD) 다양체 위의 도메인별 모멘텀 BN이다. 미니배치를 도메인별 부분 배치로 구성하고, *"tracking the domains' Fréchet means in latent space as they are changing during training"*으로 종단 학습한다. 새 도메인은 라벨 없는 데이터로 통계를 추정한다(오프라인 평가에서는 도메인 전체 데이터, 원문). | **같은 점**: EEG에서 "잠재 특징의 도메인별 재중심화를 배치로 계산하며 종단 학습하고, 배포도 같은 연산으로 하며, 경사하강이 없다". CAFT ①의 "현재 모델로 평균을 계산한다"는 쟁점을 명시적으로 다룬다. **다른 점**: 공분산(SPD) 재중심화에 분산까지 쓰고, 소형 TSMNet이며, 운동 상상 위주다. 대상 도메인의 무라벨 데이터 전체를 쓰고(전이적), 에피소드가 캘리브레이션 블록 모양이 아니다. | **높음** |
| 26. Kwak, 2023, IEEE JBHI — "Subject-Invariant Deep Neural Networks Based on Baseline Correction for EEG Motor Imagery BCI" | [DOI 10.1109/JBHI.2023.3238421](https://doi.org/10.1109/JBHI.2023.3238421) (PubMed 37022076) | 휴지기(기저선) EEG를 짝 입력으로 써서 깊은 특징에서 피험자 변이 성분을 빼는 기저선 보정 모듈(BCM)을 학습한다. 같은 클래스를 피험자와 무관하게 모으는 피험자 불변 손실도 더한다. *"Using 1-min baseline-EEG signals of the new subject, our algorithm can eliminate subject-variant components from test data without the calibration process"*(초록). | **같은 점**: 배포 때 새 사용자의 짧은 별도 기록으로 구한 피험자 성분을 특징에서 빼고, 학습도 같은 연산으로 하며, 배포 경사하강이 없다. 여기에 사람 간 불변 손실(②와 유사)까지 있어, CAFT ①+② 구조와 가장 비슷한 EEG 선행이다. **다른 점**: 기준이 휴지기(감정·자극 없음)이고 CAFT는 감정 균형 자극 블록이다. BCM은 학습되는 모듈로 보인다(초록 기준). 불변 손실의 기준이 "같은 클래스"이고 CAFT ②는 "같은 (클립, 초)"이다. 운동 상상이고 소형 DNN이다. | **높음** |
| 27. Zhang Q., 2025, NeurIPS — "Multi-dataset Joint Pre-training of Emotional EEG Enables Generalizable Affective Computing" (mdJPT) (A4 재사용) | [arXiv 2510.22197](https://arxiv.org/abs/2510.22197) | 감정 EEG 다중 데이터셋 사전학습이다. 배치마다 데이터셋별 피험자 2명 × 시행 수만큼 표본을 넣고, 두 손실을 쓴다(원문 확인). (1) 교차 데이터셋 정렬(CDA): 배치 안 피험자별 공분산 중심을 서로 맞춘다. (2) 피험자 간 정렬(ISA): 같은 자극·같은 시점 양성쌍 대조. | **같은 점**: CLISA식 정렬 배치, 배치 안 피험자 통계, 사람 간 같은 자극 정렬. **다른 점**: 피험자 통계를 빼지 않고 손실로 맞추며, 사전학습 단계이고, 배포 캘리브레이션이 없다. | 중간 |
| 28. Tao X., 2026, arXiv — "Separating personal from population gains when calibrating EEG foundation models for new users" (A4 재사용) | [arXiv 2609.34801](https://arxiv.org/abs/2609.34801) | 동결 CBraMod·REVE·LaBraM, 운동 상상 235명이다. CBraMod에서 집합 인코더가 휴지기 또는 앞쪽 지원 시행 특징("문맥")으로 FiLM 변조나 LoRA 혼합 가중치를 생성한다. 이 문맥 모델을 학습 피험자 에피소드로 학습하는데, 자기 문맥으로 교차 엔트로피를 쓰고 남의 문맥보다 손실이 낮도록 여유 항을 둔다. 배포 때는 경사하강 없이 새 사용자 문맥을 넣는다. 결과: 진짜 문맥과 섞은 문맥의 차이 중앙값 0.00%p, 진짜 문맥과 기본 문맥 평균 −0.21/−0.23%p, 1차 MAML 초기화도 뚜렷한 이점 없음(원문). | **같은 점**: EEG FM에서 "새 사용자의 짧은 캘리브레이션 문맥을 쓰도록 학습 단계에서 에피소드로 준비하고, 배포 경사하강은 없다". CAFT ①의 학습형(ARM-CML형) 일반화이자 가장 가까운 EEG-FM 시험이다. **다른 점**: 문맥 인코더와 생성기라는 새 파라미터를 학습하고, 백본은 동결하며, 운동 상상이고, 결과는 무효과다. 교환·섞은 문맥 대조와 모집단 학습량 대조를 요구할 근거가 된다. | 중간 |
| 29. Cui Y., 2019, IEEE TNSRE — "EEG-Based Driver Drowsiness Estimation Using Feature Weighted Episodic Training" (FWET) | [DOI 10.1109/TNSRE.2019.2945794](https://doi.org/10.1109/TNSRE.2019.2945794), [arXiv 1909.11456](https://arxiv.org/abs/1909.11456) | 특징 가중과 에피소드 DG 학습을 결합해, 새 운전자의 캘리브레이션 데이터 없이 졸음 정도를 추정(회귀)한다(초록). | **같은 점**: EEG에서 새 피험자 이동을 에피소드로 흉내 낸다. **다른 점**: 캘리브레이션 데이터를 전혀 쓰지 않는 DG이고 적응 연산이 없다. | 낮음 |
| 30. Li D., 2021, IEEE NER — "Model-Agnostic Meta-Learning for EEG Motor Imagery Decoding in Brain-Computer-Interfacing" (같은 계열: Duan 2020 IEEE Access MLCL [DOI](https://doi.org/10.1109/ACCESS.2020.3045225) / Ng & Guan 2024 Neural Networks [DOI](https://doi.org/10.1016/j.neunet.2024.106108) / Han 2024 ESWA META-EEG [DOI](https://doi.org/10.1016/j.eswa.2023.121986) / Miao 2026 J. Neurosci. Methods [DOI](https://doi.org/10.1016/j.jneumeth.2026.110742)) | [DOI 10.1109/NER49283.2021.9441077](https://doi.org/10.1109/NER49283.2021.9441077), [arXiv 2103.08664](https://arxiv.org/abs/2103.08664) | MAML로 사용자·세션 간에 빠르게 일반화하는 운동 상상 디코더의 초기값을 학습한다(PhysioNet MI). 적은 데이터에서 전이학습보다 좋았다(초록). 같은 계열 논문들은 피험자를 과제로 보는 MAML 변형으로 무보정(zero-calibration)과 소수샷을 다룬다(Ng & Guan은 내적 발화도 다룸). | **같은 점**: 피험자를 과제로 보고 새 피험자 적응을 학습에서 흉내 낸다. **다른 점**: 배포 때 라벨 있는 소수 시행으로 경사 갱신하거나 아예 무보정이다. 고정 연산이 아니다. | 낮음 |
| 31. Li J., 2022, arXiv — "A Novel Semi-supervised Meta Learning Method for Subject-transfer Brain-computer Interface" (SSML) | [arXiv 2209.03785](https://arxiv.org/abs/2209.03785) | 기존 피험자로 메타 모델을 학습한 뒤, 새 피험자의 소수 라벨과 다수 무라벨로 반지도 파인튜닝한다. 사건 관련 전위(ERP), 감정, 수면 단계에서 평가했다(초록). | **같은 점**: ERP와 감정에서 메타학습으로 캘리브레이션을 준비한다. **다른 점**: 배포 경사하강을 하고 무라벨 대상 데이터를 쓴다. | 낮음 |
| 32. Bhosale, 2022, Biomed. Signal Process. Control — "Calibration free meta learning based approach for subject independent EEG emotion recognition" (A3 재사용) | [DOI 10.1016/j.bspc.2021.103289](https://doi.org/10.1016/j.bspc.2021.103289) | 감정 EEG 메트릭 메타학습이다. Li T. 2026 서베이([arXiv 2604.27033](https://arxiv.org/abs/2604.27033))에 따르면, 에피소드의 지원·질의 표본을 반드시 서로 다른 피험자에서 뽑는 "피험자 독립 표집"으로 피험자가 아니라 감정으로 모이는 공간을 학습한다(원문 접근 실패). 소수샷과 무보정을 둘 다 평가했다(A3). | **같은 점**: "원형은 다른 사람, 질의는 새 사람"이라는 CAFT 배포 분류(학습 피험자 원형 + 새 사용자 코사인)를 에피소드로 흉내 낸다. **다른 점**: 도메인 중심화가 없다. | 중간 |
| 33. Chen C., 2025, J. Neural Eng. — "Model-agnostic meta-learning for EEG-based inter-subject emotion recognition" | [DOI 10.1088/1741-2552/ad9956](https://doi.org/10.1088/1741-2552/ad9956) (PubMed 39622162) | 여러 학습 피험자로 메타 디코더를 학습한 뒤 새 피험자에 1샷으로 적응한다. SEED·DEAP·DREAMER에서 여러 디코더 구조 모두 일반 지도학습보다 좋았다(초록). | **같은 점**: 감정 교차 피험자에서 새 사람 적응을 학습에서 흉내 낸다. **다른 점**: 라벨 1샷으로 경사 적응한다. | 낮음 |
| 34. Jaiswal, 2023, CIKM 2023 MUWS 워크숍 — "Towards Subject Agnostic Affective Emotion Recognition" | [arXiv 2310.15189](https://arxiv.org/abs/2310.15189) | 순환망·분류기와 "합 분해 집합 함수" 기반 분포 이동 조절기를 메타학습과 적대학습으로 학습한다. 시험 때 대상 데이터로 몇 번 자기 적응한다(초록). | **같은 점**: 감정 EEG에서 집합 요약(평균 풀링 계열)으로 새 도메인에 적응하도록 메타학습한다. **다른 점**: 시험 데이터와 경사 적응을 쓰고, 학습되는 모듈이다. | 낮음 |
| 35. An S., 2026, MICCAI — "Subject- and Task-Aware EEG Foundation Model" (STEM) (A4 재사용) | [MICCAI 2026](https://papers.miccai.org/miccai-2026/1006-Paper2756.html) | EEG FM 사전학습에 메트릭 기반 에피소드 메타학습과 대조 자기지도를 넣어 피험자 성분과 과제 성분을 나눈다. 새 피험자 적응이 향상됐다(PhysioNet-MI, SHU-MI, ISRUC, 초록과 심사평 기준). | **같은 점**: EEG FM 학습 단계에서 새 피험자 적응을 에피소드로 준비한다. **다른 점**: 사전학습 단계이고, 감정 과제가 없으며, 배포 방식의 세부를 확인하지 못했다. | 낮음 |

### A-6. 이름 충돌 (본표 편수에서 제외)

우리 "calibration"은 새 사용자 캘리브레이션 데이터를 뜻한다. 아래는 "확률 보정(calibration)" 등 다른 의미로 같은 이름을 쓰는 경우다.

| 논문 | 링크 | 충돌 내용 |
|---|---|---|
| Xiao J., 2025, ICML — "Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach" | [arXiv 2505.01997](https://arxiv.org/abs/2505.01997) (arXiv 주석 "ICML 2025") | **이름 그대로 충돌.** 선호 정렬 뒤 LLM의 과신을 줄이는 확률 보정(ECE)용 파인튜닝이며, 약칭은 CFT다. 검색하면 우리 가제와 섞인다. |
| Chen M.K., 2025, arXiv — "Improving Large Language Models with Concept-Aware Fine-Tuning" | [arXiv 2506.07833](https://arxiv.org/abs/2506.07833) | **약칭 CAFT 충돌**(다중 토큰 학습). |
| Casademunt, 2025, arXiv — "Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning" | [arXiv 2507.16795](https://arxiv.org/abs/2507.16795) | **약칭 CAFT 충돌**(해석 도구로 찾은 개념을 제거하며 파인튜닝). |
| Bohdal, 2023, TMLR — "Meta-Calibration: Learning of Model Calibration Using Differentiable Expected Calibration Error" | [arXiv 2106.09613](https://arxiv.org/abs/2106.09613) (arXiv 주석 "TMLR 08/2023") | "meta-calibration"과 "learning to calibrate"는 확률 보정의 메타학습을 뜻한다. 우리 글에서 이 표현을 쓰면 혼동된다. |
| Mai, 2024, NeurIPS — "Fine-Tuning is Fine, if Calibrated" | [arXiv 2409.16223](https://arxiv.org/abs/2409.16223) (arXiv 주석 "NeurIPS 2024") | 제목 구조가 비슷하다("fine-tuning … calibrated"). 여기서 calibration은 로짓 척도 사후 보정이다. |

**명명 권고**: 제목 "Fine-Tune Like You Calibrate"는 그대로 써도 충돌이 없다. 방법 이름 "Calibration-Aware Fine-Tuning"과 약칭 "CAFT"는 위 충돌 때문에 바꾸는 것이 낫다. 후보로 "calibration-matched fine-tuning"이나 "calibration-episode fine-tuning"을 생각할 수 있으나, 이 후보들의 충돌 여부는 확인하지 않았다.

---

## (B) 가장 가까운 선행 5편 정밀 비교

### B-0. 한눈 비교

| 항목 | **CAFT (우리)** | CLISA (Shen 2023) | Kwak 2023 | ARM-BN (Zhang 2021) | MetaBN (Bronskill 2020) | Tao & Chen 2026 |
|---|---|---|---|---|---|---|
| 과제·데이터 | 감정, SEED-V 등, LOSO | 감정, SEED·THU-EP | 운동 상상(MI) | 이미지 (MNIST, FEMNIST, CIFAR-10-C, Tiny ImageNet-C, WILDS) | 새 클래스 소수샷 | MI 3개 데이터셋, 235명 |
| 모델·단계 | LaBraM 전체 파인튜닝 (지도) | 소형 CNN, 자기지도 대조 사전학습 → DE+MLP | 소형 DNN + BCM (지도) | BN이 있는 임의 모델, 메타 학습 | 메타학습 모델의 BN 층 | 동결 CBraMod + 문맥→FiLM/LoRA 생성기 |
| 학습 때 통계 단위 | 같은 세션 피험자 1명의 15창 (감정마다 3) | 미니배치 안 피험자 1명의 표본 (시행마다 1개) | 짝지은 휴지기 입력 (세부 미확인) | 한 도메인에서 뽑은 배치 | 과제의 컨텍스트(지원) 집합 | 학습 피험자의 문맥 (휴지기·앞쪽 시행) |
| 적응 연산 | 분류 토큰 평균 빼기 (1개 위치) | z-점수 (입력·풀링 출력·투영기 중간) | BCM으로 피험자 변이 특징 제거 | 모든 BN 층 평균·분산 정규화 | 모든 BN 층 평균·분산 정규화 | 집합 인코더 → FiLM 변조 또는 LoRA 혼합 |
| 연산 학습 여부 / 새 파라미터 | 아니오 / 0 | 아니오 / 0 | 예(초록 기준) / BCM | 아니오 / BN 외 0 | 아니오 / 0 (TaskNorm은 혼합 계수 학습) | 예 / 문맥 인코더·생성기 |
| 배포 통계 출처 | 테스트와 분리된 캘리브레이션 블록 (감정마다 1편, 20 s~전체) | DE 특징을 시험 데이터로 온라인 갱신(전이적). 인코더 쪽 처리는 미확인 | 새 피험자 1분 휴지기 | 시험 배치 자체(전이적) | 컨텍스트 집합(비전이적) | 새 사용자 휴지기·앞쪽 지원 시행 |
| 배포 경사하강 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 |
| 분류 대상이 통계에 포함? (학습 / 배포) | **포함 / 불포함 (불일치)** | 포함 / 포함 | 불포함 / 불포함 | 포함 / 포함 | 불포함 / 불포함 | 불포함 / 불포함 |
| 에피소드를 배포 모양에 맞춤 | 감정 균형, 같은 세션, 피험자 간 같은 (클립, 초) | 피험자 간 같은 시점 정렬 (감정 균형은 시행 구성상) | 미확인 | 도메인 단일성만 | 클래스 균형 N-way K-shot | 자기 문맥 대 남의 문맥 |
| 사람 간 정렬 손실 | ② 같은 (클립, 초) 양성만, 1−코사인 | 같은 자극 양성쌍 + 음성쌍 InfoNCE | 같은 클래스 피험자 불변 손실 | 없음 | 없음 | 없음 (남의 문맥 대비 여유 항) |
| 보고된 결과 | (5명) 중심화 뒤 클립 +0.06, 5/5 | SEED LOSO 86.4±6.4 (A3) | 기존 DNN 대비 유의 향상(초록) | 이미지 4과제 평균은 BN 적응보다 좋음, FMoW는 악화 | 학습 속도·정확도 개선, 작은 컨텍스트에서 약함 | 진짜−섞은 문맥 중앙값 0.00%p |

### B-1. CLISA (Shen, TAFFC 2023) — 감정 EEG에서 가장 가까움

**같은 점**
- 배치를 "여러 피험자 × 같은 자극 시점"으로 짠다(피험자 2명, SEED 한 세션의 시행마다 1구간씩, 같은 시점).
- 배치 안에서 피험자별로 통계를 내어 정규화한다(z-점수).
- 같은 자극 구간끼리 사람 간 표현을 끌어당긴다.
- 즉 CAFT의 배치 구성, ①, ②의 세 요소가 모두 있다.

**다른 점**
- 정규화가 감정 분류 학습이 아니라 자기지도 대조 사전학습 안에 있다.
- 감정 분류는 그 뒤 DE 특징 + MLP로 하고, 그 단계의 정규화는 시험 데이터로 갱신한다(전이적).
- "학습 때의 정규화 = 배포 때의 정규화"로 설계하지 않았다. 학습은 배치 안 z-점수, 예측은 온라인 지수 가중 갱신이다.
- 별도 캘리브레이션 블록이라는 개념이 없다.
- 소형 CNN이고, z-점수를 여러 층에 건다.

**리뷰어가 할 말**: "CAFT = CLISA의 표집기와 층화 정규화를 FM 지도 파인튜닝에 옮긴 것 아닌가."

**대응**
- CAFT의 핵심은 "배포의 비전이적 캘리브레이션 연산과 학습 연산을 일치시킨 것"이다.
- CLISA식 설정(시험 스트림 통계, z-점수, 여러 층)과 직접 비교한다.
- 위치는 분류 토큰 하나, 연산은 평균만으로 둔 선택이 왜 나은지(또는 같은지) 보여 준다(C-1의 6).

### B-2. Kwak (JBHI 2023) — "짧은 별도 기록으로 피험자 성분 빼기"를 학습·배포에 똑같이 쓴 EEG 선행

**같은 점**
- 새 사용자가 짧은 별도 기록(1분 휴지기)을 제공하고, 모델은 그 기록으로 깊은 특징의 피험자 성분을 뺀다.
- 학습 피험자에게도 같은 방식(짝지은 휴지기)으로 학습한다.
- 배포 때 경사하강이 없다.
- 같은 클래스를 사람과 무관하게 모으는 손실이 ②와 같은 역할을 한다.

**다른 점**
- 기준 기록이 과제와 무관한 휴지기다. CAFT의 캘리브레이션 블록은 감정이 균형 잡힌 자극 블록이라, "보정 블록의 감정 구성"이 결과를 좌우한다는 이전 연구 결과와 맞물린다.
- BCM은 학습되는 모듈로 보인다(초록 기준, 본문 미확인). CAFT는 연산 자체가 고정(평균 빼기)이다.
- 운동 상상, 소형 DNN이다.
- 후속 ResTL(An 2024, MICCAI)은 BCM이 "RS EEG 신호로 추출 특징만 보정하므로 대상 피험자에 모델을 적응시키는 능력이 제한된다"고 평했다. 휴지기 기반 고정 보정에는 한계가 있다는 지적이다.

**대응**
- 감정 과제에서 "휴지기 대신 감정 균형 자극 블록"을 쓰는 이유를 실험으로 보이면 차별화된다.
- 예: 같은 길이의 중립 클립만으로 μ를 계산하는 대조. 휴지기 기록이 없는 SEED 계열에서는 중립 클립이 가장 가까운 대용물이다.

### B-3. ARM-BN과 ARM-CML (Zhang, NeurIPS 2021) — 원리·연산의 직접 원형

**같은 점**
- ARM-BN은 "학습 배치를 한 도메인에서만 뽑고, 그 배치 통계로 정규화하며 학습"한 뒤, 시험 때 새 도메인 배치 통계로 같은 정규화를 한다.
- 논문은 이를 "BN 적응(사후)에 메타 학습 단계를 더한 것"이라고 부르고, BN 적응 대비 개선을 보였다.
- 이전 연구(사후 중심화)와 CAFT의 관계가 정확히 이것이다.
- ARM-CML의 "배치 평균 문맥"은 CAFT ①의 학습형 일반화다.

**다른 점**
- 시험 배치 자체로 통계를 낸다(전이적, 같은 배치 안 시험 표본들이 서로 영향을 준다).
- 모든 BN 층에 평균·분산을 쓴다.
- 배치가 "한 도메인 무작위 표본"이지, 배포 보정 블록과 같은 구성(감정 균형, 자극 정렬)이 아니다.
- 이미지 과제이며, WILDS FMoW에서는 ARM-BN이 오히려 나빴다. 이 학습 방식이 항상 이득은 아니다.

**인용 근거**
- ARM 저자들은 *"All of these methods are straightforward extensions of existing meta-learning and adaptation methods, and this is intentional"*라고 썼다(원문).
- 기존 적응 연산에 학습 단계를 맞춰 주는 것 자체를 기여로 내세운 선례이므로, CAFT도 같은 방식으로 포지셔닝할 수 있다(분야 관례상 정당한 기여 유형).

### B-4. TaskNorm / MetaBN (Bronskill, ICML 2020) — "보정 집합 통계로 정규화"를 학습·시험에 똑같이 쓴 원형

**같은 점**
- 작은(소수샷), 클래스 균형 잡힌 컨텍스트 집합으로만 통계를 내어, 컨텍스트·대상을 모두 정규화한다.
- 메타 학습과 메타 시험에 똑같이 적용한다.
- 새 파라미터가 없고(MetaBN), 배포 경사하강이 없으며, 비전이적이다.
- "캘리브레이션 블록 모양의 에피소드 + 고정 연산 + 학습·배포 동일"이라는 CAFT 조합이 형식상 여기에 이미 있다.

**다른 점**
- "과제"가 새 클래스 소수샷 과제이지 사람(도메인)이 아니다.
- 모든 BN 층의 평균·분산을 쓴다.

**중요한 시사점 두 가지**
1. TaskNorm은 대상 표본끼리 통계를 섞는 TBN을 비판하고 비전이성을 요건으로 삼았다. 그런데 CAFT 학습은 분류 대상 15창이 자기 평균에 들어가 TBN에 가깝다. 배포는 테스트와 분리된 블록이라 MetaBN과 같다. 즉 CAFT의 학습·배포 일치가 완전하지 않다. → 에피소드를 "평균용(컨텍스트)"과 "손실용(질의)"으로 나누는 변형이 필요하다(C-2의 8).
2. MetaBN은 컨텍스트가 작으면 추정 잡음 때문에 약하다. TaskNorm은 표본 자체 통계와 학습된 비율로 섞어 해결했다. → 캘리브레이션 블록이 짧을 때(20 s), 학습 피험자 평균 쪽으로 수축하는 변형이 도움이 될 수 있다.

### B-5. Tao & Chen (arXiv 2026-09) — EEG FM에서 "캘리브레이션 문맥을 쓰도록 학습"한 가장 가까운 시험(결과 무효과)

**같은 점**
- EEG FM이고 새 사용자 캘리브레이션 상황이다.
- 학습 피험자 에피소드에서 자기 문맥(휴지기·앞쪽 시행 특징)으로 적응된 모델을 교차 엔트로피로 학습한다. 남의 문맥보다 손실이 낮도록 하는 여유 항도 둔다.
- 배포 때는 새 사용자 문맥으로 경사하강 없이 적응한다.
- CAFT ①의 "학습되는 버전"이다.

**다른 점**
- 문맥 인코더와 FiLM/LoRA 생성기라는 새 파라미터가 있다.
- 백본은 동결이고, 운동 상상이다.
- 결과가 무효과다: 진짜 대 섞은 문맥 중앙값 0.00%p, 진짜 대 기본 문맥 평균 −0.21%p(FiLM), −0.23%p(LoRA). 1차 MAML 초기화도 일반 연속 학습 대비 이점이 없었다.

**함의**
- "EEG FM을 캘리브레이션을 흉내 낸 에피소드로 학습한 첫 사례"라는 주장은 불가능하다. STEM(2026)의 에피소드 사전학습도 있다.
- 반대로 CAFT가 같은 대조군 아래에서 양의 효과를 보이면, "학습되는 문맥 조건화는 섞은 문맥과 차이가 없었는데, 파라미터 없는 평균 빼기를 학습에서 흉내 내면 효과가 있다"는 대비가 그 자체로 기여가 된다.
- 단, 과제(감정 대 운동 상상)와 백본 학습 여부(전체 파인튜닝 대 동결)가 다르므로, 이 비교는 해석으로만 쓰고 직접 우열 주장은 피한다.

### B-6. 판정: 무엇이 새롭고 무엇이 아닌가

**새롭지 않은 것** (각각 선행 있음)
- "시험 조건을 학습에서 흉내" (Matching/ProtoNets/MAML/MLDG)
- "도메인별 배치 통계로 정규화하며 학습 → 배포 때 같은 연산" (ARM-BN, MetaBN, CBN, HAR, DSBN)
- 같은 원리의 EEG 사례 (층화 정규화, Kobler, CLISA)
- "평균을 현재 모델로 계산하고 기울기가 평균을 통해 흐름" (BN 학습 모드 일반, Kobler가 명시)
- "새 파라미터 0, 배포 경사하강 없음" (MetaBN, CBN, ARM-BN, Kwak)
- "여러 피험자 × 정렬된 자극 위치" 배치 (CLISA, mdJPT)
- 같은 자극 시점의 사람 간 정렬 (CLISA, mdJPT, 하이퍼정렬 계열 — A5)

**이번 검색 범위에서 선행을 찾지 못한 것** (주장 가능)
1. EEG FM 전체 파인튜닝에서 분류 토큰 평균 빼기 하나만을, 배포의 비전이적 캘리브레이션 블록 연산과 같게 학습에 넣은 설계
2. 에피소드를 배포 블록과 같은 구성으로 짠 것(같은 세션, 감정 균형, 피험자 간 같은 (클립, 초))
3. 이 조합의 감정 LOSO 실증

또 Tao & Chen의 무효과와 대비되는 "고정 연산의 학습 일치" 효과(대조군을 갖춘 경우)도 주장할 수 있다.

**권장 표현 (영문 초안)**
> CAFT follows the principle that training conditions should match deployment conditions (Vinyals et al., 2016), and in particular the idea of meta-training a model under the very normalization it will use at test time (ARM-BN, Zhang et al., 2021; MetaBN, Bronskill et al., 2020; camera-based BN, Zhuang et al., 2020). Subject-wise normalization during training has also been used for EEG (stratified normalization, Fdez et al., 2021; Shen et al., 2023; Kobler et al., 2022), and baseline-corrected features computed from a short separate recording have been used for motor imagery (Kwak et al., 2023). We adapt this principle to the non-transductive calibration protocol of EEG foundation models. Each episode has the same structure as a calibration block: same session, emotion-balanced, and stimulus-aligned across subjects. The only adaptation operation, subtracting the calibration mean from the [CLS] embedding, is the same during fine-tuning and deployment, and it adds no parameters and no test-time optimization. Learned context conditioning did not beat shuffled-context controls in a recent motor-imagery study (Tao & Chen, 2026); in contrast, we find …

"first"는 쓰지 말고, 꼭 써야 하면 "to our knowledge, in EEG emotion recognition with a held-out calibration block"처럼 범위를 좁힌다.

---

## (C) 리뷰어가 요구할 기준선·대조 실험

### C-1. 반드시 (방법의 핵심 주장을 지키는 데 필요)

1. **표집기와 연산을 분리하는 2×2.** 표집기 {기존 무작위 배치, CAFT 배치(K×M, 정렬, 감정 균형)} × 연산 {중심화 없음, 배치 안 피험자별 중심화}로 네 조건을 돌린다.
   - 근거: ARM은 같은 논리로 "도메인 단위 배치 없이 전체에서 무작위로 뽑은" 대조를 두어 메타 학습의 기여를 분리했다.
   - CLISA·mdJPT를 보면 정렬·균형 배치만으로도 효과가 날 수 있다.
2. **사후 중심화 대 CAFT** (ARM 논문의 "BN 적응 대 ARM-BN"에 해당).
   - 같은 모델·같은 학습량에서 "표준 파인튜닝 + 배포 중심화"(이전 연구)와 "CAFT + 배포 중심화"를 비교한다.
   - CAFT 모델을 중심화 없이 배포했을 때 얼마나 떨어지는지도 보고한다(배포 연산 의존도).
3. **학습량 맞춤과 모집단 학습량 변화** (Tao & Chen 권고).
   - 같은 갱신 수와 같은 표본 노출 수로 비교한다.
   - 기준선 학습을 늘리면 차이가 줄어드는지 본다.
4. **교환·섞은 캘리브레이션 대조** (Tao & Chen).
   - 다른 사용자(같은 데이터셋)의 캘리브레이션 평균이나, 같은 사용자의 다른 세션 평균으로 중심화한다.
   - 이것으로 "개인 특이성"을 따로 측정한다.
5. **상한과 하한을 함께 보고.**
   - 전이적 상한: 테스트 세션 자체 평균(ARM-BN, 층화 정규화, CLISA식)
   - 우리: 테스트와 분리된 캘리브레이션 블록
   - 하한: 학습 피험자 평균, 즉 무보정
6. **층화 정규화·도메인별 정규화 기준선.**
   - 분류 토큰 z-점수(평균+분산)와 평균만 빼기를 비교한다.
   - 중간 층 토큰의 피험자별 중심화를 시험한다. LaBraM은 LayerNorm 구조라 BN 교체 대신 중심화 층을 넣는다.
   - "여러 층에 걸어도 이득이 없다/있다"를 보인다.
   - 근거: Fdez 2021, CLISA, Kobler 2022.
7. **규모와 일반성.**
   - 16명 전원, SEED·SEED-IV·DEAP, 가능하면 FM 하나 더(CBraMod 또는 REVE)로 확장한다.
   - 피험자별 짝 차이와 효과 크기를 보고하고, 모델 선택에 시험 피험자를 쓰지 않았음을 밝힌다(A3).
   - 무효과·음의 결과도 모두 보고한다.

### C-2. 강하게 권장 ("학습-배포 일치" 주장을 정밀하게)

8. **자기 포함 평균 없애기.** 지금은 분류 대상 창이 자기 평균의 1/15를 차지한다. 두 변형을 시험한다.
   - 자기 창을 뺀 평균(GE2E식)
   - 에피소드를 평균용 창과 손실용 창으로 나눈 구성(MetaBN식). 예: 감정마다 3곳 중 1곳은 평균용, 2곳은 손실용
   - 이렇게 하면 배포와 똑같이 비전이적이 된다. TaskNorm을 아는 리뷰어가 짚을 가능성이 높다.
9. **에피소드 모양 민감도.**
   - M(창 수)과 감정 구성(균형, 불균형, 단일 감정)을 바꾼다.
   - 배포 캘리브레이션 길이(20 s~전체)와 맞춘 경우와 아닌 경우를 비교한다.
   - 학습 평균을 배포 캘리브레이션 클립과 같은 클립에서 계산하는 "정확 일치" 변형도 시험한다.
   - 이것이 "캘리브레이션 블록 모양" 주장의 직접 증거다.
10. **배포 분류기와 학습 손실 맞추기.**
    - 배포는 학습 피험자 원형과의 코사인인데, 학습은 선형 head 교차 엔트로피라 아직 불일치가 남는다.
    - 배치 안에서 다른 3명으로 감정 원형을 만들고 나머지 1명으로 질의하는 코사인 원형 손실을 시험한다.
    - 근거: Meta-Baseline, ProtoNets, Bhosale의 피험자 독립 표집.
11. **평균 경로의 기울기.** μ에 stop-gradient를 건 변형으로, "현재 모델로 계산하고 기울기가 흐른다"는 설계가 기여하는지 확인한다.
12. **학습형 대안 하나.**
    - 캘리브레이션 평균으로 FiLM 편향·척도를 생성하는 문맥 조건화(ARM-CML, Tao & Chen식)를 같은 에피소드로 학습해 비교한다.
    - 이것이 "왜 고정 연산인가"에 대한 직접적인 답이 된다.
13. **경사 기반 대안** (같은 캘리브레이션 데이터, 비용과 함께 보고).
    - MAML 또는 1차 MAML (Li D. 2021, Chen C. 2025)
    - 캘리브레이션 블록의 감정 라벨로 하는 소수샷 파인튜닝 (A4의 Compass 방식)
    - 정확도와 함께 계산 비용·지연을 비교한다.

### C-3. ② 관련

14. **② 대안 손실과 비교.**
    - 음성쌍이 있는 InfoNCE (CLISA, mdJPT)
    - 같은 클래스 피험자 불변 손실 (Kwak)
    - 정합 쌍 손실 (MatchDG)
    - 양성만 쓰는 손실이 표현 붕괴 없이 동작하는지(중심화와 교차 엔트로피가 막아 주는지)도 확인한다.
15. **미지 자극 시험.** 학습 피험자 데이터에서 시험 클립을 빼고 다시 학습하는 CLISA식 일반화 시험이다. A3·A5의 권고와 같다.
    - SEED 계열은 모든 피험자가 같은 클립을 보므로, ②의 이득이 감정 정렬인지 자극 고유 성분 정렬인지를 가르는 데 필요하다.
    - ②가 "방향 일치도는 크게 올리지만 정확도 이득은 작다"는 중간 결과는 그대로 보고한다.

---

## (D) 확인하지 못한 항목

**본문을 보지 못하고 초록이나 2차 서술만 본 것**
- Kwak 2023: BCM 내부 구조(학습 파라미터 유무)와 학습 배치 구성은 초록과 ResTL(An 2024)의 서술로만 판단했다. **(미확인)**
- Bhosale 2022: "피험자 독립 표집"은 Li T. 2026 서베이(arXiv 2604.27033)의 서술이다. 출판사 페이지는 접근이 막혔다(HTTP 403). **(미확인)**
- Fdez 2021(층화 정규화): 시험 피험자 통계를 어떻게 계산했는지 본문에 명시가 없다. A3는 전이적이라고 판단했다. **(미확인)**
- CLISA: 예측 단계에서 인코더의 층화 정규화를 시험 피험자에게 어떻게 적용했는지 확인하지 못했다. DE 특징의 온라인 정규화만 확인했다. **(미확인)**
- STEM(MICCAI 2026): 배포 방식, 새 피험자 라벨 수, 배포 때 경사하강 여부. **(미확인)**
- META-EEG(Han 2024, ESWA), MLCL(Duan 2020), Ng & Guan 2024: 서지와 초록(일부는 제목)만 확인했고 방법 세부는 보지 못했다. **(미확인)**
- mdJPT의 NeurIPS 2025 게재: A4의 확인을 재사용했고, 이번에 학회 페이지를 다시 보지는 않았다.
- Meta-Calibration의 TMLR 게재와 Xiao 2025의 ICML 게재: arXiv 주석으로만 확인했다(Xiao는 검색 결과에 ICML 가상 포스터 페이지도 보였다).
- Meta-TTT(arXiv 2410.01709), Learning to (Learn at Test Time)(arXiv 2310.13807): 학회 게재 여부를 확인하지 못했다.

**검색 결과에서 존재만 보고 서지를 확인하지 못한 것 (표에서 제외)**
- "MAML-EEG: A Meta-learning Strategy Based Domain Generalization Framework for Unseen Subject Motor Imagery Classification": 배치 안에서 가상 메타 과제를 만드는 방식이다. ResearchGate만 봤고 게재처는 **미확인**이다.
- 게재처나 연도를 확인하지 못한 MI 메타학습·소수샷 논문:
  - "EEG-TriNet++" (MDPI Bioengineering로 보임)
  - "TCPL: task-conditioned prompt learning for few-shot cross-subject motor imagery EEG decoding" (Front. Neurosci. 2025로 보임)
  - "Memory-augmented-based meta-learning framework for cross-subject motor imagery classification" (BSPC 2025로 보임)
  - "Dual Attention Relation Network With Fine-Tuning for Few-Shot EEG Motor Imagery Classification" (IEEE)
  - "A Study of Prototypical Network Techniques for Cross-Subject EEG Analysis" (FAIA 2024)
  - 모두 **미확인**이다.
- IEEE OJEMB 2026 논문(DOI 10.1109/OJEMB.2026.3667029): 검색에 ERP·메타학습 맥락으로 나왔으나 제목을 확인하지 못했다. **미확인**.
- "Meta-Learning for BCI: A Promising New Direction" (UTS 저장소). **미확인**.
- "Few-Shot Gaze Estimation with Model Offset Predictors" (ICASSP 2022): 사용자별 오프셋 예측으로 보이며 CAFT ①의 분야 밖 유사례일 수 있다. 내용은 **미확인**.

**찾지 못한 것** (없다는 뜻이 아니라, 이번 검색에서 나오지 않았다는 뜻)
- SSVEP에서 메타학습이나 에피소드 학습으로 캘리브레이션을 준비한 연구. SSVEP 캘리브레이션 단축은 템플릿 전이가 주류다(A1).
- "adaptation-aware pre-training"이라는 이름의 논문. 같은 개념은 ARM, MT3, Meta-TTT, MABN 등의 이름으로 있다.
- BCI 의미의 "calibration-aware training/fine-tuning"과 "learning to calibrate"(새 사용자 캘리브레이션 의미)라는 이름의 논문.
- EEG FM 전체 파인튜닝에서 배치 안 피험자별 분류 토큰 중심화를 쓴 2025–2026 논문.

**범위 한계**
- 웹 검색은 약 25회였다. 2026년 하반기 저널 논문과 중국어권 저널은 빠졌을 수 있다.
- EEG 메타학습 논문(특히 운동 상상)은 수가 많아, 대표만 확인하고 나머지는 위 미확인 목록에 남겼다.
