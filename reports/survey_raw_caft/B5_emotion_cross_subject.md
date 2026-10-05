# B5. 교차 피험자 EEG 감정 인식: 피험자·세션 정규화, 피험자 불변 학습, 적은 캘리브레이션 (2018–2026) — CAFT 새로움 점검

기준일 2026-10-05. 갈래 B5 원 보고서.

**검증 방법.**
- 표의 모든 논문은 제목·저자·연도·게재처를 1차 출처(arXiv 원문·abs 페이지, Crossref, OpenAlex, 학회·출판사 페이지)로 확인했다. 수치는 원문 표·본문에서 직접 읽은 것만 적었다.
- 공개 코드가 있는 핵심 논문(mdJPT, DAEST, DMMR, MS-MDA, MoGE)은 GitHub 원본 파일을 직접 읽었다(2026-10-05의 main 브랜치). "코드 확인"이라고 적은 내용은 이 코드 기준이다. 논문 수치가 그 코드로 나왔는지는 확인하지 못했다.
- 기존 조사(A1–A5)에서 이미 확인한 논문은 다시 조사하지 않았다. 그 보고서의 확인 결과를 가져다 썼고, 해당 행에 "(A3 확인)"처럼 표시했다.
- 확인하지 못한 것은 "(미확인)"으로 표시하고 (E)절에 모았다.

**약어 (처음 나올 때 풀어 쓴 이름).**
- 이 연구: 캘리브레이션 인지 파인튜닝(CAFT), 자극 시점 정렬(SLA)
- 평가: 피험자 하나 빼기 교차검증(LOSO)
- 특징·평활: 미분 엔트로피(DE), 선형 동적 시스템 평활(LDS)
- 적응·일반화: 도메인 적응(DA), 도메인 일반화(DG), 도메인 적대 신경망(DANN), 다중 소스 주변 분포 적응(MS-MDA), 최대 평균 불일치(MMD)
- 모델: 파운데이션 모델(FM)
- mdJPT의 손실: 피험자 간 정렬 손실(ISA), 교차 데이터셋 정렬 손실(CDA)
- 그 밖: 운동 상상(MI), 균형 정확도(BAcc), 경험적 위험 최소화(ERM)

---

## 0. 핵심 결론

1. **"CAFT와 거의 같은 일을 한 논문"은 찾지 못했다. 다만 CAFT의 세 부품은 감정 EEG에서 이미 한 묶음으로 쓰였다.** 세 부품은 다음과 같다.
   - (i) 같은 세션의 여러 피험자 × 모든 클립에서 같은 (클립, 시각) 하나씩을 담은 배치
   - (ii) 배치 안에서 피험자마다 자기 표본으로 하는 정규화
   - (iii) 같은 자극·같은 시각의 피험자 간 정렬 손실

   이 세 부품을 함께 쓴 논문이 CLISA(IEEE TAFFC 2022/2023)와 mdJPT(NeurIPS 2025)이고, mdJPT는 코드로도 확인했다. 같은 계보의 DAEST(TAFFC 2025)도 같은 구조다. 따라서 CAFT의 배치 설계, ①의 "배치 안 피험자별 정규화", ②의 "같은 자극 시점 일관성"은 각각 단독으로는 새롭다고 주장할 수 없다.
2. **"학습 때 배포 때와 같은 피험자별 정규화를 넣는다"는 ①의 논리에도 선례가 있다.**
   - 층화 정규화(Fdez 2021, Frontiers in Neuroscience): 지도 학습 안에서 참가자·세션별로 정규화한다. 다만 시험 때는 시험 세션 전체 통계를 쓴다(전이적).
   - Kwak 2023(IEEE JBHI, 운동 상상): 새 사용자의 1분 휴지기 기록으로 피험자 성분을 빼는 모듈을 학습 안에 넣었다.
3. **방어 가능한 새로움은 다음 조합으로 좁혀야 한다.** 짧은 감정 균형 캘리브레이션 블록의 평균을 빼고 프로토타입 코사인으로 분류하는 배포 연산을, 범용 EEG FM(LaBraM)의 **전체 지도 파인튜닝 안에서 그대로 재현**한다. 그리고 그 효과를 **비전이적이고 캘리브레이션 클립을 평가에서 뺀 현실적 프로토콜**에서 클립 단위와 창 단위로 보인다. 이 조합을 한 연구는 찾지 못했다(검색 범위 안).
   - 분야 관례로 보면, 감정 EEG에서는 같은 부품 위의 변형(CLISA → DAEST → TA2CL)도 TAFFC 급 학술지에 실린다. 그러니 이 조합과 엄격한 프로토콜은 기여로 인정될 여지가 충분하다.
   - 단, CLISA·mdJPT·층화 정규화를 부품의 원전으로 반드시 인용해야 한다.
4. **SEED-V 교차 피험자 문헌 수치(약 58–82%)는 우리 수치와 직접 비교할 수 없다.** 이 수치들에는 다음 요인이 섞여 있다.
   - DE 특징에 시행 전체 LDS 평활(비인과)
   - 시험 피험자 데이터로 하는 정규화: CLISA 계열은 온라인 갱신, DMMR·MS-MDA 코드는 피험자별 최소·최대 정규화
   - 시험 피험자 정확도로 하는 모델 선택: DMMR·MS-MDA·DAEST 공개 코드에서 확인
   - 첫 세션만 쓰는 평가

   엄격한 조건의 외부 기준은 두 가지다.
   - MST(2026): 보정 없는 LOSO에서 LaBraM 32.5±4.9%(2초 창)
   - LibEER(TAFFC 2025): 피험자 분할과 검증 피험자 기반 선택에서 20.8–44.3%(DE+LDS 1초 창)

   우리 수치(클립 0.30–0.54)는 이 범위와 맞는다.
5. **②는 평가 위협을 키울 수 있다.** Gerster 등(bioRxiv 2026)은 FACED에서 교차 피험자 분류가 감정보다 **자극(영상) 정체**를 반영함을 보였다. 영상 수를 감정당 1개로 줄이면 오히려 정확도가 올랐다(39.4% → 44.9%). SEED-V의 LOSO에서는 시험 클립을 학습 피험자도 봤다. 그래서 같은 (클립, 초) 일관성 손실이 "클립 지문" 학습을 부추길 수 있다. 미지 자극 시험(CLISA식)과 시간 셔플 대조가 필요하다.
6. **리뷰어가 요구할 기준선 다섯 가지 (모두 공개 코드 있음).** 단 PPDA의 공식 코드는 찾지 못했다.
   - (1) 보정 없는 LaBraM 파인튜닝(MST 조건)
   - (2) 캘리브레이션 중심화만 쓰고 CAFT 학습은 없는 조건, 그리고 전이적 상한
   - (3) CLISA/mdJPT식 학습(같은 샘플러 + 층화 정규화 + 같은 시점 대조)을 LaBraM에 적용
   - (4) 층화 정규화를 캘리브레이션 블록 통계로 돌린 조건
   - (5) 도메인 적대/다중 소스 DA(DANN·MS-MDA)를 캘리브레이션 블록을 대상 도메인으로 돌린 조건과 DG(DMMR 또는 MAT)

---

## (A) 논문 표

"새로움 위협"은 CAFT의 구성 요소(배치 설계, ① 배치 안 중심화와 배포 일치, ② 피험자 간 자극 시점 일관성, 현실적 캘리브레이션 프로토콜)를 이미 한 정도다. 높음은 핵심 부품을 둘 이상 이미 쓴 경우, 중간은 하나를 쓰거나 원리가 같은 경우, 낮음은 관련 맥락이거나 비교 기준인 경우다.

### A-1. 같은 자극 배치 · 배치 안 피험자 정규화 · 피험자 간 일관성 (가장 가까운 계열)

| 논문 (첫 저자, 연도, 학회/저널) | 링크 | 무엇을 하나 | 프로토콜·수치 (확인된 것만) | CAFT와의 관계 | 새로움 위협 |
|---|---|---|---|---|---|
| Shen X., "Contrastive Learning of Subject-Invariant EEG Representations for Cross-Subject Emotion Recognition" (CLISA), 2023 (온라인 2022), IEEE TAFFC 14(3) | https://doi.org/10.1109/TAFFC.2022.3164516 (arXiv 2109.09559) | **미니배치**: 피험자 두 명(A, B)에서 시행마다 같은 시간 구간의 표본을 하나씩 뽑는다(SEED는 30초 창, 15초 간격). **학습 중 층화 정규화**: 한 피험자의 배치 안 표본들을 채널별로 이어 z-점수를 낸다. 적용 위치는 인코더 입력, 평균 풀링 출력, 투영기 시간 합성곱 출력이다. **손실**: 같은 시행·같은 구간 쌍을 양성으로 하는 대조 손실(NT-Xent). 그 뒤 인코더 출력에서 DE를 뽑아 LDS 평활을 하고 MLP를 교차 엔트로피로 학습한다. | SEED LOSO 86.4±6.4%. 미지 자극 시험(학습은 시행 2/3, 시험은 나머지 1/3)에서 77.4±13.4%(A3 확인). 예측 단계 특징 정규화는 학습 통계로 시작해 시험 피험자 데이터로 지수 가중 갱신한다(감쇠 0.99, 온라인 전이적). 코드는 FACED 데이터셋과 함께 Synapse(doi:10.7303/syn50614194)에 공개돼 있다. | CAFT의 배치 구성(피험자 × 같은 시각, 클립 전부), ①의 "배치 안 피험자별 정규화", ②를 한 논문에서 모두 쓴다. 다른 점은 다음과 같다. (a) 인코더를 자기지도 대조로만 학습한다. (b) 정규화가 z-점수이고 여러 층에 걸린다. (c) 분류기는 따로 학습하고 정규화도 다르다(온라인 전이적). (d) 범용 FM이 아니고 캘리브레이션 블록 개념이 없다. | **높음** |
| Zhang Q., "Multi-dataset Joint Pre-training of Emotional EEG Enables Generalizable Affective Computing" (mdJPT), 2025, NeurIPS 2025 | https://arxiv.org/abs/2510.22197 (doi:10.52202/085713-5515), 코드 https://github.com/ncclab-sustech/mdJPT_nips2025 | 감정 데이터셋 6개에서 대상 데이터셋을 하나 빼고 공동 사전학습한다. **배치**: 데이터셋마다 피험자 두 명 × 시행마다 표본 하나(5초 창, 2초 간격). **손실**: ISA는 같은 시행·같은 시작 시각 쌍을 양성으로 하는 대조 손실이다. CDA는 배치 안에서 피험자별 공분산 중심을 서로 맞춘다. **코드 확인**: 샘플러는 두 피험자가 같은 세션이어야 한다고 강제(assert)하고, 시행마다 두 피험자에게 같은 시간 인덱스를 쓴다. 인코더 순전파 안에서 `stratified_layerNorm(x, n_samples=B//2)`으로 배치를 피험자 단위로 나눠 z-점수를 낸다. | 교차 데이터셋 평가(대상 데이터셋 피험자 1/4로 MLP 학습, 나머지 3/4로 시험), SEED-V: mdJPT 65.02±0.98, LaBraM 41.80±3.53, DE 45.58±1.92. 제로샷(대상 데이터셋 안 최근접 이웃) SEED-V: mdJPT 52.91, LaBraM 39.70. 시각을 맞추지 않은 양성 쌍으로 바꾸면 성능이 크게 떨어진다(부록 표 S5, 수치는 읽지 않음). 코드 확인: 특징 추출 때 피험자·세션별 온라인 정규화(시험 피험자 데이터로 갱신)와 시행 단위 RTS 역방향 LDS 평활(비인과)을 쓴다. | CAFT 배치, ①의 형태, ②를 감정 FM 사전학습에서 모두 쓴다. 다른 점은 다음과 같다. (a) 자기지도 사전학습이고, 중심화된 표현에 지도 CE를 거는 단계가 없다. (b) 학습 연산과 배포 연산이 다르다(온라인 정규화, LDS, MLP). (c) 새 사용자 캘리브레이션이 없고, 평가는 교차 데이터셋 소수 피험자와 제로샷이다. (d) 범용 FM 파인튜닝이 아니다. | **높음** |
| Shen X., "Dynamic-Attention-Based EEG State Transition Modeling for Emotion Recognition" (DAEST), 2025, IEEE TAFFC 16(4):3552 | https://doi.org/10.1109/TAFFC.2025.3593630 (arXiv 2411.04568), 코드 https://github.com/RunminGan1218/DAEST | CLISA와 같은 샘플러(두 피험자 × 시행마다 하나, 같은 영상 구간이 양성)와 대조 학습에 동적 주의 인코더를 붙였다. 코드 확인: 인코더에 같은 `stratified_layerNorm`(선택 적용)이 있다. 분류 단계에서는 피험자별 시간 적응 정규화, LDS, MLP를 쓴다. | SEED 88.1±3.6%, SEED-V 73.6±12.7%(5감정; 같은 표의 DE+MLP 59.3±17.2, CLISA 67.3±13.0). 코드 기본 설정은 SEED-V 16명, 10겹 피험자 교차검증, 온라인 정규화 감쇠 0.99다(본문은 SEED-V를 20명으로 적음). 코드 확인: `train_mlp.py`는 빼 둔 피험자 정확도를 '검증' 지표로 감시하고, 에폭 최댓값(`best_model_score`)의 평균을 보고한다. | CLISA 계열의 SEED-V 대표 수치다. 부품은 CAFT와 같고(배치, 층화 정규화, 자극 시점 대조), 그 외 차이는 CLISA와 같다. 리뷰어가 SEED-V 비교 대상으로 가장 먼저 들 논문이다. | 중간~높음 |
| Xie Y., "Cross-Subject EEG Emotion Recognition Based on Temporal Asynchronous Alignment Contrastive Learning" (TA2CL), 2026, arXiv 2605.22379 | https://arxiv.org/abs/2605.22379 | 같은 샘플러를 쓴다(무작위 두 피험자, 같은 자극·같은 시간 구간이 양성). 토큰 단위 top-K 비동기 매칭(Async-InfoNCE)으로 피험자 간 반응 시차를 허용한다. 인코더를 고정하고, 오프라인 특징에 정규화와 LDS를 건 뒤 MLP로 분류한다. | SEED-V **LOSO** 70.1±12.2%(재현 DAEST 68.2±12.4, CLISA 67.3±13.0, DE+MLP 59.3±17.2). SEED는 5겹 86.4±4.8%. 코드 공개 문구는 없다. | ②의 "같은 초" 가정에 대한 직접적 대안(시차 허용)이다. CAFT ②를 시차 허용 버전과 비교할 근거가 된다. | 중간 |
| Shen X., "Contrastive learning of shared spatiotemporal EEG representations across individuals for naturalistic neuroscience" (CL-SSTER), 2024, NeuroImage 301:120890 | https://arxiv.org/abs/2402.14213 | 정확히 시간 정렬된 피험자 쌍을 양성으로 하는 대조 학습으로 개인 간 공유 시공간 표현을 학습한다(A3·A5 확인). | 피험자 간 상관(ISC) 상승(A5 확인: 센서 공간 0.034에서 0.062로). | ②의 신경과학적 근거 계열이다. | 낮음~중간 |
| Kan H., "Self-supervised group meiosis contrastive learning for EEG-based emotion recognition" (SGMC), 2023, Applied Intelligence 53(22):27207–27225 | https://doi.org/10.1007/s10489-023-04971-0 (arXiv 2208.00877) | **그룹 샘플러**: 1초 자극 구간 P개 × 피험자 2Q명을 뽑아, 같은 자극을 본 피험자 묶음을 만든다. 감수분열식 교차(짝짓기, 일부 교환, 분리)로 그룹을 증강하고 그룹 수준 표현을 대조한다. | 평가는 1초 구간을 무작위로 70:15:15 분할한다(모든 피험자가 학습·시험 양쪽에 있어 교차 피험자가 아님). SEED는 시행별 L2 정규화를 쓴다. SEED 94.04%, DEAP valence/arousal 94.72/95.68%. | "피험자 묶음 × 같은 자극 시점" 배치의 또 다른 선례다. 평가는 시행 내 분할이라 비교할 수 없다(함정 사례). | 중간 (배치) |
| Alghamdi A.M., "Cross-subject EEG signals-based emotion recognition using contrastive learning" (CSCL), 2025, Scientific Reports 15:28295 | https://doi.org/10.1038/s41598-025-13289-5 | CLISA 전처리를 따르고, 피험자별로 채널 방향 층화 정규화를 한다. 교차 피험자 대조 학습이다. | LOSO와 10겹을 썼다고 적었고 SEED 3클래스 97.7%를 보고한다. 프로토콜 서술이 불명확하고 수치가 문헌과 크게 어긋나 신뢰도가 낮다. | 층화 정규화가 CLISA 이후 퍼졌다는 근거 정도로만 쓴다. | 낮음 |
| Meng R., "Group Resonance Network: Learnable Prototypes and Multi-Subject Resonance for EEG Emotion Recognition" (GRN), 2026, ICANN 2026 | https://arxiv.org/abs/2603.11119 | 추론 때 시험 표본과 같은 자극 시간축에 있는 학습 피험자 참조 집합 사이의 위상 동기(PLV/코히어런스)를 입력으로 쓴다(A5 확인). | SEED LOSO 87.90%(A1 기록). 시험 표본마다 자극·시점 식별이 필요하다. | 같은 자극을 시험 시점에 쓰는 방식이다. 정규화나 캘리브레이션 학습은 아니다. | 낮음 |

### A-2. 학습 안의 피험자(세션)별 정규화 · 기준 보정 · 도메인별 정규화

| 논문 | 링크 | 무엇을 하나 | 프로토콜·수치 | CAFT와의 관계 | 새로움 위협 |
|---|---|---|---|---|---|
| Fdez J., "Cross-Subject EEG-Based Emotion Recognition Through Neural Networks With Stratified Normalization", 2021, Frontiers in Neuroscience 15:626277 | https://doi.org/10.3389/fnins.2021.626277, 코드 https://github.com/javiferfer/cross-subject-eeg-emotion-recognition-through-nn | 시행(클립) 하나당 특징 벡터 하나를 쓴다(62채널 × 4대역 = 248). 입력은 최소·최대 정규화하고, 처음 세 은닉층 출력마다 **참가자·세션별**(그 세션 15개 시행) 평균·분산으로 정규화한다. 학습 데이터 전체를 한 배치에 넣고 지도 손실(NLL)로 학습한다. | SEED LOSO(15겹)에서 3감정 79.6%, 2감정 91.6%. 배치 정규화보다 유의하게 높다. 시험 참가자도 자기 세션의 15개 시행 전체로 정규화한다(**전이적**, 시행 전체). 특징이 시행 단위라 정확도는 사실상 클립 단위다. | ①의 논리, 즉 "배포와 같은 피험자별 정규화를 학습에 넣고 지도 학습한다"를 이미 구현했다. 정규화 통계도 감정 균형(감정마다 5시행)이다. 다른 점은 다음과 같다. (a) 시험 통계로 시험 세션 전체를 쓴다(전이적). (b) 손특징 MLP다. (c) 미니배치 에피소드나 같은 자극 시점 손실이 없다. | **높음** (① 한정) |
| Chen H., "Personal-Zscore: Eliminating Individual Difference for EEG-Based Cross-Subject Emotion Recognition", 2023 (온라인 2021), IEEE TAFFC 14(3):2077–2088 | https://doi.org/10.1109/TAFFC.2021.3137857 | 개인차를 재는 지표 4개와 피험자별 z-점수 특징 처리(PZ)를 제안했다(A3 확인). | SEED 14명. 정확도와 강건성이 올랐다(초록). 시험 피험자 통계 출처는 **(미확인)**(원문 비공개). | 피험자별 정규화의 감정 분야 선례다(전처리 단계). | 중간 |
| Apicella A., "On the effects of data normalization for domain adaptation on EEG data", 2023, Engineering Applications of AI 123:106205 | https://doi.org/10.1016/j.engappai.2023.106205 | 정규화 전략과 DA(DANN·TCA 등)를 조합해 비교했다(A2 확인). | SEED 교차 피험자에서 정규화만으로 81.52±7.26%. 최선 전략은 시험 피험자·세션 자신의 통계를 쓴다(전이적). | "정규화가 DA를 대부분 대신한다"는 근거다. ①이 큰 이득을 내는 이유를 설명할 때 인용한다. | 낮음~중간 |
| Ahmed M.Z.I., "A Novel Baseline Removal Paradigm for Subject-Independent Features in Emotion Classification Using EEG" (InvBase), 2023, Bioengineering 10(1):54 | https://doi.org/10.3390/bioengineering10010054 | 피험자의 휴지기(기준) 전력 스펙트럼으로 시행 전력을 나누는 기준 제거로 피험자 독립 특징을 만든다. | DEAP valence·arousal. MLP에서 기준 보정 없음 대비 +29%, 빼기식 기준 제거 대비 +15%(초록). | 새 사용자의 별도 기준 기록으로 피험자 성분을 없애는 감정 분야 선례다. CAFT는 휴지기 대신 감정 균형 캘리브레이션 클립을 쓴다. | 낮음~중간 |
| Kwak Y., "Subject-Invariant Deep Neural Networks Based on Baseline Correction for EEG Motor Imagery BCI", 2023, IEEE JBHI 27(4):1801–1812 | https://doi.org/10.1109/JBHI.2023.3238421 | 피험자의 기준선(휴지) EEG로 피험자 변이 특징을 추정해 깊은 특징에서 빼는 **기준 보정 모듈(BCM)을 네트워크 안에서 학습**한다. 같은 클래스 특징을 피험자와 무관하게 모으는 피험자 불변 손실을 함께 쓴다. 배포 때는 새 사용자의 **1분 기준선 EEG**만 쓰고 과제 캘리브레이션은 없다. | 운동 상상. 기존 심층 모델의 해독 정확도가 유의하게 올랐다(초록. 데이터셋과 수치는 **(미확인)**, 원문 비공개). | "배포 때 쓸 짧은 새 사용자 기록으로 피험자 성분을 빼는 연산을 학습에 넣는다"는 원리가 CAFT ①과 같다. 다른 점은 다음과 같다. (a) 운동 상상이다. (b) 기준이 휴지기다. (c) 같은 자극 시점 대신 클래스 기준 불변 손실을 쓴다. (d) FM이 아니다. | 중간~높음 (분야 밖) |
| Bakas S., "Latent alignment in deep learning models for EEG decoding", 2025, J. Neural Eng. 22(1):016047 | https://doi.org/10.1088/1741-2552/adb336 (arXiv 2311.17968) | 층마다 피험자별 표준화를 한다. 학습은 피험자 단위 묶음으로 하고, 추론 때는 시험 피험자의 라벨 없는 시행 전체를 쓴다(A2 확인). | 운동 실행·운동 상상에서 EA·AdaBN과 비슷한 크기의 이득. 클래스가 불균형한 문맥에서는 붕괴할 수 있다. | 학습·배포 일치형 피험자 정규화의 분야 밖 선례다(전이적). 캘리브레이션 블록의 클래스 균형이 왜 중요한지 보여 주는 근거이기도 하다. | 중간 (분야 밖) |
| Kobler R.J., "SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG" (TSMNet), 2022, NeurIPS 2022 | https://arxiv.org/abs/2206.01323 (doi:10.52202/068431-0450) | 대칭 양정치(SPD) 다양체에서 도메인별 모멘텀 배치 정규화를 해, 피험자·세션마다 재중심화한다(A1·A2 확인). | 6개 BCI 데이터셋, 대상 도메인 통계 사용(오프라인 전이적). 감정 데이터 없음. | 도메인별 배치 정규화 계보다. CAFT는 LayerNorm 계열 FM의 분류 토큰에 하는 1차 중심화다. | 낮음~중간 |

### A-3. 피험자 적대 학습 · 도메인 적응 · 도메인 일반화 (대표)

| 논문 | 링크 | 무엇을 하나 | 프로토콜·수치 | CAFT와의 관계 | 새로움 위협 |
|---|---|---|---|---|---|
| Ganin Y., "Domain-Adversarial Training of Neural Networks" (DANN), 2016, JMLR 17(59) | https://jmlr.org/papers/v17/15-239.html (arXiv 1505.07818) | 도메인 판별기와 경사 반전층으로 도메인 불변 특징을 학습한다. | (일반 ML) | 피험자 적대 기준선의 원전이다. | 낮음 |
| Li Y., "A Bi-Hemisphere Domain Adversarial Neural Network Model for EEG Emotion Recognition" (BiDANN), 2021, IEEE TAFFC 12(2):494–504 (전신은 IJCAI 2018) | https://doi.org/10.1109/TAFFC.2018.2885474 (IJCAI판 doi:10.24963/ijcai.2018/216) | 전역 판별기 하나와 반구별 판별기 둘을 쓰는 도메인 적대 학습이다. 피험자 독립판 BiDANN-S는 개인 정보의 영향을 낮춘다(초록). | SEED 피험자 독립: 83.28±9.60(BiHDM 원문 표), BiDANN-S 84.14±6.87(RGNN 원문 표). 라벨 없는 시험 데이터를 쓴다(BiHDM 원문이 명시: 전이적). | 피험자 적대 계열의 대표다. CAFT는 적대 손실 없이 정규화와 일관성 손실을 쓴다. | 낮음 |
| Li Y., "A Novel Bi-Hemispheric Discrepancy Model for EEG Emotion Recognition" (BiHDM), 2021, IEEE TCDS 13(2):354–367 | https://doi.org/10.1109/TCDS.2020.2999337 (arXiv 1906.01704) | 좌우 반구 차이 특징에, 학습·시험 데이터 차이를 줄이는 도메인 판별기를 더했다. | LOSO: SEED 85.40±7.53, SEED-IV 69.03±8.66, MPED 28.27±4.99(전이적). | 같은 계열. | 낮음 |
| Li Y., "From Regional to Global Brain: A Novel Hierarchical Spatial-Temporal Neural Network Model for EEG Emotion Recognition" (R2G-STNN), 2022 (온라인 2019), IEEE TAFFC 13(2):568–578 | https://doi.org/10.1109/TAFFC.2019.2922912 | 지역에서 전역으로 가는 BiLSTM과 영역 주의에, 학습·시험 도메인 판별기를 더했다(초록). | 원 논문 수치는 원문 비공개로 읽지 못했다. LibEER 재현 SEED-V 교차 피험자 37.23±12.47. | 같은 계열. | 낮음 |
| Zhong P., "EEG-Based Emotion Recognition Using Regularized Graph Neural Networks" (RGNN), 2022, IEEE TAFFC 13(3):1290–1301 | https://doi.org/10.1109/TAFFC.2020.2994159 (arXiv 1907.07835), 코드 https://github.com/zhongpeixiang/RGNN | 정규화 그래프 신경망에 노드별 도메인 적대 학습(NodeDAT, 라벨 없는 시험 데이터 사용)과 감정 인지 분포 학습을 더했다. | LOSO(SEED는 한 세션): SEED 85.30±6.72, SEED-IV 73.84±8.02(전이적). | 같은 계열. 리뷰어가 자주 요구하는 그래프 기준선이다. | 낮음 |
| Chen H., "MS-MDA: Multisource Marginal Distribution Adaptation for Cross-subject and Cross-session EEG Emotion Recognition", 2021, Frontiers in Neuroscience 15:778488 | https://doi.org/10.3389/fnins.2021.778488, 코드 https://github.com/VoiceBeer/MS-MDA | 소스 피험자마다 분기를 두고 각 분기가 대상과 MMD를 맞추는 다중 소스 DA다(A3 확인). | SEED 교차 피험자 89.63±6.79(A3). **코드 확인**: 기본 정규화 `norm_type='ele'`는 세션·피험자별 특징 최소·최대 정규화이고 시험 피험자도 포함한다. 학습 중 시험 정확도의 최댓값을 결과로 반환한다. | 다중 소스 DA 기준선. 평가 함정의 구체 사례이기도 하다. | 낮음 |
| Wang Y., "DMMR: Cross-Subject Domain Generalization for EEG-Based Emotion Recognition via Denoising Mixed Mutual Reconstruction", 2024, AAAI 38(1):628–636 | https://doi.org/10.1609/aaai.v38i1.27819, 코드 https://github.com/CodeBreathing/DMMR | 대상 데이터 없이 일반화한다고 주장하는 DG다. 잡음 섞은 혼합 상호 재구성으로 사전학습한 뒤 미세조정한다. | 첫 세션 LOSO: SEED 88.27±5.62, SEED-IV 72.70±8.01. **코드 확인**: `preprocess.py`는 시험 피험자도 자기 첫 세션 전체 최소·최대로 정규화한다(전이적 정규화). `train.py`는 에폭마다 시험 피험자 정확도를 재서 최댓값을 보고값으로 쓴다. | DG 기준선이다. 다만 "보정 없음" 주장과 코드가 어긋난다. 그대로 비교하면 안 되는 사례다. | 낮음 |
| Li G., "Learning Domain- and Class-Disentangled Prototypes for Domain-Generalized EEG Emotion Recognition" (MAT), 2025, arXiv 2509.01135 | https://arxiv.org/abs/2509.01135, 코드 https://github.com/WuCB-BCI/MAT | 도메인·클래스를 분리한 이중 프로토타입 DG다(A3 확인). | DE(1초)+LDS. **SEED-V 단일 세션(첫 세션) LOSO**: SVM 53.14±10.10, DANN 56.28±16.25(전이적), DMMR 58.63±15.30, MAT 63.48±10.86. **SEED-V 교차 세션 LOSO**: SVM 41.20±10.76, DMMR 55.16±12.71, MAT 58.39±8.63. 특징 정규화 방식과 모델 선택 방식은 본문에 없다. | SEED-V 교차 피험자 최신 DG 수치. 프로토타입 계열 비교 대상이다. | 낮음 |
| Liu X.-H., "MoGE: Mixture of Graph Experts for Cross-subject Emotion Recognition via Decomposing EEG", 2024, IEEE BIBM 2024 | https://doi.org/10.1109/BIBM62325.2024.10822354, 코드 https://github.com/XuanhaoLiu/MoGE | 채널을 전문가에 배정하는 희소 그래프 전문가 혼합 DG다(ERM으로 학습). | SEED 88.0, SEED-IV 74.3, **SEED-V 81.8**(초록). 공개 코드에는 모델 정의만 있어 정규화·분할·모델 선택은 **(미확인)**. | 리뷰어가 "SEED-V 81.8%"로 인용할 수 있다. 프로토콜을 확인하기 전에는 비교하지 않는다. | 낮음 |

### A-4. 적은 캘리브레이션 (PPDA 이후 "플러그 앤 플레이" 계열 포함)

| 논문 | 링크 | 무엇을 하나 | 프로토콜·수치 | CAFT와의 관계 | 새로움 위협 |
|---|---|---|---|---|---|
| Zhao L.-M., "Plug-and-Play Domain Adaptation for Cross-Subject EEG-based Emotion Recognition" (PPDA), 2021, AAAI 35(1):863–870 | https://doi.org/10.1609/aaai.v35i1.16169 | 세션 맨 앞의 라벨 없는 45초로 새 사용자의 개인 인코더만 학습하고, 유사도 가중 앙상블을 한다(A3 확인). | SEED 세션 1 LOSO 86.7±7.1%(보정을 빼면 85.4%). 보정 구간을 시험에서 뺐는지는 명시가 없다. | 짧은 비라벨 캘리브레이션 블록 프로토콜의 선례다. 학습 단계에서 배포를 흉내 내지는 않는다. 공저 연구실의 후속 보정 연구는 OpenAlex 저자 목록에서 찾지 못했다. | 중간 (프로토콜) |
| Li Z., "Reducing the Calibration Effort of EEG Emotion Recognition using Domain Adaptation with Soft Labels", 2021, IEEE EMBC 2021 | https://doi.org/10.1109/EMBC46164.2021.9629649 | 적대 DA에 소프트 라벨 손실을 더해, 보정 세션의 소량 라벨로 적응한다. | SEED, 보정 세션에서 **시행마다 라벨 15개**로 평균 87.28%(초록). 시험 표본이 보정 표본과 같은 시행에서 나오는지는 **(미확인)**. | 라벨 캘리브레이션 계열. CAFT는 감정 균형 클립(자극 라벨) 평균만 쓴다. | 낮음~중간 |
| Luo G., "Discriminative Knowledge Fuzzy Transfer Learning Guided by Resting-State EEG for Cross-Subject Emotion Recognition" (DKFTL-R), 2026, IEEE Trans. Fuzzy Systems 34(8):2614 | https://doi.org/10.1109/TFUZZ.2026.3696832 | 대상 피험자의 **휴지기 EEG**로 개인 신경 특성을 잡아 원천 피험자를 고르고, 감정 양식 투영 정렬과 TSK 퍼지 분류기를 쓴다. 대상의 과제 EEG는 필요 없다. | DEAP·DENS 58.79/55.89/62.91/60.42%, 자체 BHE-EMO 2클래스 67.00%, 3클래스 44.92%(초록). | "과제 없는 짧은 보정"의 최신 감정 사례다. CAFT는 감정 영상 캘리브레이션 블록을 쓰고 학습 안에서 배포를 흉내 낸다. | 낮음~중간 |
| Jin M., "EvoFA: Evolvable Fast Adaptation for EEG Emotion Recognition", 2024, arXiv 2409.15733 | https://arxiv.org/abs/2409.15733 | 세션을 분리한 메타학습 퓨샷이다(학습 에피소드로 배포 적응을 흉내 냄)(A3 확인). | SEED 피험자 내 교차 세션: 1샷 84.27%(ProtoNet 84.07%). | "에피소드 학습 = 배포 흉내"의 원리가 같다. 연산은 다르다(경사 적응·프로토타입 대 중심화). | 중간 |
| Liu H., "FACE: Few-shot Adapter with Cross-view Fusion for Cross-subject EEG Emotion Recognition", 2025, arXiv 2503.18998 | https://arxiv.org/abs/2503.18998 | 메타학습 퓨샷 어댑터(A3 확인). | 시행 내 무작위 분할(같은 시행의 인접 창이 지원·시험 양쪽에 섞임). SEED-V 3/5/10샷 89.55/95.20/98.95. | 비교 불가 수치의 전형이다(C절). | 낮음 |

### A-5. SEED-V 공정 비교 기준 · 평가 함정

| 논문 | 링크 | 무엇을 하나 | 프로토콜·수치 | CAFT와의 관계 | 새로움 위협 |
|---|---|---|---|---|---|
| Qing J. & Li L., "Context-aware tokenization for Cross-subject Emotion Decoding from EEG" (MST), 2026, arXiv 2606.00884 | https://arxiv.org/abs/2606.00884 | 같은 시행 이웃 창(라벨 없음, 앞뒤 5개)의 Morlet 스펙트럼 평균을 기준으로 목표 창 토큰을 조건화한다(목표와 기준의 차이를 함께 넣음). | **엄격한 LOSO**: 보정·대상 미세조정 없음, 원천 통계로만 정규화(LaBraM은 μV/100), 검증·체크포인트 선택 없이 마지막 스텝, 2초 창, SEED-V 16명 3세션. **SEED-V 창 정확도**: MST 36.1±4.5, LaBraM(공개 사전학습 가중치를 원천 피험자로 파인튜닝, 평균 풀링 + 선형 헤드) 32.5±4.9, CBraMod 31.5±5.1, CSBrain 31.0±4.4, BIOT 30.6±4.9, EEGPT 28.6±3.0. SEED: LaBraM 62.4±7.1, MST 66.5±8.1. 코드 공개 문구는 없다. | 우리 "적응 없음" **창 단위**와 가장 공정하게 비교할 외부 기준이다. "같은 시행 이웃 기준과의 차이"는 시험 시점 자기 기준 빼기와 닮았다(시행 내 전이적). | 낮음 |
| Liu H., "LibEER: A Comprehensive Benchmark and Algorithm Library for EEG-based Emotion Recognition", 2025, IEEE TAFFC 16(4):3596 | https://doi.org/10.1109/TAFFC.2025.3605833 (arXiv 2410.09767), 코드 https://github.com/XJTU-EEG/LibEER | 감정 모델 18개를 통일 전처리·분할·선택으로 평가한다. 원 논문이 "시험 집합 최고 에폭"을 보고하는 관행을 "심각한 결함"이라고 지적한다. | 교차 피험자: 피험자 60/20/20 분할(LOSO 아님), 검증 F1로 체크포인트 선택, DE+LDS 1초 창, SEED-V는 3세션. **SEED-V 교차 피험자 정확도(arXiv v3 표 VI)**: MS-MDA 44.33±12.09(최고), NSAL-DGAT 41.56, R2G-STNN 37.23, BiDANN 36.58, RGNN 36.45, DGCNN 36.25, PR-PL 33.77, SVM 28.03, DANN 23.99, DAN 20.82. **SEED**: MS-MDA 64.00, DANN 63.89, RGNN 59.45, PR-PL 56.33. 원 논문 보고치보다 20–35%p 낮다. | SEED-V 교차 피험자 수치가 우리 범위와 같은 자릿수라는 근거다. 기준선을 한 번에 돌리는 도구로도 쓸 수 있다. | 낮음 |
| Gerster M., "Stimulus identity rather than emotion drives EEG classification on the FACED dataset", 2026, bioRxiv | https://doi.org/10.64898/2026.06.12.731889, 코드 https://github.com/moritz-gerster/faced-stimulus-confound | LinearSVC와 CLISA로 FACED의 혼입을 점검한다. | **교차 피험자 9감정**: 기본 39.4±1.1%. 감정당 영상 1개로 줄이면(데이터 1/3) 44.9±1.6%로 오른다. 주관 라벨로 바꾸면 26.8±0.9%. 느낀 감정과 할당 감정이 다른 시행(36.0%)과 같은 시행(33.9%)의 정확도가 비슷하다. CLISA 재현은 34.2±0.8%. 원 방법은 시험 폴드로 하이퍼파라미터를 골랐다(낙관 편향). 피험자 내 평가(영상 안 시간 분할)에서도 감정당 영상 1개로 줄이면 58.8%에서 71.3%로 오른다. | SEED-V도 감정마다 클립이 적고(세션당 감정마다 3개) 라벨이 자극에 할당된다. 그래서 **②가 자극 정체 학습을 키울 위험**이 있다. 미지 자극 시험이 필요하다. | 낮음 (새로움) / 평가 위협 높음 |
| Brookshire G., "Data leakage in deep learning studies of translational EEG", 2024, Frontiers in Neuroscience 18:1373515 | https://doi.org/10.3389/fnins.2024.1373515 | 구간 단위 무작위 분할과 피험자 단위 분할을 비교했다. | 구간 분할에서는 피험자 고유 패턴이 누출돼 정확도가 부풀려진다(임상 EEG). | 창 단위 무작위 분할을 피하는 근거다(A3의 블록 설계·RSC와 함께). | 낮음 |
| Zhang S., "EEG-PRIME: Prototype-Aligned Representation Learning with Multi-Level Conditioning for EEG Decoding", 2026, arXiv 2608.13072 | https://arxiv.org/abs/2608.13072 | 마스크 사전학습, 프로토타입 정렬 지시 조정, 피험자 불변 적대 정규화를 쓰고, 클래스 텍스트 임베딩 프로토타입과의 코사인으로 예측한다. | "교차 피험자" 표에 넣었지만 **SEED-V는 같은 피험자의 시행 1–5로 학습하고 6–10으로 시험(교차 시행)**한다. SEED-V BAcc 0.4051. | 프로토타입 코사인 분류와 피험자 불변 학습을 FM에서 함께 쓴 최신 예다. 감정 분할이 피험자 내라 비교할 수 없다. | 낮음 |

참고: 2026년 서베이 Li T. 외, "Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods"(Progress in Biomedical Engineering, doi:10.1088/2516-1091/ae65f0, arXiv 2604.27033)는 "피험자별 정규화" 범주를 따로 두었다. 이 범주에 휴지기 기준 보정(InvBase, Kwak BCM), 자기 비교(DMNet), SPD 도메인별 배치 정규화(Kobler)를 묶었고, CLISA는 "공유 자극" 범주에 넣었다. CAFT의 related work 분류 틀로 쓸 수 있다.

---

## (B) 가장 가까운 선행 연구 정밀 비교

### B-0. 구성 요소별 대조

○는 그 요소가 있음, ×는 없음, △는 부분적, —는 해당 없음을 뜻한다.

| 구성 요소 | CLISA 2022 | mdJPT 2025 | DAEST 2025 | 층화 정규화 Fdez 2021 | Kwak 2023 (MI) | PPDA 2021 | **CAFT** |
|---|---|---|---|---|---|---|---|
| 배치 = 여러 피험자 × 클립마다 같은 (클립, 시각) 하나 | ○ (2명) | ○ (2명, 같은 세션 강제, 코드 확인) | ○ (2명) | △ (학습 전체가 한 배치, 시행 단위) | × | × | ○ (4명, 같은 세션) |
| 배치 안 피험자별 정규화 | ○ z-점수, 입력·중간층 | ○ z-점수, 인코더 안(코드) | ○ (코드, 선택) | ○ 참가자·세션별, 세 층 | ○ (기준 특징 빼기) | × | ○ 평균 빼기, 분류 토큰 |
| 정규화 기준 집합 | 배치 안 그 피험자 표본(모든 시행) | 같음 | 같음 | 그 세션 15시행 전체 | 별도 휴지기 기록 | — | 배치 안 그 피험자 15창(감정 균형) |
| 정규화된 표현에 지도 교차 엔트로피 | × (대조) | × (ISA + CDA) | × (대조) | ○ | ○ | — | ○ |
| 같은 자극·같은 시각의 피험자 간 손실 | ○ NT-Xent | ○ ISA(시각 정렬이 필수, 부록 S5) | ○ | × | × (클래스 기준 불변 손실) | × | ○ 일관성 손실 |
| 배포 때 새 사용자 정규화 통계 | 시험 스트림 온라인 갱신 | 시험 스트림 온라인 갱신(코드) | 같음 | 시험 세션 전체 | 1분 휴지기 | 첫 45초 | **짧은 감정 균형 캘리브레이션 블록** |
| 학습 연산 = 배포 연산 | × | × | × | ○ (단 전이적) | ○ | — | ○ |
| 범용 EEG FM 전체 파인튜닝 | × | × (자체 사전학습) | × | × | × | × | ○ (LaBraM-base) |
| 비전이적 + 캘리브레이션 클립 평가 제외 | × | × | × | × | ○ (휴지기 별도) | (미확인) | ○ |
| 클립 단위 지표 | × (창 + LDS) | × | × | ○ (시행 단위 특징) | — | × | ○ (창 단위도 보고) |

**읽는 법.**
- CAFT의 왼쪽 다섯 줄(배치, 배치 안 정규화, 같은 시점 손실)은 CLISA·mdJPT에 이미 있다.
- "학습 연산 = 배포 연산"과 "지도 CE"는 층화 정규화와 Kwak에 있다.
- CAFT에만 있는 것은 아래 네 줄의 **조합**이다. 짧은 감정 균형 캘리브레이션 블록, 배포 연산과의 일치, 범용 FM 파인튜닝, 비전이적·클립 분리 평가가 그것이다.

### B-1. CLISA (Shen 외, IEEE TAFFC 2022/2023) — 높음

- **같은 점**
  - 데이터 샘플러가 "피험자 A, B × 각 시행에서 같은 시간 구간 표본 하나"로 미니배치를 만든다. CAFT의 "같은 세션 학습 피험자 4명 × 같은 (클립, 초) 15곳(감정마다 3곳)"과 구조가 같다. SEED·SEED-V에서 세션당 클립이 15개이므로 클립마다 하나씩이면 정확히 15곳이다.
  - 학습 중 층화 정규화는 "한 피험자의 배치 안 표본을 채널별로 모아 z-점수"를 낸다. 감정이 균형 잡힌 15표본 통계로 피험자 오프셋을 지운다는 점에서 ①과 같다.
  - 같은 시행·같은 구간 쌍을 양성으로 하는 대조 손실은 ②와 같은 목적이다.
- **다른 점**
  - 인코더를 자기지도 대조로만 학습한다. 중심화된 표현에 지도 CE를 걸지 않는다.
  - 정규화가 입력과 중간층의 z-점수다. CAFT는 분류 토큰 z의 평균 빼기만 한다.
  - 분류기(DE, LDS, MLP)는 따로 학습하고, 시험 정규화는 시험 스트림 온라인 갱신이다. 학습 정규화와 배포 정규화가 다르다.
  - 범용 FM이 아니고, 새 사용자 캘리브레이션 블록이 없다.
- **CAFT가 보여야 할 것**
  - 같은 샘플러에서 "CLISA식 층화 z-점수(입력·중간층) + 대조"와 "CAFT 출력 중심화 + 지도 CE + 일관성"을 같은 배포 프로토콜로 비교해야 한다.
  - 이득이 "배포 일치" 덕분인지 보여야 한다. 예: 학습 때 중심화 기준을 캘리브레이션과 비슷한 부분집합(감정마다 1클립)으로 바꿨을 때와 15창 전체로 했을 때를 비교한다.

### B-2. mdJPT (Zhang 외, NeurIPS 2025) — 높음 (+ 같은 계보 DAEST·TA2CL)

- **같은 점** (코드로 확인)
  - 샘플러가 두 피험자를 **같은 세션**에서 고르고(assert), 시행마다 **같은 시간 인덱스**를 쓴다.
  - 인코더 순전파 안에서 배치를 피험자 단위로 나눠 z-점수를 낸다(`stratified_layerNorm`).
  - ISA는 같은 시행·같은 시작 시각 쌍을 양성으로 하는 대조 손실이다. 시각 정렬을 빼면 성능이 크게 떨어진다(부록 S5).
  - CDA는 배치 안에서 피험자별 공분산 중심을 맞춘다. 2차 통계 정렬을 학습 손실로 쓴 것이다.
  - 감정 전용 FM 사전학습이고, LaBraM·EEGPT를 기준선으로 이긴다.
- **다른 점**
  - 지도 파인튜닝 단계에서 중심화된 표현으로 분류하지 않는다. 동결 인코더 특징에 온라인 정규화, LDS, MLP를 쓴다.
  - 평가가 교차 데이터셋이다. 대상 피험자 1/4의 라벨로 분류기를 학습하거나 제로샷이며, 새 사용자 캘리브레이션 블록은 없다.
  - 범용 FM(LaBraM)을 파인튜닝하지 않는다.
- **DAEST(TAFFC 2025)**는 같은 샘플러와 층화 정규화(코드)에 동적 주의 인코더를 쓴 SEED-V 대표 수치다(73.6±12.7). **TA2CL(2026)**은 같은 샘플러에서 시차 허용 대조로 바꿨다(SEED-V LOSO 70.1±12.2).
- **CAFT가 보여야 할 것**
  - ②를 ISA(대조)와 비교한다. 예: 같은 LaBraM 파인튜닝에 ISA를 보조 손실로 붙인 조건.
  - ②를 시차 허용 버전(TA2CL식 top-K 매칭, ±1–3초)과 비교한다.
  - CDA식 2차 통계 정렬을 더하면 추가 이득이 있는지 확인한다.

### B-3. 층화 정규화 (Fdez 외, Frontiers in Neuroscience 2021) — 높음 (① 한정)

- **같은 점**: 지도 학습 안에서 참가자·세션별 정규화를 쓰고, 시험 때도 같은 정규화를 쓴다. 학습 연산과 배포 연산이 같다. 정규화 통계가 감정 균형 집합(그 세션 15시행)이고, 정확도가 사실상 클립 단위(시행 단위 특징)다.
- **다른 점**
  - 시험 통계가 시험 세션 15시행 전체다(전이적).
  - 손특징(대역 전력·DE) MLP다.
  - 학습 전체를 한 배치로 쓴다. 에피소드형 미니배치나 같은 자극 시점 손실이 없다.
- **CAFT가 보여야 할 것**
  - 층화 정규화를 "캘리브레이션 블록 통계로 정규화하는 비전이적 변형"으로 돌린 기준선을 넣는다.
  - CAFT의 비전이적 결과와 시험 세션 전체 통계를 쓴 전이적 상한을 나란히 보고한다.

### B-4. Kwak 외 (IEEE JBHI 2023, 운동 상상) — 중간~높음 (분야 밖)

- **같은 점**: 배포 때 쓸 짧은 새 사용자 기록(1분 휴지기)으로 피험자 변이 성분을 빼는 모듈을 **학습 안에서** 같이 학습한다. 같은 클래스 특징이 피험자와 무관하게 모이도록 불변 손실을 건다. 배포 때 과제 캘리브레이션이 없다. "캘리브레이션 인지 학습"이라는 원리가 CAFT ①과 가장 비슷하다.
- **다른 점**: 운동 상상이다. 기준이 감정 자극이 아닌 휴지기이고, 클래스 기준 손실이며, FM이 아니다.
- **인용 방식**: 분야 밖의 원리 선례로 인용한다. 감정에서 휴지기 대신 감정 균형 캘리브레이션 클립을 쓰는 이유는 InvBase와 DKFTL-R 대비로 설명한다. 실험으로는 "휴지기만 또는 중립 클립 하나로 중심화" 대조가 A4가 권한 실험과 겹친다.

### B-5. PPDA (Zhao 외, AAAI 2021) — 중간 (프로토콜)

- **같은 점**: 테스트와 분리된 짧은 캘리브레이션 블록(세션 첫 45초)만 쓴다.
- **다른 점**
  - 학습 단계에서 배포를 흉내 내지 않는다.
  - 블록에서 새 사용자의 개인 인코더를 따로 학습한다.
  - SEED는 순서가 고정이라 이 블록은 첫 클립 하나, 즉 감정 하나다(A3의 해석).
- **인용 방식**: 현실적 캘리브레이션 프로토콜의 원조로 인용한다. 우리 프로토콜이 "감정 포괄(감정마다 1클립)과 초 단위 예산"을 통제한다는 차이를 강조한다.

### B-6. 정직한 새로움 문장 (제안)

- **쓸 수 있는 주장 (검색 범위 안, "to our knowledge"로 한정)**
  > "We make the fine-tuning of an EEG foundation model *calibration-aware*: each mini-batch contains several training subjects from the same session viewing the same (clip, second) positions, every subject's embeddings are centered with its own emotion-balanced in-batch mean before the classification head — the exact operator applied at deployment with a short calibration block — and a cross-subject stimulus-time consistency term is added. The batch sampler, in-batch subject-wise normalization and stimulus-locked cross-subject objectives follow CLISA (Shen et al., 2022) and mdJPT (Zhang et al., 2025); unlike them, CAFT trains the supervised head on exactly the deployment-time centered representation, and is evaluated non-transductively with calibration clips excluded from testing."
- **쓰면 안 되는 주장**
  - "같은 자극·같은 시점 배치를 처음 썼다" (CLISA, mdJPT)
  - "배치 안 피험자별 정규화를 처음 썼다" (CLISA, mdJPT, Fdez)
  - "학습과 배포의 정규화를 일치시킨 첫 연구" (Fdez, Kwak, Bakas)
  - "피험자 간 같은 자극 시점 손실이 새롭다" (CLISA, CL-SSTER, mdJPT, TA2CL)

---

## (C) 공정 비교 표 초안과 주의점

### C-1. SEED-V(5감정, 우연 20%) 교차 피험자 수치 정리

| 출처 | SEED-V 수치 | 입력 | 분할 | 시험 피험자 데이터 사용 | 시간 평활 | 지표 단위 | 모델 선택 | 우리와 비교 |
|---|---|---|---|---|---|---|---|---|
| MST (Qing & Li, arXiv 2026) | LaBraM 32.5±4.9, CBraMod 31.5±5.1, MST 36.1±4.5 | 원신호 2초 | LOSO, 16명, 3세션 | 없음(원천 통계. MST는 같은 시행 이웃 창을 문맥으로 씀) | 없음 | 창 | 고정 스텝 마지막 | **가장 공정.** 우리 "적응 없음" 창 단위와 바로 비교 |
| LibEER (TAFFC 2025) | 20.8–44.3 (MS-MDA 44.33±12.09 최고, DGCNN 36.25, RGNN 36.45, DANN 23.99, DAN 20.82) | DE + LDS 1초 | 피험자 60/20/20 (LOSO 아님), 3세션 | DA 계열이 대상 데이터를 쓰는지 **(미확인)** | LDS(시행 전체) | 창(평활) | 검증 피험자 F1 | 범위 비교 가능. 분할·평활 차이를 명시 |
| MAT (arXiv 2025) | 단일 세션: SVM 53.14, DANN 56.28, DMMR 58.63, MAT 63.48 / 교차 세션: SVM 41.20, DMMR 55.16, MAT 58.39 | DE + LDS 1초 | LOSO, 첫 세션만 / 교차 세션 | "보지 않은 대상"이라 주장. 특징 정규화 방식은 서술 없음 | LDS | 창(평활) | 서술 없음 | 조건부. LDS·세션 차이 큼 |
| CLISA·DAEST·TA2CL | DE+MLP 59.3±17.2, CLISA 67.3±13.0, DAEST 73.6±12.7(재현 68.2±12.4), TA2CL 70.1±12.2 | 학습 인코더 출력의 DE 1초 | TA2CL LOSO, DAEST 코드 기본 10겹 | **있음**: 시험 스트림 온라인 정규화(감쇠 0.99) | LDS | 창(평활) | DAEST 코드는 빼 둔 피험자 정확도의 에폭 최댓값 | **직접 비교 불가.** 우리 프로토콜로 재실행 필요 |
| mdJPT (NeurIPS 2025) | 소수 피험자: LaBraM 41.80, mdJPT 65.02 / 제로샷: LaBraM 39.70, mdJPT 52.91 | 원신호 → 인코더 평균 5초 | 데이터셋 하나 빼기. 대상 피험자 1/4로 분류기 학습 | **있음**: 온라인 정규화(코드) | LDS(코드, RTS 역방향) | 창(평활) | (미확인) | 비교 불가(설정이 다름) |
| MoGE (BIBM 2024) | 81.8 | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) | 확인 전 비교 금지 |
| FM 벤치마크 (A4) | LaBraM 0.4095, CBraMod 0.4091 (BAcc) | 원신호 | **피험자 내** 시행 5:5:5 | — | 없음 | 창 | 검증 | 비교 불가 |
| EEG-PRIME (arXiv 2026) | BAcc 0.4051 | 원신호 | **피험자 내** 교차 시행(1–5 → 6–10) | — | — | 창 | — | 비교 불가 |
| FACE (arXiv 2025) | 3/5/10샷 89.55/95.20/98.95 | DE | 시행 내 무작위 퓨샷 | 라벨 퓨샷 | — | 창 | — | 비교 불가 |
| **우리 (CAFT 연구)** | 클립: 적응 없음 0.30–0.36, 중심화 0.37–0.46, CAFT ① 약 0.46(5명), SLA 포함 0.47–0.54 | 원신호, LaBraM-base 전체 파인튜닝 | LOSO + 학습 피험자 2명 검증 | 캘리브레이션 블록만(비전이적). 캘리브레이션 클립은 평가에서 제외 | 없음 | 클립과 창 | 검증 피험자 | — |

### C-2. 비교할 때 주의점

1. **창 단위 대 클립 단위, 그리고 LDS.** 문헌의 SEED 계열 "창 정확도"는 대부분 시행 전체에 LDS(칼만 필터 + RTS 역방향 평활, 비인과)를 건 뒤 잰 값이다. mdJPT 코드에서 `givenAll=1` 역방향 평활을 확인했다. 그래서 시행 단위 정보가 섞여 있어, 순수 창 정확도와 클립 정확도의 중간쯤이다.
   - 우리 창 단위 수치는 MST·LibEER의 원신호·무평활 수치와 비교한다.
   - 우리 클립 단위 수치는 "LDS 평활 창 정확도"와 비교하지 말고, 우리 프로토콜로 다시 돌린 기준선의 클립 정확도와 비교한다.
2. **전이적 정규화.** 시험 피험자 통계가 쓰이는 사례는 다음과 같다.
   - CLISA·DAEST·mdJPT·TA2CL: 시험 스트림 온라인 갱신(인과적이지만 전이적)
   - DMMR·MS-MDA 코드: 피험자별 세션 전체 최소·최대(비인과, 전이적)
   - 층화 정규화: 시험 세션 전체
   - Apicella: 시험 피험자·세션 통계

   우리 비전이적 결과와 나란히 놓으려면 같은 방법을 "캘리브레이션 블록 통계" 변형으로 다시 돌리거나, 우리 쪽 전이적 상한(시험 세션 전체 중심화)을 함께 보고해야 한다.
3. **시험 기반 모델 선택.** DMMR(`train.py`), MS-MDA(`msmdaer.py`), DAEST(`train_mlp.py`)의 공개 코드는 빼 둔 시험 피험자의 에폭별 정확도 최댓값을 보고한다. LibEER은 이 관행이 성능을 "크게 과장"한다고 썼고, Suo 2026(A3)은 후보를 5개에서 80개로 늘리면 +6.24%p가 된다고 보였다. 우리는 학습 피험자 2명으로 검증하므로 이 점을 본문에 명시한다.
4. **세션·피험자 수.** MAT, DMMR 등은 "첫 세션만" 쓰고 우리는 3세션이다. SEED-V 피험자 수도 문헌마다 다르다. 공개판은 16명이다(MST·MAT·mdJPT·DAEST 코드). DAEST 본문은 20명이라고 적었다. 표에 세션 수와 피험자 수를 항상 함께 적는다.
5. **자극 정체 혼입.** SEED-V의 LOSO에서는 시험 클립을 학습 피험자도 봤다. 그래서 교차 피험자 정확도에 "클립 지문 인식"이 섞일 수 있다.
   - CLISA는 SEED에서 미지 자극 조건 86.4%가 77.4%로 떨어졌다.
   - Gerster 2026은 FACED에서 감정당 영상 1개로 줄이면 오히려 정확도가 오르고, 주관 라벨이면 떨어진다고 보였다.

   CAFT ②(같은 (클립, 초) 일관성)는 이 혼입을 키울 수 있다. 그러니 (a) 시험 클립을 학습 피험자 데이터에서도 빼는 미지 자극 분할, (b) 같은 클립의 다른 초와 짝짓는 시간 셔플 대조를 보고해야 한다.
6. **캘리브레이션 데이터 누출.** 캘리브레이션 클립을 평가에서 뺐다고 명시한다(PPDA는 명시가 없음). 캘리브레이션 블록의 감정 구성이 균형인지(감정마다 1클립)도 적는다. 불균형 문맥에서는 평균 중심화가 붕괴할 수 있다(A2의 Bakas).
7. **보고 형식.** 피험자별 분포, 평균±표준편차, 피험자 짝 비교를 보고한다(MST 방식). 지표는 클립과 창을 둘 다 쓰고, 가능하면 BAcc도 함께 쓴다(FM 벤치마크와 비교하기 위해).

---

## (D) 리뷰어가 요구할 기준선

| 우선 | 기준선 | 왜 필요한가 | 공개 코드 | 우리 프로토콜로 돌릴 때 고칠 점 |
|---|---|---|---|---|
| 1 | 보정 없는 LaBraM 전체 파인튜닝(ERM) | 하한이자 MST와 같은 조건의 외부 대조 | 있음: https://github.com/935963004/LaBraM | 시험 통계를 쓰지 않음. 마지막 체크포인트 또는 검증 피험자로 선택 |
| 1 | 캘리브레이션 중심화만(배포 연산만, CAFT 학습 없음)과 전이적 상한(시험 세션 전체 중심화) | CAFT 학습 자체의 기여와 상한을 분리 | 내부 | — |
| 1 | **CLISA/mdJPT식 학습**: 같은 샘플러 + 층화 z-점수 정규화 + 같은 시점 대조(ISA)를 LaBraM 파인튜닝에 보조로 붙임 | 가장 가까운 선행(높음)을 같은 백본·프로토콜로 비교 | 있음: mdJPT https://github.com/ncclab-sustech/mdJPT_nips2025, DAEST https://github.com/RunminGan1218/DAEST, CLISA는 FACED와 함께 Synapse(doi:10.7303/syn50614194) | 온라인 시험 정규화와 LDS를 빼고 캘리브레이션 블록 통계로 통일 |
| 1 | 층화 정규화(Fdez 2021) | ①의 직접 선례(높음) | 있음: https://github.com/javiferfer/cross-subject-eeg-emotion-recognition-through-nn | 시험 정규화를 시험 세션 전체가 아니라 캘리브레이션 블록 통계로. 전이적 원판은 상한으로 따로 보고 |
| 1 | 도메인 적대(DANN, 피험자 판별기) 또는 다중 소스 DA(MS-MDA) | 리뷰어가 가장 흔히 요구하는 전이 학습 기준선 | 있음: MS-MDA https://github.com/VoiceBeer/MS-MDA. DANN, BiDANN, R2G-STNN, RGNN, DGCNN, PR-PL은 LibEER(https://github.com/XJTU-EEG/LibEER)에 구현돼 있음 | 대상 도메인 데이터를 시험 데이터가 아닌 **캘리브레이션 블록**으로 바꾼 변형을 주 비교로 하고, 전이적 원판은 참고로. 시험 최댓값 선택(MS-MDA 코드)을 제거 |
| 2 | DG: DMMR 또는 MAT | "보정 없이 일반화" 계열의 최신 대표(SEED-V 수치 있음: MAT) | 있음: https://github.com/CodeBreathing/DMMR, https://github.com/WuCB-BCI/MAT | DMMR은 시험 피험자 최소·최대 정규화와 시험 최댓값 선택을 고쳐야 함 |
| 2 | PPDA(짧은 비라벨 캘리브레이션) | 현실적 캘리브레이션 프로토콜의 원조 | 공식 코드는 찾지 못함 → 재구현 필요 | 캘리브레이션 블록을 우리 블록(감정마다 1클립)으로 통일 |
| 2 | 에피소드형 퓨샷(ProtoNet, EvoFA식): 캘리브레이션 클립을 지원 집합으로 | "학습에서 배포를 흉내" 원리의 다른 연산 | EvoFA 코드 공개 여부 **(미확인)**. ProtoNet은 표준 구현 | 지원 집합 = 캘리브레이션 클립, 질의 = 나머지 클립(클립 분리) |
| 3 | 시차 허용 일관성(TA2CL식 top-K 매칭) | ②의 "같은 초" 가정 점검 | 코드 공개 문구 없음 → 재구현 | ±1–3초 범위 |
| 3 | 분야 밖 원리 대조: Kwak BCM식 "기준 특징 빼기 모듈 학습" | ①의 학습형 대안 | **(미확인)** | 기준 = 캘리브레이션 블록 |

**최소 세트를 권한다면 다섯 가지.** (1) 보정 없는 LaBraM, (2) 캘리브레이션 중심화만과 전이적 상한, (3) CLISA/mdJPT식 학습을 LaBraM에, (4) 층화 정규화의 캘리브레이션 변형, (5) DANN 또는 MS-MDA의 캘리브레이션 블록 대상 변형. 모두 공개 코드가 있거나 내부 구현이다. 여기에 미지 자극 분할과 시간 셔플 대조(C-2의 5번)를 같은 표에 넣는다.

---

## (E) 확인하지 못한 항목

- **MoGE(BIBM 2024)의 SEED-V 81.8% 프로토콜.** 원문이 비공개이고, 공개 코드에는 모델 정의만 있어 정규화·분할·모델 선택을 확인하지 못했다.
- **Personal-Zscore의 시험 피험자 통계 출처.** 원문 비공개(A3에서도 미확인).
- **Kwak 2023(JBHI)의 데이터셋·수치·코드.** 초록만 확인했고 arXiv판이 없다.
- **R2G-STNN 원 논문의 SEED 피험자 독립 수치.** 원문 비공개.
- **DAEST의 SEED-V 교차검증 방식(LOSO 대 10겹)과 피험자 수(본문 20명 대 코드 16명).** 논문 수치가 공개 코드의 `best_model_score` 경로로 나왔는지도 확인하지 못했다. TA2CL은 SEED-V를 LOSO로 평가했다고 적었다.
- **MAT의 특징 정규화와 모델 선택 방식.** 본문에 서술이 없다.
- **TA2CL과 MST의 코드 공개 여부.** 원문에 공개 문구가 없다.
- **LibEER의 DA 계열이 대상(시험) 피험자 데이터를 쓰는지, 그리고 저널판(TAFFC)과 arXiv v3 표 VI 수치가 같은지.**
- **CSCL(Sci Rep 2025)의 실제 분할·정규화.** 서술이 불명확하다. SEED 97.7%는 신뢰하기 어렵다.
- **Li Z. 외 EMBC 2021에서 보정 표본과 시험 표본이 같은 시행에서 나오는지.**
- **EvoFA 코드 공개 여부.**
- **mdJPT 부록 표 S5(시각 비정렬 양성 쌍)의 구체 수치, 그리고 제로샷 최근접 이웃 평가에서 같은 시행의 인접 창이 이웃 후보에서 빠지는지.** 원문에 명시가 없다. 빠지지 않는다면 과대평가일 수 있다.
- **실험 단위 배치 정규화(ELBN).** 논문은 Li G. 외, "Cross-subject EEG linear domain adaption based on batch normalization and depthwise convolutional neural network", Knowledge-Based Systems 280:111011, 2023, doi:10.1016/j.knosys.2023.111011이다. 서지만 확인했다. 방법(실험 단위 배치 정규화와 깊이별 합성곱의 선형 사상)과 대상 데이터 사용 여부는 검색 요약으로만 봐서 표에서 뺐다.
- **DEAP 관련.** DEAP의 시행 직전 3초 기준 빼기 관행(Yang 외, IJCNN 2018, doi:10.1109/IJCNN.2018.8489331, 서지 확인)이 우리 DEAP 실험의 중심화와 어떻게 겹치는지는 확인하지 않았다. DEAP의 영상 제시 순서가 피험자마다 같은지도 확인하지 않았다. ②를 DEAP에 쓰려면 이 점이 필요하다. DEAP은 라벨이 피험자별 자기 보고라서 같은 클립·같은 초라도 감정 라벨이 다를 수 있는데, 이것은 우리 추론이다.
- **SEED-V의 세션 간 클립 반복 여부** (A3에서도 미확인).
- **새로 나온 비슷한 연구가 빠졌을 가능성.** arXiv API와 Crossref가 조사 중 여러 번 요청 한도(429)를 걸어, 일부 검색은 웹 검색과 OpenAlex 인용 목록(PPDA 146편, CLISA 305편을 제목 수준으로 훑음)으로 대신했다. 2026년 하반기 학술지 논문 중 비슷한 연구가 빠졌을 수 있다. 투고 직전에 다음 검색어로 다시 확인하기를 권한다: "stratified normalization" + "foundation model", "inter-subject alignment" + "fine-tuning" + emotion, "calibration" + "emotion" + "EEG foundation model".
