# 고전 BCI 보정 시간 단축 문헌 조사 (운동 심상·ERP/P300·SSVEP/c-VEP, 2015–2026)

기준일은 2026-10-05입니다. 표에 실은 논문은 모두 PubMed eutils, Crossref, arXiv API, 출판사·학회 페이지 가운데 하나 이상에서 제목·저자·연도·학술지를 확인했습니다. LST(Chiang 2021)와 SSVEP-DAN(Chen 2024)은 원문 PDF를 직접 읽었습니다. 원문을 확인하지 못한 세부 내용에는 "(세부 미확인)"을 붙였고, 확인하지 못한 항목은 끝에 따로 모았습니다. 이번 세션 중간에 웹 검색 할당량(200회)이 소진되어, 이후 검증은 eutils, Crossref, arXiv API를 직접 조회해서 했습니다.

## 0. 핵심 결론

- **도메인 중심화는 새롭다고 주장할 수 없습니다.** 새 세션의 무라벨 평균을 빼는 방식은 고전 BCI에서 이미 표준입니다.
  - 선례: 선형판별분석(LDA)의 전체 평균 비지도 적응(Vidaurre 2011), 리만 재중심화(Zanini 2018), 유클리드 정렬(Euclidean alignment, EA; He & Wu 2020), 대칭 양의 정부호(SPD) 행렬용 도메인별 배치정규화(Kobler 2022).
  - 특히 Mellot 2023(Imaging Neuroscience)은 "새 피험자에는 재중심화가 가장 폭넓게 유효하고, 회전 보정은 짝지은 데이터가 있을 때 필수"라는, 우리 발견 (1)·(3)과 거의 같은 결론을 이미 냈습니다.
- **SLA의 핵심 아이디어는 SSVEP에서 10년 된 주류 계보입니다.** "새 사용자가 자극에 동기된 보정 응답을 내고, 이를 다른 피험자의 같은 자극 템플릿과 짝지어 변환을 추정한다"는 발상은 정상상태 시각유발전위(SSVEP)에서 다음 순서로 발전했습니다.
  - 전이 템플릿 정준상관분석(tt-CCA, 2015) → 최소제곱 변환(least-squares transformation, LST; 2019 학회 / 2021 저널) → 피험자 전이 CCA(stCCA, 2020) → 소량 데이터 LST(sd-LST, 2023) → 다중 자극 LST+온라인 적응(ms-LST-OA)·SSVEP-DAN·OS-SSVEP(2024) → SQ-HAF(2026).
  - 직교 프로크루스테스 회전도 ALPHA(2022)가 SSVEP에서 이미 사용했습니다.
  - 수학적으로 SLA에 가장 가까운 것은 fMRI의 하이퍼정렬(Haxby 2011: 영화 시점별 대응 + 직교 프로크루스테스)과 공유 응답 모델(shared response model, SRM; Chen 2015)입니다.
- **감정 EEG에도 직접 선례가 있습니다.**
  - CLISA(Shen 2022, IEEE TAFFC): 같은 영화 구간을 본 피험자들의 표현을 학습 단계에서 정렬합니다. SEED도 사용했습니다.
  - Group Resonance Network(GRN; Meng, ICANN 2026, arXiv:2603.11119): SEED 피험자 제외 교차검증(LOSO)에서 시험 피험자와 학습 피험자 3명 사이의 "같은 자극 시간축" 위상 동기화(PLV/coherence)를 추론 시점에 씁니다(87.90%).
  - 따라서 SLA의 차별점은 다음 문장으로 좁혀야 합니다. **"자극 시간축은 보정 클립에서만 쓰고, 시험 데이터에는 자극 정보가 필요 없으며(비전이적), 고정된 파운데이션 모델 임베딩을 회전한다."**
- **"감정 라벨 없이"라는 표현은 위험합니다.** SEED 계열에서는 클립 정체가 곧 감정 라벨입니다. SSVEP 문헌의 기준으로 보면 SLA는 라벨이 있는(자극 동기) 보정에 해당합니다. 클립 안 시간 섞기 같은 대조실험으로 시간 동기화 자체의 기여를 보여야 합니다(아래 C-3).

### 약어
- 유클리드 정렬(EA), 리만 프로크루스테스 분석(RPA), 라벨 정렬(LA), 접공간 정렬(TSA)
- 최소제곱 변환(LST), 정준상관분석(CCA), 과제 관련 성분 분석(TRCA), 과제 판별 성분 분석(TDCA)
- 정상상태 시각유발전위(SSVEP), 사건관련전위(ERP), 코드 변조 시각유발전위(c-VEP), 운동 심상(MI)
- 정보 전송률(ITR, 단위 bpm = bits/min), 상관 정렬(CORAL), 위상 고정 값(PLV)
- 공유 응답 모델(SRM), 자극 동기 정렬(SLA), 피험자 제외 교차검증(LOSO), 파운데이션 모델(FM)

---

## (A) 논문 표

**"보정 데이터" 열의 표기**

| 표기 | 뜻 |
|---|---|
| Z | 없음(무보정) |
| W | 같은 사용자의 과거 세션 또는 다른 장치 데이터 |
| U-off | 무라벨 대상 데이터, 오프라인·전이적(시험 데이터 포함 가능) |
| U-on | 무라벨 대상 데이터, 온라인 스트리밍(시험 시행을 인과적으로 사용) |
| L | 새 사용자의 라벨 있는 소량 데이터 |
| S | 자극 동기 시행. 자극 정체가 알려진 시행이며, SSVEP·ERP·감정 클립에서는 사실상 라벨 |

### A1. 정렬·재중심화 계열 (운동 심상·ERP 중심, 일부 범용)

| 인용 | 연도 | 학술지·학회 | 링크 | 보정 데이터 | 방법 (한 줄) | 핵심 수치 | 우리 연구와의 관련성 |
|---|---|---|---|---|---|---|---|
| Vidaurre, Kawanabe, von Bünau, Blankertz, Müller | 2011 | IEEE TBME 58(3):587–597 | [ResearchGate](https://www.researchgate.net/publication/224196044_Toward_Unsupervised_Adaptation_of_LDA_for_Brain-Computer_Interfaces) | U-on | LDA의 전체 평균(pooled mean)을 무라벨로 추적해 결정 경계를 평행이동 | 클래스와 무관한 비정상성을 상쇄(MI 온라인) | 도메인 중심화의 직접 선례 |
| Zanini, Congedo, Jutten, Said, Berthoumieu | 2018 | IEEE TBME 65(5):1107–1116 | [doi](https://doi.org/10.1109/TBME.2017.2742541) | U-off | 세션·피험자별 기준 공분산으로 공분산을 아핀 변환(리만 재중심화) | 교차 세션·교차 피험자에서 유의한 개선 | 입력 수준 중심화. 우리 중심화의 리만판 |
| Rodrigues, Jutten, Congedo (RPA) | 2019 | IEEE TBME 66(8):2390–2401 | [doi](https://doi.org/10.1109/TBME.2018.2889705) | 재중심화·늘이기는 U-off, 회전은 L | SPD 공분산에 평행이동·스케일·회전 적용. 회전은 원천과 대상의 클래스 평균을 맞추므로 대상 라벨 필요(pyRiemann TLRotate 문서로 확인) | 8개 데이터셋, 243명, 3개 패러다임 | 발견 (3)·(5)와 같은 구조: "중심화 뒤 남은 회전"을 클래스 평균으로 보정 |
| He & Wu (EA) | 2020 | IEEE TBME 67(2):399–410 | [doi](https://doi.org/10.1109/TBME.2019.2913914) | U-off(보정 구간만으로도 가능) | 새 피험자 시행 공분산의 산술평균으로 백색화. 비지도 | MI·ERP에서 리만 정렬보다 우수 | 우리가 추가 이득 약 0으로 확인한 입력 백색화의 원전 |
| He & Wu (LA) | 2020 | IEEE TNSRE 28(5):1091–1108 | [doi](https://doi.org/10.1109/TNSRE.2020.2980299) | L(클래스당 1개 이상) | 클래스 조건부 평균을 맞추는 라벨 정렬. 라벨 공간이 달라도 적용 | 클래스당 라벨 1개로 동작 | 발견 (5) "보정 클립 라벨로 클래스 프로토타입 회전·혼합"의 선례 |
| Mellot, Collas, Rodrigues, Engemann, Gramfort | 2023 | Imaging Neuroscience 1 | [doi](https://doi.org/10.1162/imag_a_00040) | U-off + 짝지은 데이터 | 공분산의 재중심·재스케일·회전(짝지은 프로크루스테스) 단계별 비교 | 새 피험자·새 데이터셋에는 재중심화가 핵심. 같은 피험자의 다른 과제에는 짝지은 회전이 필수 | 발견 (1)·(3)과 거의 같은 결론. 반드시 인용 |
| Junqueira, Aristimunha, Chevallier, de Camargo | 2024 | J Neural Eng 21(3):036038 | [doi](https://doi.org/10.1088/1741-2552/ad4f18) | U-off | EA와 딥러닝 조합을 체계적으로 평가 | 대상 피험자 +4.33%, 수렴 시간 −70% 이상 | EA 재평가. 우리의 "EA 추가 이득 약 0" 결과와 대비 |
| Wu (EA 재검토) | 2025 | J Neural Eng 22:031005 | [arXiv:2502.09203](https://arxiv.org/abs/2502.09203) | — | EA의 절차, 올바른 사용법, 확장을 정리 | 13개 BCI 패러다임에서 효과 | EA를 올바르게 썼는지(기준행렬 추정 구간) 근거로 인용 |
| Li S, Wang, Luo, Ding, Wu D (T-TIME) | 2024 | IEEE TBME 71(2):423–432 | [doi](https://doi.org/10.1109/TBME.2023.3303289) | U-on | 정렬된 원천 데이터로 분류기 앙상블을 초기화한 뒤, 스트리밍 시험 시행으로 조건부 엔트로피 최소화 | MI 3개 데이터셋에서 약 20개 전이 기법을 상회. "plug-and-play" 주장 | 인과적 시험 시점 적응의 대표. 시험 데이터를 쓰지 않는 우리 프로토콜이 더 엄격 |
| Li S, Kawanabe, Kobler (SPDIM) | 2025 | ICLR 2025 | [arXiv:2411.07249](https://arxiv.org/abs/2411.07249) | U-off(원천 데이터 무접근) | SPD 다양체에서 대상 도메인마다 매개변수 하나를 정보 최대화로 학습. 라벨 분포 이동 상황을 다룸 | 라벨 이동이 있을 때 단순 재중심화의 한계를 보완 | 우리 보정은 감정당 1클립으로 클래스 균형 → 중심화 편향이 작은 이유를 설명할 때 인용 |

### A2. SSVEP/c-VEP 템플릿 전이 (가장 중요한 부분)

| 인용 | 연도 | 학술지·학회 | 링크 | 보정 데이터 | 방법 (무엇을, 어느 방향으로, 어떤 사상으로) | 핵심 수치 | 관련성 |
|---|---|---|---|---|---|---|---|
| Yuan, Chen, Wang, Gao, Gao (tt-CCA) | 2015 | J Neural Eng 12(4):046006 | [doi](https://doi.org/10.1088/1741-2560/12/4/046006) | Z (온라인판 ott-CCA는 U-on) | 기존 피험자들의 자극별 평균 원신호 템플릿을 **변환 없이** 새 사용자의 CCA 기준으로 사용 | 데이터 길이 1.5 s에서 표준 CCA 대비 정확도 +18.78% | "다른 피험자의 자극 동기 템플릿" 개념의 기원 |
| Wong et al. (stCCA) | 2020 | IEEE TNSRE 28(10):2123–2135 | [IEEE](https://ieeexplore.ieee.org/document/9177172), [코드](https://github.com/edwin465/SSVEP-stCCA) | S/L: 일부 자극만, 총 9시행(자극 9개 × 1회) | 피험자 내: 새 사용자 공간필터를 다중 자극 CCA로 추정. 피험자 간: 원천 피험자들의 공간필터 적용 템플릿을 **가중합**하고, 가중치는 보정 자극에서 최소제곱으로 적합 | 9시행으로 ITR 198.18±59.12 bpm(Benchmark), 127.86±60.43 bpm(BETA) | 일부 클래스만 보정해 전체 템플릿을 구성 → "감정당 클립 1개"와 유사. 피험자 가중 참조 아이디어 |
| Chiang, Wei, Nakanishi, Jung (LST; 예비 연구 NER 2019) | 2021 | J Neural Eng 18(1):016002 | [doi](https://doi.org/10.1088/1741-2552/abcb6e), [arXiv:2102.05194](https://arxiv.org/abs/2102.05194) | S/L: 자극당 2–5회 × 40자극 | 원천 피험자의 **각 원신호 시행** x′를 새 사용자 템플릿 x̄(보정 시행 평균)로 보내는 **비제약 채널 선형사상** P = x̄x′ᵀ(x′x′ᵀ)⁻¹. 표본 단위 시간 대응 x(t) ≈ P x′(t)를 쓰고, 자극·하위대역·원천 시행마다 따로 추정. 변환된 시행을 TRCA 학습 데이터에 추가(원천 → 대상 방향) | 10명, 40클래스, 1.5 s. 자극당 2회 보정 시 약 52→61%(건식), 약 78→90%(습식). 그림 4에서 읽은 근사값 | **가장 가까운 선행.** SLA와의 차이: 방향(원천→대상), 수준(원신호), 제약 없음, 대응 단위가 자극=클래스 |
| Liu B, Chen X, Li X, Wang Y, Gao X, Gao S (ALPHA) | 2022 | IEEE TBME 69(2):795–806 | [doi](https://doi.org/10.1109/TBME.2021.3105331) | W: 같은 사용자의 습식 전극 데이터. 새 장치는 재보정 없음 | 공간패턴을 **직교 프로크루스테스**로 회전 정렬하고, 공분산을 CORAL로 정렬한 뒤 부분공간 풀링. 세부는 Liu X 2022 Front Neurosci 설명으로 확인 | 75명, 12타깃. tt-CCA·LST를 상회하고, 습식→건식에서는 완전 보정 TRCA도 상회 | SSVEP에서 직교 프로크루스테스 정렬의 선례. 단, 피험자 내 장치 간 전이 |
| Bian, Wu H, Liu B, Wu D (sd-LST) | 2023 | IEEE TNSRE 31:446–455 | [doi](https://doi.org/10.1109/TNSRE.2022.3225878) | S/L: 일부 자극 주파수만 소량 | 자극별 행렬 대신 **모든 자극에 공통인 단일 LST 행렬**을 추정하고, 변환된 원천으로 템플릿과 필터 구성 | 40타깃에서 보정 약 10회 | SLA처럼 "모든 클립에 공통인 변환 하나"를 쓰는 선례 |
| Chen SY, Chang, Chiang, Wei (SSVEP-DAN) | 2024 | IEEE TNSRE 32:2027–2037 | [doi](https://doi.org/10.1109/TNSRE.2024.3404432), [arXiv:2311.12666](https://arxiv.org/abs/2311.12666) | S/L: 자극당 2–4회(Benchmark) | 원천 시행 → 대상 템플릿(같은 자극의 보정 평균)으로 가는 **비선형 신경망**(공간 합성곱 + 채널별 완전연결 + tanh). 모든 자극을 함께 학습하고, 사전학습 뒤 원천 피험자별 미세조정 | Benchmark 자극당 2회: TRCA 74.68% → 91.56%. 건식(저품질) 원천에서는 LST가 음의 전이를 보임 | 학습형 정렬의 최신 사례. 음의 전이 경고 |
| Li D, Wang X, Dou, Zhao, Cui, Xiang, Wang B (ms-LST-OA) | 2024 | IEEE TNSRE 32:1606–1615 | [doi](https://doi.org/10.1109/TNSRE.2024.3387283) | S/L 소량 + U-on | 인접 자극 데이터까지 함께 써서 자극별 LST 행렬을 정밀화하고, 시행마다 온라인 적응 | ITR 210.01(Benchmark), 172.31(BETA), 139.04(UCSD) bpm | 소량 보정 + 온라인 적응의 결합 |
| Deng, Ji, Wang Y, Zhou SK (OS-SSVEP) | 2024 | Neural Networks 180:106734 | [doi](https://doi.org/10.1016/j.neunet.2024.106734), [arXiv:2311.07932](https://arxiv.org/abs/2311.07932) | S/L: 자극당 1회 | 다중 기준 LST로 원천과 대상을 모두 사인-코사인 기준 영역에 사상. 소스 에일리어싱 행렬 추정(SAME) 증강과 TRCA·TDCA를 결합 | 3개 데이터셋 중 2개에서 최고 성능 | 클래스당 1회(원샷) 보정의 현재 수준 |
| Wong, Wang, Nakanishi, Wang, Rosa, Chen, Jung, Wan (OACCA) | 2022 | IEEE TBME 69(6):2018–2028 | [doi](https://doi.org/10.1109/TBME.2021.3133594) | U-on (무보정) | 앞선 무라벨 시험 시행으로 공간필터를 온라인 갱신 | CCA ITR 94.60→158.87 bpm(데이터셋 I), 85.80→123.91 bpm(데이터셋 II) | 무보정 주장이 시험 데이터 흐름에 의존한다는 점에서 우리와 대비 |
| Luo et al. (SAME) | 2023 | IEEE TBME 70(6):1775–1785 | [doi](https://doi.org/10.1109/TBME.2022.3227036) | L 소량 + 증강 | 소스 에일리어싱 행렬을 추정해 인공 SSVEP 시행 생성 | 보정 시행이 적을 때 eTRCA 약 +12%, TDCA 약 +3%. 후속 msSAME(J Neural Eng 2023, [doi](https://doi.org/10.1088/1741-2552/ad0b8f))은 40타깃을 24 s로 보정, ITR 213.8 bpm | 생성형 보정의 대표 |
| Miao, Shi, Huang, Song, Chen, Wang, Gao (c-VEP) | 2024 | Expert Systems with Applications | [arXiv:2311.11596](https://arxiv.org/abs/2311.11596) | S: 단일 타깃을 약 1분 시청 | 타깃 하나를 응시하는 동안 공간·시간 패턴을 추출하고, 교차 피험자 전이로 시간 패턴 구성 | 보정 1분 미만으로 ITR 250 bpm | "자극 하나를 길게 보는" 보정의 BCI 쪽 유사 사례 |

### A3. 무보정·세션 간 전이·능동학습·파운데이션 모델 (MI/ERP/c-VEP)

| 인용 | 연도 | 학술지·학회 | 링크 | 보정 데이터 | 방법 | 핵심 수치 | 관련성 |
|---|---|---|---|---|---|---|---|
| Lotte | 2015 | Proc. IEEE 103(6):871–890 | [doi](https://doi.org/10.1109/JPROC.2015.2404941) | 전 범주 | 진동 기반 BCI의 보정 최소화·제거 신호처리 체계: 사용자·세션 간 전이, 사전지식 정규화, 인공 데이터 생성, 비지도 적응 | 리뷰 | 서론 프레이밍 |
| Wu D, Xu, Lu BL | 2022 | IEEE TCDS 14(1):4–19 | [arXiv:2004.06286](https://arxiv.org/abs/2004.06286) | 전 범주 | 2016년 이후 BCI 전이학습 리뷰. 교차 피험자·세션·장치·과제로 분류하고 감정 BCI도 포함 | 리뷰(ESI 고피인용) | 분류 체계 인용 |
| Krauledat, Tangermann, Blankertz, Müller | 2008 | PLoS One 3(8):e2967 | [doi](https://doi.org/10.1371/journal.pone.0002967) | W (과거 세션) | 과거 세션들의 원형 공간필터로 새 세션 보정을 생략 | 숙련 사용자 온라인 실험에서 성능 손실 없음 | 세션 간 전이의 고전 |
| Kindermans, Tangermann, Müller, Schrauwen | 2014 | J Neural Eng 11(3):035005 | [PubMed 24834896](https://pubmed.ncbi.nlm.nih.gov/24834896/) | Z + U-on | 교차 피험자 전이, 비지도 적응, 언어모델, 동적 정지를 결합한 ERP 철자기 | 22명. 지도 학습 최신 기법과 경쟁할 수준 | ERP 무보정의 대표. 시험 시행 흐름을 활용 |
| Wu D, Lawhern, Hairston, Lance (AwAR) | 2016 | IEEE TNSRE 24(11):1125–1137 | [doi](https://doi.org/10.1109/TNSRE.2016.2544108) | 이전 장치의 라벨 + 새 장치에서 능동 선택한 소량 라벨 | 가중 적응 정규화와 능동학습으로 새 헤드셋에서 필요한 라벨 수 축소 | 새 장치에 필요한 라벨 수 감소(ERP) | 능동학습형 보정의 대표 |
| Kim J & Kim SP | 2025 | IEEE TNSRE 33:3443–3454 | [PubMed 40880337](https://pubmed.ncbi.nlm.nih.gov/40880337/) | Z | 사전학습 xDAWN 공간필터 + CNN으로 무보정 P300 | 온라인 85.2% (오프라인 기준 87.8%) | 최신 "plug-and-play" 주장 |
| Behboodi, Kinney-Lang, Etemad, Kirton, Abou-Zeid | 2026 | arXiv | [arXiv:2601.06028](https://arxiv.org/abs/2601.06028) | Z / L(약 43 s) | 파운데이션 모델 기반 c-VEP 무보정·소량 보정 | 데이터셋 1: 무보정 68.8%, 약 11분 보정한 기존 방식 66.2%. 데이터셋 2: 무보정 71.8%, 보정 기준 93.7%. 약 43 s 보정 시 92% | 파운데이션 모델 + 소량 보정이라는 우리와 같은 방향 |

### A4. 자극 동기 피험자 간 정렬 (고전 BCI 밖이지만 SLA 신규성에 직접 위협)

| 인용 | 연도 | 학술지·학회 | 링크 | 보정 데이터 | 방법 | 핵심 수치 | 관련성 |
|---|---|---|---|---|---|---|---|
| Haxby, Guntupalli, Connolly, Halchenko, Conroy, Gobbini, Hanke, Ramadge (하이퍼정렬) | 2011 | Neuron 72(2):404–416 | [PubMed 22017997](https://pubmed.ncbi.nlm.nih.gov/22017997/) | S (영화 전편 시청) | 영화 시청 반응의 **시점별 대응**으로 각 피험자의 복셀 공간을 공통 공간에 **직교 프로크루스테스** 사상 | 공통 반응 조율 함수 35개 | SLA의 수학적 원형 |
| Chen PH, Chen J, Yeshurun, Hasson, Haxby, Ramadge (SRM) | 2015 | NeurIPS 28 | [proceedings](https://proceedings.neurips.cc/paper/2015/hash/b3967a0e938dc2a6340e258630febd5a-Abstract.html) | S | 공유 자극 시계열에서 피험자별 직교 열 사상과 공유 응답을 함께 추정. 새 피험자는 공유 응답에 사상 | — | "새 사용자 → 공유(참조) 응답" 방향이 SLA와 같음 |
| Shen X, Liu X, Hu X, Zhang D, Song S (CLISA) | 2022 | IEEE Trans. Affective Computing | [arXiv:2109.09559](https://arxiv.org/abs/2109.09559) | 학습 시 S, 시험 시 불필요 | 같은 영화 구간을 본 피험자 간 표현을 대조학습으로 정렬 | 피험자와 클립을 모두 처음 보는 9클래스 과제에서 45.7% (차선 34.3%). SEED도 사용 | SEED에서 "같은 자극 = 피험자 간 정렬 신호"를 쓴 선례(학습 단계) |
| Meng R (GRN) | 2026 | ICANN 2026 | [arXiv:2603.11119](https://arxiv.org/abs/2603.11119) | 시험 시 자극 시간축 필요(HTML 본문 요약 기준) | 개인 인코더 + 학습 프로토타입 + 학습 피험자 3명(K_r=3)과의 **같은 자극 시간축** PLV/coherence 동기화 분기 | SEED LOSO 87.90±3.85%. 자극을 어긋나게 맞추거나 라벨을 섞으면 성능 붕괴 | SEED에서 "자극 동기 교차 피험자 정렬"을 명시적으로 사용. 반드시 인용하고 차별화 |

### A5. 추가로 존재를 확인한 문헌 (간략)

**정렬·적응**
- Shenoy, Krauledat, Blankertz, Rao, Müller 2006, J Neural Eng 3(1):R13 ([doi](https://doi.org/10.1088/1741-2560/3/1/R02)): 보정 세션과 피드백 세션 사이의 이동을 보이고, 단순 적응만으로 유의하게 개선.
- Bleuzé, Mattout, Congedo 2022, Front Hum Neurosci 16:1049985: 접공간 정렬(TSA). 데이터베이스 18개·349명에서 RPA 대비 +2.7%.
- Kobler, Hirayama, Zhao, Kawanabe 2022, NeurIPS ([arXiv:2206.01323](https://arxiv.org/abs/2206.01323)): SPD 도메인별 모멘텀 배치정규화.
- Wimpff, Döbler, Yang 2024, IEEE BCI Winter Conf. ([arXiv:2311.18520](https://arxiv.org/abs/2311.18520)): 공분산 정렬 + 배치정규화 통계 교체 + 엔트로피 최소화로 온라인 시험 시점 적응.
- Gnassounou, Collas, Flamary, Lounici, Gramfort 2024 ([arXiv:2407.14303](https://arxiv.org/abs/2407.14303)): 최적수송 기반 시공간 몽주 정렬. 시험 시점 적응.
- Collas, Flamary, Gramfort 2024 ([arXiv:2402.03345](https://arxiv.org/abs/2402.03345)): 슈티펠 행렬 기반 약지도 공분산 정렬(MEG).
- Duan et al. 2025 ([arXiv:2509.19403](https://arxiv.org/abs/2509.19403)): 온라인 EA → 배치정규화 갱신 → 자기지도. 시행 1개씩 갱신해 SSVEP +4.9%, MI +3.6%.
- Ahmed et al. 2026, Front Syst Neurosci ([doi](https://doi.org/10.3389/fnsys.2026.1840121)): RPA·EA·CORAL 비교. EA가 무정렬 대비 +3.44%.
- Heskebeck, Bernhardsson, Bergeling 2026, Front Hum Neurosci ([doi](https://doi.org/10.3389/fnhum.2026.1824613)): RPA 회전의 기하학적 한계와 원천 선택 지표.
- Lopes et al. 2026 ([arXiv:2606.16462](https://arxiv.org/abs/2606.16462)): "EA는 피험자 공분산 재중심화로 공유 인코더를 돕는다." 피험자별 인코더를 쓰면 EA 의존이 줄어듦.
- Rodrigues GH et al. 2024, EUSIPCO ([arXiv:2405.14994](https://arxiv.org/abs/2405.14994)): EA와 증강의 시너지. 미세조정 공유 모델에서 +8.41%.

**SSVEP**
- Nakanishi, Wang YT, Wei, Chiang, Jung 2020, IEEE TBME 67(4):1105–1113 ([doi](https://doi.org/10.1109/TBME.2019.2929745)): 장치 간 공유 잠재 응답(SLR) 전이. 같은 사람 내 전이.
- Wong et al. 2020, J Neural Eng 17(1):016026: 다중 자극 학습(ms-eCCA/ms-eTRCA)으로 보정 부족 완화.
- Wong et al. 2021, IEEE TASE 18(2):552–563 ([doi](https://doi.org/10.1109/TASE.2021.3054741)): 자극 주파수 간 개인 지식 전이 (세부 미확인).
- Zhang Y, Xie SQ, Shi, Li J, Zhang ZQ 2023, IEEE TNSRE 31:1574–1583: 전이 템플릿과 최소제곱 전이 공간필터, 원천 기여 점수.
- Huang J et al. 2023, IEEE TNSRE 31:3307–3319: 도메인 일반화로 대상 데이터 없이 공간필터·템플릿 전이.
- He X et al. 2024, IEEE TBME 71(11):3071–3084: 전이 중첩 이론. 무보정이며 FBCCA·tt-CCA·CSSFT를 상회.
- Flores et al. 2024, EMBC: ELM 오토인코더 비선형 변환 84.23% vs LST 82.19% (템플릿 1개, 35명).
- Wang Z et al. 2026, IEEE JBHI 30(1):328–338 ([arXiv:2506.10933](https://arxiv.org/abs/2506.10933)): 인스턴스 기반 TRCA + 유사도 기반 원천 피험자 선택.
- Lin H et al. 2026, Sensors 26(18):5830 (SQ-HAF): 자극당 1회 보정, 공분산 정렬, 원천 품질 순위. Benchmark 1 s에서 81.57%, ITR 227.83 bpm.
- Hu K et al. 2025, Biomed Phys Eng Express: SSVEP 무보정에 EA와 원천 선택.
- Li H et al. 2024, ESWA 249:123492와 Li H et al. 2025, ESWA 276:127208: 교차 피험자 지식 전이, 극소 보정 (세부 미확인).
- Pan Y et al. 2024, Cogn Neurodyn 18(5):2925–2945: GAN 기반 짧은 SSVEP 신호 확장.

**MI/ERP/c-VEP**
- Kwon, Lee, Guan, Lee 2020, IEEE TNNLS 31(10):3839–3852: 54명 대규모 MI 데이터로 무보정 CNN. 피험자 의존 CSP/FBCSP를 상회.
- Fahimi et al. 2021, IEEE TNNLS 32(9):4039–4051: DCGAN 증강으로 +7.32% / +5.45%.
- Kostas & Rudzicz 2020, J Neural Eng 17(5):056008: 피험자 불변 딥넷(TIDNet).
- Kindermans et al. 2014, PLoS One 9(7):e102504: 완전 무보정 온라인. 30시행 뒤 지도 학습과 대등.
- Verhoeven et al. 2017, J Neural Eng 14(3):036021: 무보정 ERP 추정기 혼합.
- Hübner et al. 2018, IEEE CIM 13(2):66–77: ERP 비지도 학습 리뷰와 온라인 비교.
- Barachant & Congedo 2014 ([arXiv:1409.0107](https://arxiv.org/abs/1409.0107)): 리만 기하 기반 plug&play P300.
- Thielen et al. 2021, J Neural Eng 18(5):056007, 그리고 Thielen, Sosulski, Tangermann 2024, Graz BCI ([arXiv:2403.15521](https://arxiv.org/abs/2403.15521)): c-VEP 무보정(비지도 평균 최대화 UMM, CCA).

**리뷰·파운데이션 모델**
- Jayaram et al. 2016, IEEE CIM 11(1):20–31.
- Huang X et al. 2021, Front Neurosci 15:733546: 보정 시간 단축 신호처리 리뷰.
- Li T et al. 2026 ([arXiv:2604.27033](https://arxiv.org/abs/2604.27033)): 교차 피험자 일반화 딥러닝 서베이.
- Liu D ... Wu D 2026, EEG-FM-Compass, National Science Review ([arXiv:2601.17883](https://arxiv.org/abs/2601.17883)): 파운데이션 모델 12개를 데이터셋 13개에서 비교. 선형 탐침만으로는 부족하고, 전문 모델도 경쟁적.

**타 영역의 자극·조건 동기 정렬**
- Heo et al. 2026 ([arXiv:2607.19394](https://arxiv.org/abs/2607.19394)): ECoG 말 이해 데이터에서 SRM으로 새 피험자를 사상하고, 디코더는 재학습하지 않음.
- Shen et al. 2024, CL-SSTER ([arXiv:2402.14213](https://arxiv.org/abs/2402.14213)): 같은 자극을 본 피험자 간 EEG 표현을 대조학습(감정 영상 포함).
- 침습 BCI 잠재공간 정렬: Degenhart et al. 2020, Nat Biomed Eng 4:672–685 / Gallego et al. 2020, Nat Neurosci 23:260–270 / Safaie et al. 2023, Nature 623:765–771. 존재만 확인했고, 정렬 방식(조건·시간 대응 CCA 등)은 원문 미확인입니다.

---

## (B) 이 분야의 경향 (6개)

1. **"재중심화가 이득의 대부분"이라는 합의가 굳어졌습니다.**
   - 흐름: Vidaurre 2011(특징 평균) → Zanini 2018(리만 재중심화) → EA 2020 → SPD 배치정규화 2022 → Mellot 2023(재중심화가 가장 폭넓게 유효) → Wu 2025, Junqueira 2024, Ahmed 2026(EA 표준화·재평가) → Lopes 2026(EA의 효과는 곧 재중심화).
   - 우리 발견 (1)·(2)는 이 합의를 파운데이션 모델 임베딩에서 재확인한 것입니다.
2. **회전은 "대응"이 있어야만 추정됩니다.** 무라벨 회전 추정은 어렵기 때문에, 분야 전체가 다음 셋 중 하나를 씁니다.
   - 라벨: RPA의 클래스 평균, LA(클래스당 1개 이상)
   - 짝지은 데이터: Mellot의 같은 피험자 다른 과제, ALPHA의 같은 사용자 다른 장치
   - 공유 자극의 시점 대응: 하이퍼정렬, SRM
   - SLA는 세 번째 유형을 감정 EEG 보정에 옮긴 것입니다.
3. **SSVEP는 "템플릿 복사"에서 "새 사용자 맞춤 변환 학습"으로, 그리고 "일부 자극만 보정"으로 이동했습니다.**
   - tt-CCA(2015, 변환 없음) → LST(비제약 선형, 자극별) → sd-LST(자극 공통 단일 행렬) → SSVEP-DAN(비선형, 자극 독립 학습) → OS-SSVEP(클래스당 1회).
   - 보정량은 stCCA 9시행, sd-LST 약 10시행, msSAME 24 s(40타깃), c-VEP 1분 미만까지 줄었습니다.
   - 동시에 **음의 전이와 원천 피험자 선택**이 핵심 쟁점이 되었습니다(SSVEP-DAN이 관찰한 LST 음의 전이, Zhang 2023 기여 점수, Wang 2026 SS-iTRCA, SQ-HAF 원천 순위).
4. **무라벨 온라인·시험 시점 적응과 "plug-and-play" 주장이 늘었습니다.**
   - 예: OACCA 2022, T-TIME 2024, Wimpff 2024, Duan 2025, SPDIM 2025, ERP의 Kindermans 2014와 Kim & Kim 2025, c-VEP의 Thielen 2021/2024.
   - 대부분 시험 시행 흐름을 인과적으로 사용합니다. 시험 데이터를 아예 쓰지 않는 우리 프로토콜은 더 엄격하므로, 그 점을 명시하면 장점이 됩니다.
   - 라벨 분포 이동은 재중심화의 알려진 실패 원인입니다(SPDIM). 클래스 균형 보정(감정당 1클립)은 그에 대한 방어로 서술할 수 있습니다.
5. **생성·증강 기반 보정이 늘었습니다.** SAME과 msSAME(2023), GAN(Fahimi 2021, Pan 2024), EA+증강(2024). 감정 BCI에서는 아직 드뭅니다.
6. **파운데이션 모델 시대의 보정**
   - c-VEP 파운데이션 모델 무보정(Behboodi 2026)은 소량 보정(약 43 s)을 더해야 완전 보정 수준에 근접합니다.
   - EEG-FM-Compass(2026)는 파운데이션 모델이 일관되게 우월하지 않고 선형 탐침만으로는 부족하다고 보고합니다.
   - 결론: 파운데이션 모델 위에서도 정렬·보정 설계는 여전히 필요합니다. 우리 연구가 정확히 이 지점에 있습니다.

---

## (C) 신규성 위협과 정직한 범위 설정

### C-1. SLA와 가장 가까운 방법 비교

| 방법 | 정렬 대상 | 방향 | 사상 형태 | 대응 단위 | 새 사용자 데이터 | 분류기 |
|---|---|---|---|---|---|---|
| tt-CCA 2015 | 원신호 템플릿 | 원천 평균 템플릿을 그대로 사용 | 없음 | 자극(=클래스) | 없음(+온라인 갱신) | CCA |
| stCCA 2020 | 공간필터 적용 템플릿 | 원천 → 대상 | 피험자 가중합(최소제곱) | 자극(일부만) | 9시행 | 템플릿 매칭 |
| LST 2021 | 원천 단일 시행(원신호) | 원천 → 대상 템플릿 | **비제약** 채널 선형. 자극·대역·시행별 | 표본 단위 시간(위상 고정) | 자극당 2–5회 × 전체 자극 | 새 사용자 TRCA 재학습 |
| sd-LST 2023 | 원천 원신호 | 원천 → 대상 | 자극 공통 단일 선형 | 자극(일부만) | 약 10시행 / 40클래스 | TRCA |
| SSVEP-DAN 2024 | 원천 원신호 | 원천 → 대상 템플릿 | 비선형 신경망 | 자극·시간 | 자극당 2–4회 | TRCA |
| OS-SSVEP 2024 | 원천과 대상 모두 | 양쪽 → 사인-코사인 기준 영역 | 다중 기준 LST | 자극 | 자극당 1회 | 융합 |
| ALPHA 2022 | 공간패턴 + 공분산 | 원천(같은 사용자) → 대상 | **직교 프로크루스테스** + CORAL | 자극 | 새 장치 재보정 없음(같은 사용자 과거 데이터) | 부분공간 풀링 |
| RPA 2019 | SPD 공분산 | 원천 ↔ 대상 | 평행이동·스케일·**직교 회전** | 클래스 평균 | 회전에 라벨 소수 | MDM 등 |
| 하이퍼정렬 2011 / SRM 2015 | fMRI 반응 | 개인 → 공통 공간 | **직교** | **영화 시점** | 공유 영화 시청 데이터 | 공통 공간 분류기 |
| GRN 2026 | PLV/coherence 동기화 특징 | 시험 피험자 vs 참조 3명 | 사상 없음(특징) | **같은 자극 시간축** | **시험 구간의 자극 시간축** | 신경망 |
| **SLA (우리)** | FM 클래스 토큰 임베딩의 상위 k 부분공간 | **대상 → 원천 평균** | 직교 프로크루스테스 | 클립 × 초(4 s 창) | 감정당 클립 1개(20 s / 40 s / 전체), 보정 구간만 사용 | **고정된** 원천 분류기 |

### C-2. 위협 목록과 권장 표현

1. **도메인 중심화**
   - 위협: Vidaurre 2011, Zanini 2018, EA 2020, Kobler 2022, Mellot 2023.
   - 권장 표현: "잘 알려진 재중심화 원리를 FM 임베딩 공간에서 재확인하고, 보정 구간만 쓰는 현실적 프로토콜에서 그 기여(+0.07–0.10)를 정량화했다. 입력 수준 EA·배치정규화 계열은 그 위에 추가 이득이 거의 없다."
   - 이 음의 결과는 Lopes 2026("EA의 이득은 재중심화")과 일관된다는 근거로 뒷받침하십시오.
2. **"남는 오차는 회전"**
   - 위협: RPA(2019)가 이미 피험자 간 이동을 "평행이동 + 스케일 + 회전"으로 모델링했고, Mellot 2023·TSA 2022·Heskebeck 2026도 회전을 다룹니다.
   - 권장 표현: "RPA 계열의 이동 모델이 FM 임베딩에서도 성립함을 보였다."
   - 우리 기여는 "어떤 대응(라벨, 자극 시간)으로 클래스당 클립 1개 이하에서 회전을 추정할 수 있는가"를 비교한 데 있다고 쓰십시오.
3. **SLA의 핵심 아이디어**
   - 위협: SSVEP의 LST 계열(2019–2026), stCCA, tt-CCA, ALPHA(직교 프로크루스테스), fMRI 하이퍼정렬·SRM(시점 대응 + 직교 사상), 감정 EEG의 CLISA·CL-SSTER(학습 시 공유 자극), **GRN 2026(SEED, 추론 시 자극 동기 교차 피험자)**.
   - 직교 제약과 상위 k 부분공간도 새롭지 않습니다(하이퍼정렬, SRM, ALPHA의 부분공간 정렬).
   - 방어 가능한 범위:
     - (a) 위상 고정되지 않은 자연주의 감정 영화 자극에서 4 s 창 단위의 시간 대응만으로 회전을 추정할 수 있음을 보인 것
     - (b) 고정된 FM 임베딩과 고정 분류기 상태에서 새 사용자를 원천 쪽으로 보내는 보정
     - (c) 시험 데이터를 전혀 쓰지 않고, 시험 구간의 자극 정보도 필요 없음. 이 점이 GRN과의 결정적 차이입니다. GRN은 HTML 본문 요약 기준으로 모든 시험 구간의 자극 시간축을 알아야 하고, SEED에서는 클립 정체가 라벨과 일대일입니다.
     - (d) 효과의 이질성: SEED-V +0.061, SEED 약 0
   - 권장 문장: "We adapt the stimulus-locked inter-subject transformation principle established in SSVEP template transfer (tt-CCA, LST, stCCA, sd-LST, SSVEP-DAN) and fMRI hyperalignment/SRM to naturalistic affective stimuli in a frozen EEG foundation-model embedding, under a strictly calibration-only protocol."
4. **"감정 라벨 없이"**
   - 위협: SEED 계열에서 클립 정체 = 클래스입니다. SSVEP 문헌 기준으로 자극당 보정 시행은 곧 라벨 있는 보정입니다(LST, stCCA 모두 그렇게 분류).
   - 권장 표현: "클래스 수준 통계(클래스 평균, 라벨 손실)는 쓰지 않고, 보정 프로토콜이 제공하는 자극-시간 대응만 사용한다. 단, 보정 클립의 정체는 클래스를 내포한다."
5. **보정량 관련 수식어**
   - SSVEP는 40클래스를 약 9–24 s, c-VEP는 1분 미만으로 보정합니다. 우리는 20 s × 감정 3–5개 = 60–100 s 이상입니다.
   - "minimal" 같은 최상급 표현은 피하고 "클래스당 짧은 클립 1개, 시험 데이터 미사용"으로 쓰십시오. 감정 반응이 느리다는 점을 근거로 덧붙이면 됩니다.
6. **시험 시점 적응과의 관계**
   - T-TIME, OTTA, OACCA는 인과적 온라인 적응이므로 배포 가능합니다.
   - "비전이적"이 오프라인 전이적 방법(EA를 시험 데이터에 적용 등)만이 아니라 온라인 시험 적응까지 배제한다는 점을 명시하십시오. 비교가 가능하다면 T-TIME류를 보조 기준선으로 두는 것이 안전합니다.

### C-3. 리뷰어가 요구할 대조실험과 기준선 (권장)

- **시간 동기화 기여를 분리하는 대조실험.** GRN도 비슷한 대조를 했습니다.
  - ① 같은 클립 안에서 시간을 섞어 짝짓기: 클립 정체(=라벨)는 유지하고 시간 동기만 파괴. 이득이 남으면 사실상 RPA/LA식 클래스 평균 회전입니다.
  - ② 같은 감정의 다른 클립과 짝짓기: 자극 특이성 대 클래스 수준 정보.
  - ③ 무작위 짝짓기: 귀무 기준.
- **사상 형태 소거 실험.** 같은 대응을 쓰되 다음을 비교합니다.
  - 비제약 최소제곱(LST식)
  - 직교 프로크루스테스(현재 SLA)
  - 클래스 평균 회전(RPA·LA식, 발견 (5))
  - 공유 응답 사상(SRM식)
- **참조 구성.** 학습 피험자 단순 평균 대신 다음을 시도해 볼 만합니다. SEED에서 이득이 없는 원인이 원천 이질성 때문인지 검증할 수 있습니다.
  - stCCA식으로 보정 클립에서 최소제곱 가중치를 적합한 피험자 가중 참조
  - SS-iTRCA·SQ-HAF식 원천 피험자 선택
- **정당화 분석.** 클립 × 초 단위로 FM 임베딩의 피험자 간 상관(inter-subject correlation)을 보고하십시오. SEED-V에서 자극 동기 공유 분산이 실제로 있고 SEED에서는 약하다는 것을 보이면, 효과의 이질성이 설명됩니다.

---

## (D) 반드시 인용할 논문

**1순위 (반드시)**
1. Lotte F. Signal processing approaches to minimize or suppress calibration time in oscillatory activity-based brain–computer interfaces. Proc. IEEE 103(6):871–890, 2015.
2. Wu D, Xu Y, Lu BL. Transfer learning for EEG-based brain–computer interfaces: a review of progress made since 2016. IEEE TCDS 14(1):4–19, 2022.
3. He H, Wu D. Transfer learning for brain-computer interfaces: a Euclidean space data alignment approach. IEEE TBME 67(2):399–410, 2020.
4. Zanini P, Congedo M, Jutten C, Said S, Berthoumieu Y. Transfer learning: a Riemannian geometry framework with applications to brain–computer interfaces. IEEE TBME 65(5):1107–1116, 2018.
5. Rodrigues PLC, Jutten C, Congedo M. Riemannian Procrustes analysis: transfer learning for brain–computer interfaces. IEEE TBME 66(8):2390–2401, 2019.
6. Vidaurre C, Kawanabe M, von Bünau P, Blankertz B, Müller KR. Toward unsupervised adaptation of LDA for brain–computer interfaces. IEEE TBME 58(3):587–597, 2011.
7. Mellot A, Collas A, Rodrigues PLC, Engemann D, Gramfort A. Harmonizing and aligning M/EEG datasets with covariance-based techniques to enhance predictive regression modeling. Imaging Neuroscience 1, 2023.
8. He H, Wu D. Different set domain adaptation for brain-computer interfaces: a label alignment approach. IEEE TNSRE 28(5):1091–1108, 2020.
9. Yuan P, Chen X, Wang Y, Gao X, Gao S. Enhancing performances of SSVEP-based brain–computer interfaces via exploiting inter-subject information. J Neural Eng 12(4):046006, 2015.
10. Chiang KJ, Wei CS, Nakanishi M, Jung TP. Boosting template-based SSVEP decoding by cross-domain transfer learning. J Neural Eng 18(1):016002, 2021. 예비 연구인 Chiang et al., NER 2019(arXiv:1810.02842)도 함께 인용.
11. Wong CM et al. Inter- and intra-subject transfer reduces calibration effort for high-speed SSVEP-based BCIs. IEEE TNSRE 28(10):2123–2135, 2020.
12. Bian R, Wu H, Liu B, Wu D. Small data least-squares transformation (sd-LST) for fast calibration of SSVEP-based BCIs. IEEE TNSRE 31:446–455, 2023.
13. Liu B, Chen X, Li X, Wang Y, Gao X, Gao S. Align and pool for EEG headset domain adaptation (ALPHA) to facilitate dry electrode based SSVEP-BCI. IEEE TBME 69(2):795–806, 2022.
14. Chen SY, Chang CM, Chiang KJ, Wei CS. SSVEP-DAN: cross-domain data alignment for SSVEP-based brain–computer interfaces. IEEE TNSRE 32:2027–2037, 2024.
15. Haxby JV et al. A common, high-dimensional model of the representational space in human ventral temporal cortex. Neuron 72(2):404–416, 2011.
16. Chen PH, Chen J, Yeshurun Y, Hasson U, Haxby J, Ramadge PJ. A reduced-dimension fMRI shared response model. NeurIPS 28, 2015.
17. Shen X, Liu X, Hu X, Zhang D, Song S. Contrastive learning of subject-invariant EEG representations for cross-subject emotion recognition. IEEE Trans. Affective Computing, 2022 (arXiv:2109.09559).
18. Meng R. Group Resonance Network: learnable prototypes and multi-subject resonance for EEG emotion recognition. ICANN 2026 (arXiv:2603.11119).

**2순위 (권장)**
- EA 재평가: Junqueira et al. 2024 (J Neural Eng), Wu D 2025 (J Neural Eng 22:031005)
- 시험 시점 적응: Li S et al. 2025 SPDIM (ICLR), Li S et al. 2024 T-TIME (IEEE TBME), Wimpff et al. 2024 (IEEE BCI Winter Conf.), Kobler et al. 2022 (NeurIPS)
- SSVEP 최신: Deng et al. 2024 OS-SSVEP (Neural Networks), Li D et al. 2024 ms-LST-OA (IEEE TNSRE)
- 세션 간 전이·무보정: Krauledat et al. 2008 (PLoS One), Kindermans et al. 2014 (J Neural Eng)
- 짧은 보정·파운데이션 모델: Miao et al. 2024 (c-VEP, 1분 미만), Behboodi et al. 2026 (파운데이션 모델 c-VEP)
- 회전 정렬: Bleuzé et al. 2022 (TSA), Heskebeck et al. 2026 (회전의 한계)

---

## 미검증 / 주의 사항 (UNVERIFIED)

- **UNVERIFIED:** Dmochowski et al. 2012 (Front Hum Neurosci, EEG 상관 성분 분석, 영화 시청). 이번 세션에서 확인하지 못했습니다.
- **UNVERIFIED:** "Through their eyes: multi-subject brain decoding with simple alignment techniques" (Imaging Neuroscience 2024, fMRI). 제목과 URL만 확인했고 저자는 확인하지 못했습니다.
- 수치·세부의 출처 주의:
  - LST의 정확도 수치는 원문 그림에서 읽은 근사값입니다.
  - ALPHA의 세부(직교 프로크루스테스와 CORAL)는 후속 논문(Liu X et al. 2022, Front Neurosci 16:863359)의 설명으로 확인했습니다.
  - RPA 회전에 대상 라벨이 필요하다는 점은 pyRiemann의 TLRotate 문서("대상 도메인의 클래스 평균에 맞춤")로 확인했습니다.
  - SPDIM이 라벨 이동 하 재중심화의 한계를 다룬다는 점은 초록 수준에서만 확인했습니다.
  - Gallego 2020 / Safaie 2023의 정렬 방식, Wong 2021 TASE와 Li H 2024/2025 ESWA의 방법 세부는 원문을 확인하지 못했습니다.
- 링크 형식: Vidaurre 2011, Haxby 2011, Wu 2022는 DOI를 직접 검증하지 못해 ResearchGate, PubMed, arXiv 링크로 대신했습니다.
