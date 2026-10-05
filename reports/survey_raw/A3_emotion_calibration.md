# 새 피험자 보정 데이터를 쓰는 교차 피험자·교차 세션 EEG 감정 인식 연구 조사 (2018–2026)

**검증 방법**: 아래 표의 논문은 모두 검색 결과나 원문으로 제목·저자·연도·학회/저널(또는 arXiv 번호)을 확인했습니다. 원문은 arXiv, 출판사 페이지, Europe PMC, Crossref, PDF 본문을 봤습니다. 수치나 DOI를 확인하지 못한 경우 "미확인"이라고 적었습니다. 조사 후반에 세션 공용 웹 검색 한도(200회)를 다 써서, 그 뒤로는 DOI·arXiv 단건 조회와 저장된 PDF의 참고문헌으로만 검증했습니다. 그래서 2026년 하반기에 나온 감정 보정 논문은 빠졌을 수 있습니다.

**약어**
- 평가·적응: 피험자 1명 제외 교차검증(LOSO), 비지도 도메인 적응(UDA), 도메인 일반화(DG), 테스트 시간 적응(TTA), 소스 프리 도메인 적응(SFDA)
- 특징·통계: 미분 엔트로피(DE), 최대 평균 불일치(MMD), 선형 동적 시스템 평활(LDS)
- 정렬: 유클리드 정렬(EA), 리만 프로크루스테스 분석(RPA), 스타일 전이 매핑(STM), 공유 응답 모델(SRM)
- 공유 자극: 피험자 간 상관(ISC), 상관 성분 분석(CorrCA), 반복 자극 혼입(RSC)
- 기타: 파운데이션 모델(FM), 우리의 자극 동기 정렬(SLA)

**"새 피험자 데이터" 열의 분류**
- 없음
- 전이적: 테스트와 같은 비라벨 데이터를 학습이나 적응에 사용
- 온라인 전이적: 테스트 스트림을 순차적으로 사용
- 비라벨 보정: 테스트와 분리된 보정 블록
- 라벨 보정: 퓨샷

## (A) 논문 표

### A1. 새 피험자 보정·퓨샷 개인화

| 논문 | 연도 | 출처 | 링크 | 새 피험자 데이터 | 방법(한 줄) | 데이터셋·결과 | 우리 연구와의 관련성 |
|---|---|---|---|---|---|---|---|
| Zhao, Yan, Lu, "Plug-and-Play Domain Adaptation for Cross-Subject EEG-based Emotion Recognition" (PPDA) | 2021 | AAAI 35(1):863–870 | https://doi.org/10.1609/aaai.v35i1.16169 | **비라벨 보정**. 세션 맨 앞 T초를 쓰고 T=45 s(15–95 s 비교). 감정 태그는 버림. 테스트에서 이 45 s를 뺐는지는 명시 안 됨 | 공유·개인 인코더(LSTM 오토인코더)를 학습함. 보정 블록으로는 새 피험자의 개인 인코더만 학습함. 개인 성분 유사도로 소스 피험자별 분류기를 가중 앙상블하고 공유 분류기와 융합함 | SEED 세션 1, LOSO: 86.7±7.1%. 보정을 빼면 85.4%, DResNet 85.3%, 전이적 WGAN-DA 87.1%. 15 s 이후 곡선이 평탄함 | **프로토콜상 가장 가까운 선행 연구**. SEED 고정 순서에서 첫 45 s는 첫 클립(긍정) 하나뿐이므로 단일 감정 보정으로 +1.3%p를 얻은 셈. 우리의 "감정 포괄이 길이보다 중요" 결과를 간접 지지함(우리 추론) |
| Li, Qiu, Shen, Liu, He, "Multisource Transfer Learning for Cross-Subject EEG Emotion Recognition" | 2020 | IEEE TCYB 50:3281–3293 | https://doi.org/10.1109/TCYB.2019.2904052 | **라벨 보정**. 보정 세션의 소량 라벨 데이터로 적응하고 이후 세션에서 테스트(비전이적) | 보정 데이터로 소스 피험자를 선택함. 소스마다 STM(아핀 변환)을 적용하고 소스 모델을 통합함 | SEED 3클래스: 비전이 대비 +12.72%p | (iii) 클래스 구조 정합에 가장 가까운 감정 분야 선행 연구. 세션 분리 보정이라는 점도 같음 |
| Lin & Jung, "Improving EEG-Based Emotion Classification Using Conditional Transfer Learning" | 2017 | Front. Hum. Neurosci. 11:334 | https://doi.org/10.3389/fnhum.2017.00334 | **라벨 보정**. 본인의 소량 라벨에 저장소의 타인 데이터를 더함 | 개인별 전이 가능성을 판정한 뒤 특징 공간이 비슷한 타인 데이터만 전이함(조건부 전이학습) | 26명: 본인 데이터만 쓸 때보다 valence 약 +15%, arousal 약 +12% | 보정 데이터 비용과 부정 전이 문제를 처음 제기한 연구 중 하나 |
| Lin, "Constructing a Personalized Cross-Day EEG-Based Emotion-Classification Model Using Transfer Learning" | 2020 | IEEE JBHI 24(5):1255–1264 | https://doi.org/10.1109/JBHI.2019.2934172 | **라벨 보정**. 첫날 세션 하나만 쓰고 5일차에 테스트 | 강건 주성분 분석(RPCA)을 넣은 전이학습에 유사 소스 세션을 추가함 | 12명, 5일 데이터: 소스 3세션 추가 시 valence +11.19%p, arousal +5.82%p. RPCA가 없으면 날짜 간 이득이 없음 | 교차 일(세션) 보정의 선례 |
| Bhosale, Chakraborty, Kopparapu, "Calibration free meta learning based approach for subject independent EEG emotion recognition" | 2022 | Biomed. Signal Process. Control | https://www.sciencedirect.com/science/article/abs/pii/S1746809421008867 | **라벨 퓨샷**(5–25샷). 미세조정 없음. 보정 없는 설정도 평가 | 메타러닝 퓨샷과 3D 합성곱-순환 임베딩 | 퓨샷 67.24–78.12%, 보정 없음 62.98–71.68%. 데이터셋 세부는 미확인 | 메타러닝 보정의 기준선 |
| Ning, Chen, Zhang, "Cross-subject EEG emotion recognition using domain adaptive few-shot learning networks" (SDA-FSL) | 2021 | IEEE BIBM 2021, 1468–1472 | DOI 미확인 | **라벨 지원 세트**와 도메인 적응을 함께 씀 | CBAM 특징 매핑, 도메인 적응, 인스턴스 어텐션 프로토타입 네트워크 | 수치 미추출 | 감정 EEG에서 프로토타입 퓨샷을 쓴 초기 사례 |
| Jin, Zhang, Zhao, Du, Li, "EvoFA: Evolvable Fast Adaptation for EEG Emotion Recognition" | 2024 | arXiv 2409.15733 | https://arxiv.org/abs/2409.15733 | **라벨 퓨샷**. 세션 1에서 클래스당 1샷 또는 5샷을 뽑고, 이후 세션으로 검증·테스트(온라인) | 메타러닝에 진화형 메타 적응 모듈을 더함. 테스트 중 소스 대비 분포 변화를 따라감 | SEED 피험자 내 교차 세션: 1샷 84.27%(ProtoNet 84.07), 5샷 88.15%(87.88). SEED-V도 평가 | **세션을 분리한 퓨샷 설계라 우리와 가장 가까운 퓨샷 연구**. 프로토타입 기준선으로 쓸 만함 |
| Liu, Chen, Zhang, "FACE: Few-shot Adapter with Cross-view Fusion for Cross-subject EEG Emotion Recognition" | 2025 | arXiv 2503.18998 | https://arxiv.org/abs/2503.18998 | **라벨 퓨샷**. 타깃 샘플을 섞은 뒤 클래스당 K∈{1,3,5,10}개를 무작위로 뽑고 나머지로 테스트. 같은 시행의 인접 1초 창이 양쪽에 섞임 | 메타러닝 퓨샷 어댑터와 교차 뷰 융합 | SEED 3/5/10샷 91.66/93.96/96.72. SEED-IV 83.91/89.51/95.95. SEED-V 89.55/95.20/98.95 | **시행 내 무작위 분할의 전형**. 우리 수치가 왜 낮아 보이는지 설명할 때 인용함 |
| Lu, Liu, Ma, Tan, Xia, "Hybrid transfer learning strategy for cross-subject EEG emotion recognition" (DFF-Net) | 2023 | Front. Hum. Neurosci. 17 | https://doi.org/10.3389/fnhum.2023.1280241 | **전이적**(비라벨 타깃으로 DA 사전학습)이면서 타깃 샘플 0.1%를 라벨로 미세조정 | DA와 퓨샷 미세조정을 결합 | SEED 93.37±1.88, SEED-IV 82.32±5.38 | 전이적 방식과 퓨샷의 혼합 사례. 비교할 때 공정성에 주의 |
| Wang, Liu, Ruan, Wang, Wang, "Cross-subject EEG emotion classification based on few-label adversarial domain adaption" | 2021 | Expert Syst. Appl. 185:115581 | (ESWA, DOI 미확인) | **소량 라벨 타깃**. 비라벨 타깃을 함께 썼는지는 미확인 | 다중 피험자 학습 기반의 소량 라벨 적대적 DA | 수치 미추출 | 소량 라벨 DA 계열의 대표 |

### A2. 피험자 정렬·정규화와 전이적 여부

| 논문 | 연도 | 출처 | 링크 | 새 피험자 데이터 | 방법 | 데이터셋·결과 | 관련성 |
|---|---|---|---|---|---|---|---|
| Zheng & Lu, "Personalizing EEG-Based Affective Models with Transfer Learning" | 2016 | IJCAI 2016, 2732–2738 | https://www.ijcai.org/Proceedings/16/Papers/388.pdf | 전이적 | 전이 성분 분석(TCA), 커널 PCA, 전이 파라미터 학습(TPT)으로 개인화 | SEED TPT 75.17% (PR-PL 표에서 재인용) | "개인화 감정 모델"의 원조 |
| Lan, Sourina, Wang, Scherer, Müller-Putz, "Domain Adaptation Techniques for EEG-Based Emotion Recognition: A Comparative Study on Two Public Datasets" | 2019 | IEEE TCDS 11(1):85–94 | https://doi.org/10.1109/TCDS.2018.2826840 | 전이적 | SEED와 DEAP에서 DA 기법을 비교 | 수치 미추출 | 고전적 비교 기준 |
| Li, Jin, Zheng, Lu, "Cross-subject emotion recognition using deep adaptation networks" | 2018 | ICONIP 2018, 403–413 | DOI 미확인(PPDA 참고문헌으로 서지 확인) | 전이적 | MMD 기반 심층 적응 네트워크(DAN) | SEED 83.81±8.56 (PR-PL 표에서 재인용) | 심층 전이적 DA의 원조 |
| Ma, Li, Zheng, Lu, "Reducing the subject variability of EEG signals with adversarial domain generalization" (DResNet) | 2019 | ICONIP 2019, 30–42 | DOI 미확인(PPDA 참고문헌으로 서지 확인) | 없음(DG) | 도메인 잔차 네트워크 | SEED 85.3±8.0 | 보정 없는 기준선. PPDA가 보정 이득을 잴 때 비교한 대상 |
| Fdez, Guttenberg, Witkowski, Pasquali, 층화 정규화 | 2021 | Front. Neurosci. 15:626277 | https://doi.org/10.3389/fnins.2021.626277 | **전이적**. 참가자·세션 단위 정규화를 테스트 참가자 자신의 세션 데이터로 계산 | 층마다 참가자·세션별로 특징을 정규화 | SEED LOSO 3클래스 79.6%, 2클래스 91.6%. 배치 정규화보다 높음 | (i)과 같은 계열 메커니즘의 전이적 버전 |
| Chen, Jin, Li, Fan, Li, He, MS-MDA | 2021 | Front. Neurosci. 15:778488 | https://doi.org/10.3389/fnins.2021.778488 | **전이적**. 소스 분기마다 타깃과 MMD를 계산 | 다중 소스 주변 분포 적응 | SEED 교차 피험자 89.63±6.79, 교차 세션 88.56±7.80. SEED-IV 각각 59.34±5.48, 61.43±15.71 | 전이적 다중 소스 DA의 대표 |
| Chen, Sun, Li 외, "Personal-Zscore: Eliminating Individual Difference for EEG-Based Cross-Subject Emotion Recognition" | 2023 (온라인 2021) | IEEE TAFFC 14(3):2077–2088 | https://ieeexplore.ieee.org/document/9662246/ | 피험자별 z-점수. 테스트 피험자 통계의 출처는 미확인이며, 자기 데이터 전체로 계산했을 것으로 추정 | 개인차 정량 지표 4종과 개인 z-점수 | SEED 14명에서 정확도와 강건성 향상 | (i)의 감정 분야 정규화 선례 |
| Zhou 외, PR-PL | 2024 | IEEE TAFFC 15(2):657–670 | https://arxiv.org/abs/2202.06509 | **전이적**. 타깃 데이터로 도메인 적대 학습을 하고 적응 임계값으로 의사 라벨을 만듦(본문 확인) | 소스 클래스 프로토타입(클래스 평균)과 쌍별 학습 | arXiv v2 기준. SEED 교차 피험자 단일 세션 93.06±5.12, 교차 피험자·교차 세션 85.56±4.78. SEED-IV 74.92±7.92 | 프로토타입 계열의 대표. 평가 프로토콜 4종을 정의함 |
| Zhou 외, EEGMatch | 2023 (2024 개정) | arXiv 2304.06496 | https://arxiv.org/abs/2304.06496 | 전이적(타깃을 포함한 다중 도메인 적응) | EEG-Mixup에 프로토타입·인스턴스 쌍별 학습을 더함. 불완전 라벨 상황 | 기존 최고 대비 SEED +6.89%p, SEED-IV +1.44%p | 라벨 부족을 다루는 전이적 방법 |
| Li, Wu, Zhou, Tian, Zhang, Liang, MAT | 2025 | arXiv 2509.01135 | https://arxiv.org/abs/2509.01135 | 없음(DG) | 도메인·클래스를 분리한 이중 프로토타입과 쌍별 학습 | 기존 최고 대비 SEED +2.87, SEED-IV +3.84, SEED-V +2.05%p | 비전이적 프로토타입 비교 대상 |
| Zhou, McEvoy, Valderrama, "Local-Global Feature Fusion for Subject-Independent EEG Emotion Recognition" | 2026 | arXiv 2601.08094 | https://arxiv.org/abs/2601.08094 | **없음**. 정규화 통계는 학습 피험자로만 계산하고, 테스트에는 학습 통계의 평균을 씀 | 국소·전역 특징 융합 | SEED-VII 7클래스 LOSO 40.1% (기준 36.4%) | 테스트 통계를 뺀 정규화를 명시한 최근 사례. 우리 중심화의 "보정 없음" 대조군 |

### A3. 공유 자극(같은 영화·같은 시점) 활용

| 논문 | 연도 | 출처 | 링크 | 새 피험자 데이터 | 방법 | 데이터셋·결과 | 관련성 |
|---|---|---|---|---|---|---|---|
| Shen, Liu, Hu, Zhang, Song, CLISA | 2023 (온라인 2022) | IEEE TAFFC 14(3):2496–2511 | https://doi.org/10.1109/TAFFC.2022.3164516 (arXiv 2109.09559) | 라벨 없음, **온라인 전이적**. DE 정규화 통계를 학습 통계로 시작해 테스트 데이터로 지수 가중 갱신(감쇠 0.99). 시행 단위 LDS 평활도 씀 | 같은 클립·같은 시간 구간의 두 학습 피험자 EEG를 양성 쌍으로 대조 학습함. 인코더 출력에서 DE를 뽑아 MLP로 분류 | SEED LOSO 86.4±6.4%(같은 정규화를 쓴 DE+MLP는 79.9%). **미지 자극 시험 SEED 77.4±13.4%**. THU-EP(80명) 2클래스 71.9%가 미지 자극에서 63.4% | 자극 동기 쌍을 **학습에만** 쓰고 새 사용자 보정에는 쓰지 않음. 이 점이 SLA와의 차이. 미지 자극 시험은 우리 혼입 점검의 선례 |
| Shen, Tao, Chen, Song, Liu, Zhang, CL-SSTER | 2024 | NeuroImage 301:120890 | https://arxiv.org/abs/2402.14213 | 해당 없음(공유 표현 학습과 ISC 분석) | 같은 자극과 다른 자극을 대조해 개인 간 공유 시공간 표현을 학습 | 합성·음성 이해·감정 영상 데이터에서 기존보다 높은 ISC | CLISA 후속 |
| Xie, Zheng, Xiao, Lu, Liu, TA2CL | 2026 | arXiv 2605.22379 | https://arxiv.org/abs/2605.22379 | 없음(인코더 고정, 분류기만 학습). 미지 자극 시험 없음 | 같은 자극·같은 시간 구간 양성 쌍에 시간 비동기 국소 매칭(Async-InfoNCE)을 더함 | FACED 9클래스 64.5%, 2클래스 79.5%. SEED 86.4%(5겹). SEED-V 70.1%(LOSO) | 피험자 간 반응 시간이 어긋나는 문제를 제기함. SLA의 "같은 초" 쌍 가정을 점검할 근거 |
| Ding, Hu, Xia, Liu, Zhang, "Inter-Brain EEG Feature Extraction and Analysis for Continuous Implicit Emotion Tagging During Video Watching" | 2021 | IEEE TAFFC 12:92–102 | https://cg.cs.tsinghua.edu.cn/people/~Yongjin/Inter-Brain_EEG_Feature_Extraction_and_Analysis_for_Continuous_Implicit_Emotion_Tagging_During_Video_Watching.pdf | 새 시청자 라벨 없음. 같은 영상을 본 집단의 EEG를 씀 | 집단 뇌간(inter-brain) 특징으로 영상 감정을 연속 태깅 | CLISA 서술에 따르면 집단이 클수록 예측이 좋아짐 | 공유 자극을 추론 단계에서 쓴 감정 연구. 다만 개인 정렬은 아님 |
| Dmochowski, Sajda, Dias, Parra, "Correlated components of ongoing EEG point to emotionally laden attention" | 2012 | Front. Hum. Neurosci. | DOI 미확인 | 해당 없음 | CorrCA로 같은 영화 시청자 간의 상관 성분을 추출 | ISC가 "감정이 실린 주의"를 반영 | 공유 자극 정렬의 신경과학적 근거. CLISA의 CorrCA 기준선 |
| Hajlaoui, Chetouani, Essid, "EEG-based Inter-Subject Correlation Schemes in a Stimuli-Shared Framework: Interplay with Valence and Arousal" | 2018 | arXiv 1809.08273 | https://arxiv.org/abs/1809.08273 | 해당 없음 | MAHNOB-HCI와 DEAP에서 ISC 계산 방식을 비교 | ISC는 valence가 높을수록 낮고, arousal이 높을수록 높음 | 감정에 따라 자극 동기 성분의 세기가 다름. SLA 효과가 데이터셋마다 다른 이유를 해석할 때 참고 |

### A4. 감정 EEG 평가 함정

| 논문 | 연도 | 출처 | 링크 | 핵심 내용 | 함의 |
|---|---|---|---|---|---|
| Li, Johansen, Ahmed, Ilyevsky, Wilbur, Bharadwaj, Siskind, "The Perils and Pitfalls of Block Design for EEG Classification Experiments" | 2021 | IEEE TPAMI | https://doi.org/10.1109/TPAMI.2020.2973153 | 블록 설계에서는 같은 블록 안의 시간 상관 때문에 정확도가 부풀려짐(객체 범주 EEG). 반박 논문(Palazzo 외, arXiv 2012.03849)도 있음 | SEED 계열은 클립 하나가 몇 분짜리 단일 라벨 블록임. 창 단위 무작위 분할은 금물이고 클립 단위 평가가 필요함 |
| Kilgallen, Pearlmutter, Siskind, "The Repeated-Stimulus Confound in Electroencephalography" | 2025 | arXiv 2508.00531 (IEEE 투고) | https://arxiv.org/abs/2508.00531 | 같은 자극에 대한 반응이 학습과 테스트 양쪽에 있으면 RSC가 생김. 4.46–7.42%p 과대평가, 16편이 해당. **감정 데이터셋은 다루지 않음**(객체 범주 SUD, THINGS-EEG 계열) | 감정 특이 논문으로 인용하면 부정확함. 그래도 SEED는 3세션 모두 같은 15개 클립을 쓰고 모든 피험자가 같은 클립을 봄. 즉 교차 세션·교차 피험자판 RSC가 생김 |
| Lei, Wu, Yi, Mo, "Impact of Trial-wise and Test Data Leakage on EEG-Based Emotion Classification" | 2025 | 4DMR@IJCAI25, CEUR-WS Vol-4115 | https://ceur-ws.org/Vol-4115/paper7.pdf | DEAP에서 시행 내 분할 누출과 테스트셋 하이퍼파라미터 선택으로 정확도가 최소 +35.71%p(valence), +25.00%p(arousal) 부풀려짐 | FACE식 분할과 테스트 기반 조기 종료가 얼마나 위험한지 보여주는 정량 근거 |
| Suo, Wang, Li, "Checkpoint Selection and Evaluation in EEG Emotion Recognition" | 2026 | arXiv 2607.27655 (Neuroinformatics 투고) | https://arxiv.org/abs/2607.27655 | SEED/SEED-IV/SEED-V에서 후보 체크포인트를 5개에서 80개로 늘리면 선택 풀 정확도는 +6.24%p, 분리된 풀은 −1.24%p | 모델 선택에 테스트 피험자를 쓰지 않았음을 명시해야 함 |
| Kukhilava 외, "Evaluation in EEG Emotion Recognition: State-of-the-Art Review and Unified Framework" | 2025 | arXiv 2505.18175 | https://arxiv.org/abs/2505.18175 | 2018–2023년 논문 216편 분석: 피험자 의존 35.7%, 피험자 독립 45.2%, 분할 방식 불명확 19.1%. EEGain 프레임워크 제안 | 프로토콜 보고 표준 |
| Apicella 외, "Toward cross-subject and cross-session generalization in EEG-based emotion recognition: Systematic review, taxonomy, and methods" | 2024 | Neurocomputing 604:128354 | https://doi.org/10.1016/j.neucom.2024.128354 | 교차 피험자 73.4%, 교차 세션 17%, 교차 데이터셋 9%. 데이터셋은 SEED 45.8%, DEAP 27.5%. 소량 라벨 타깃을 쓴 연구는 소수(PPDA 등) | 교차 세션 연구와 보정 연구가 부족하다는 정량 근거 |

LibEER(Liu 외, arXiv 2410.09767, 2024)는 17개 모델과 6개 데이터셋으로 감정 인식 평가를 표준화한 벤치마크입니다.

### A5. (C) 판단에 필요한 교차 분야 선행 연구

| 논문 | 연도 | 출처 | 링크 | 핵심 | 관련 주장 |
|---|---|---|---|---|---|
| Haxby 외, "A common, high-dimensional model of the representational space in human ventral temporal cortex" (하이퍼정렬) | 2011 | Neuron 72(2):404–416 | DOI 미확인 | 영화 시청 중의 시간 동기 fMRI 반응으로 반복 프로크루스테스 직교 변환을 추정함. 그 공통 공간에서 별도 실험의 피험자 간 분류를 수행함 | **(ii) SLA 메커니즘의 원전.** "영화로 변환을 얻고 다른 데이터에 적용한다"는 비전이적 구조까지 같음 |
| Chen, Chen, Yeshurun, Hasson, Haxby, Ramadge, "A Reduced-Dimension fMRI Shared Response Model" (SRM) | 2015 | NeurIPS 2015, 460–468 | https://neurips.cc/virtual/2015/oral/5374 | 시간 동기 다중 피험자 반응을 피험자별 직교 사상과 k차원 공유 반응으로 분해 | (ii) top-k 부분공간 직교 정렬과 사실상 같은 모형 |
| Cui, Kan, Li, Wang, Wu, SCORE | 2026 | arXiv 2608.19134 | https://arxiv.org/abs/2608.19134 | 배치 시점에 라벨 없이 EEG–이미지 랜드마크를 매칭해 직교 변환을 추정함. 테스트 배치를 쓰므로 전이적임(본문 확인). THINGS-EEG2 top-1 53.23% | (ii) 라벨 없는 직교 정렬의 최신 EEG 사례 |
| Zanini, Congedo, Jutten, Said, Berthoumieu, "Transfer Learning: A Riemannian Geometry Framework With Applications to BCI" | 2018 | IEEE TBME 65(5):1107–1116 | DOI 미확인 | 세션·피험자별 기준 공분산으로 리만 재중심화. 기준은 휴지기 같은 별도 데이터로 계산할 수 있음 | (i)의 2차 통계 버전 |
| He & Wu, EA | 2020 | IEEE TBME 67(2):399–410 | https://doi.org/10.1109/TBME.2019.2913914 | 평균 공분산으로 입력을 백색화 | 우리가 추가 이득 ≈0을 보인 비교 대상 |
| Rodrigues, Jutten, Congedo, RPA | 2019 | IEEE TBME 66(8):2390–2401 | https://doi.org/10.1109/TBME.2018.2889705 | 재중심화, 신축, 회전 순으로 정렬. 회전은 라벨 클래스 평균을 맞춰 추정 | (ii)와 (iii)의 라벨 기반 대응물 |
| Zhang & Liu, "Writer Adaptation with Style Transfer Mapping" (STM) | 2013 | IEEE TPAMI 35(7):1773–1787 | https://doi.org/10.1109/TPAMI.2012.239 | 소량 라벨 표본으로 소스 클래스 원형 쪽으로 가는 아핀 변환을 학습 | (iii)의 원형. Li 2020이 감정 EEG에 적용 |
| Lee, Pradeepkumar, Sun, "Test-Time Adaptation for EEG Foundation Models: A Systematic Study under Real-World Distribution Shifts" | 2026 | arXiv 2604.16926 (워크숍) | https://arxiv.org/abs/2604.16926 | CBraMod, TFM-Tokenizer, REVE에서 Tent·SHOT·T3A를 비교함. 경사 기반 방법은 자주 성능을 떨어뜨리고, T3A만 평균적으로 양의 이득. **감정 데이터와 LaBraM은 포함하지 않음** | "TTA 추가 이득 ≈0"이라는 우리 결론의 맥락 |
| Jiang, Zhao, Lu, LaBraM | 2024 | ICLR 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf | 우리가 쓰는 기반 FM | 공저자 Li-Ming Zhao가 PPDA의 제1저자임. 그래서 PPDA는 반드시 인용하고 비교해야 함 |

### 전이적 여부 요약
- **새 피험자 데이터를 전혀 쓰지 않음**: DResNet(2019), MAT(2025), TA2CL(2026), Zhou 외(2026)
- **테스트와 분리된 보정만 씀**: PPDA(비라벨 45 s), Li 2020(라벨 보정 세션), Lin & Jung 2017, Lin 2020(첫날 세션), EvoFA(세션 1 퓨샷), Bhosale 2022(K샷). FACE도 라벨 퓨샷이지만 보정과 테스트가 같은 시행에서 나옴.
- **온라인 전이적**: CLISA(테스트 스트림 정규화와 시행 단위 LDS)
- **전이적**: Zheng & Lu 2016, Lan 2019, DAN 2018, 층화 정규화 2021, MS-MDA 2021, PR-PL 2024, EEGMatch, DFF-Net 2023, SCORE 2026, 그리고 SFDA인 Imtiaz & Khan, "Towards Practical Emotion Recognition: An Unsupervised Source-Free Approach for EEG Domain Adaptation"(IEEE TAFFC, arXiv 2504.03707)
- Personal-Zscore는 통계 출처 미확인

### 보정 비용을 어떻게 보고했나
- CLISA 서론: 기존 DA 방법은 "일반적으로 새 피험자의 30분–1시간 데이터(라벨 불필요)"를 요구한다고 지적함
- PPDA: 비라벨 45 s(15–95 s 비교, 15 s 이후 평탄)
- EvoFA: 클래스당 1개 또는 5개 라벨 샘플
- FACE: 클래스당 1–10개 라벨 1초 창
- Bhosale: 5–25샷
- DFF-Net: 타깃 0.1% 라벨에 비라벨 전체를 더함
- Lin 2020: 첫날 라벨 세션 전체

**감정별 클립 하나 × 20 s/40 s/전체처럼 초 단위 예산과 감정 범주 포괄을 함께 통제한 연구는 찾지 못했습니다.**

## (B) 동향

1. **전이적 UDA가 여전히 주류이고, "플러그 앤 플레이"는 목표로만 언급됨.** TCA/TPT(2016), DAN(2018), MS-MDA(2021), PR-PL·EEGMatch(2023–24), SFDA(2025, TAFFC)로 이어지는 흐름 대부분이 새 피험자의 테스트 데이터를 적응에 씁니다. 2026년 리뷰(Li 외, "Cross-subject generalization for EEG emotion recognition: a review of methods, challenges, and future trends", Front. Comput. Neurosci., https://doi.org/10.3389/fncom.2026.1865513)는 SFDA와 인과 표현을 결합해 "진정한 플러그 앤 플레이 aBCI"를 만드는 것을 미래 과제로 꼽았습니다. 비전이적 짧은 보정은 아직 열린 문제로 인식된다는 뜻입니다.
2. **짧은 보정 연구는 적고 프로토콜이 제각각임.** 비라벨 45 s(PPDA), 라벨 보정 세션(Li 2020), 첫날 세션(Lin 2020), 세션 1의 1–5샷(EvoFA), 시행 내 무작위 1–10샷(FACE)처럼 모두 다릅니다. 감정 포괄과 초 단위 예산을 함께 통제한 연구는 없었습니다.
3. **퓨샷·메타러닝 수치가 부풀려져 있음.** 2021–2025년 퓨샷 연구는 SEED 계열에서 90%대(SEED-V 10샷 98.95%)를 보고합니다. 하지만 지원 세트와 테스트가 같은 시행에서 나오는 경우가 많고, 이는 블록 설계(TPAMI 2021)와 시행 내 누출(Lei 2025)이 정량화한 문제와 그대로 겹칩니다.
4. **공유 자극 정렬이 퍼지고 있지만 학습 단계에만 쓰임.** CLISA(2022), CL-SSTER(2024), TA2CL(2026)로 "같은 클립·같은 시간" 대조 학습이 자리 잡았습니다. 그러나 새 사용자에게는 정규화만 하거나 아무 적응도 하지 않습니다. 한편 하이퍼정렬과 SRM의 직교 정렬 아이디어는 EEG 배치 단계로 들어오고 있습니다(SCORE 2026, 시각 디코딩, 전이적).
5. **정규화가 숨은 주역임.** 층화 정규화, Personal-Zscore, MS-MDA 정규화, CLISA 적응형 정규화가 모두 큰 이득을 내지만 대개 테스트 피험자 자신의 통계를 씁니다. CLISA는 DE+MLP 기준선에도 같은 정규화를 적용했습니다. 2026년에는 학습 통계만 쓴다고 명시한 연구(Zhou 외)도 나왔습니다.
6. **FM 시대의 적응과 함께 평가가 엄격해지는 중.** EEG FM용 TTA 체계 연구(2026)는 경사 기반 TTA가 불안정하고 최적화가 없는 T3A가 가장 안정적이라고 보고했습니다(감정은 미포함). 감정 과제에서 FM 보정을 비전이적으로 평가한 연구는 찾지 못했습니다. 동시에 RSC(2025), 누출(2025), 체크포인트 선택(2026), 평가 리뷰(2025), LibEER(2024) 등 평가 비판이 빠르게 늘고 있습니다.

## (C) 신규성 위협과 범위 설정

### C-1. 가장 가까운 선행 연구

**(i) 보정 블록 평균 중심화**
- **프로토콜이 가장 가까움: PPDA(Zhao 2021).** 세션 맨 앞의 비라벨 45 s 블록을 테스트와 분리해 1분 안에 보정합니다. 메커니즘은 다릅니다(개인 인코더 재학습과 유사도 가중 앙상블). 이득은 +1.3%p뿐이고 15–95 s 구간에서 평탄합니다. SEED는 클립 순서가 고정되어 있어 이 블록은 첫 클립(긍정) 하나에 해당합니다. 따라서 우리의 "단일 감정 보정은 실패하고 감정 포괄이 길이보다 중요하다"는 결과의 외부 근거로 다시 읽을 수 있습니다. 단, 이건 우리 추론이라고 명시해야 합니다.
- **메커니즘이 가장 가까움(전이적)**: CLISA의 온라인 적응형 정규화, 층화 정규화, Personal-Zscore, MS-MDA 정규화. 모두 1차·2차 통계로 피험자 오프셋을 없애지만 테스트 데이터로 계산합니다.
- **교차 분야**: Zanini 2018 리만 재중심화(별도 기준 데이터를 쓸 수 있음), EA(He & Wu 2020).
- **"보정 없음" 대조군**: Zhou 2026(학습 통계만 사용), DResNet, MAT.
- **위협**: 중심화 자체는 새롭지 않습니다. 신규성은 다음 세 가지로 좁혀야 합니다.
  - (a) 비전이적이고, 보정과 테스트를 클립 단위로 분리하고, 초 단위 예산을 둔 프로토콜
  - (b) FM 클래스 토큰 공간에서 1차 중심화가 이득의 대부분을 차지하며, EA·층화 정규화·AdaBN·T3A·잠재 정렬을 더해도 ≈0이라는 경험적 결론
  - (c) 보정 블록의 감정 구성이 길이보다 중요하다는 발견. 평균 추정치가 클래스 구성에 따라 편향되기 때문입니다.

**(ii) SLA**
- **메커니즘 원전**: 하이퍼정렬(Haxby 2011)과 SRM(Chen 2015). 공유 영화에 대한 시간 동기 반응 쌍으로 피험자별 직교(프로크루스테스) 사상을 추정하고, k차원 공유 공간을 쓰고, 영화로 얻은 변환을 다른 데이터에 적용합니다. "학습 피험자 평균 = 템플릿"이라는 구조도 같습니다. **위협이 가장 큽니다.**
- **EEG 배치 단계에서 가장 가까움**: SCORE(2026). 라벨 없이 직교 변환을 쓰지만 전이적이고, 시각 디코딩이며, 앵커가 학습 피험자의 시간 궤적이 아니라 이미지 임베딩입니다.
- **감정 분야에서 가장 가까움**: CLISA, CL-SSTER, TA2CL. 같은 클립·같은 시간 쌍을 학습 목적에만 쓰고 새 사용자 정렬에는 쓰지 않습니다.
- **라벨 기반 대응물**: RPA(회전을 클래스 평균으로 추정), CorrCA/ISC.
- **주장 가능한 범위**: "감정 EEG에서, 학습 코호트가 본 표준 보정 영화와 코호트 평균 궤적을 템플릿으로 삼아, 감정 라벨과 테스트 데이터 없이 새 사용자의 FM 임베딩을 직교 정렬한 첫 사례(우리가 찾은 범위 안에서)". 여기에 중심화와의 보완성, 그리고 데이터셋 의존성(SEED-V +0.061, SEED ≈0)을 더합니다. 하이퍼정렬과 SRM은 반드시 원전으로 인용해야 합니다.
- **추가 위협**: SLA는 보정 영화가 학습 데이터에 있어야 작동합니다. 또 SEED 계열 LOSO에서는 테스트 클립도 학습 피험자가 이미 본 클립입니다. 그래서 개선이 감정 방향의 정렬인지 자극 고유 성분의 정렬인지 구분해야 합니다. 이는 RSC의 교차 피험자판이며, CLISA도 미지 자극에서 86.4%가 77.4%로 떨어졌습니다. TA2CL이 지적한 피험자 간 반응 지연도 "같은 초" 쌍짓기를 위협합니다.

**(iii) 보정 클래스 평균으로 프로토타입 회전·혼합**
- **감정 분야에서 가장 가까움**: Li 2020 TCYB. 보정 세션의 소량 라벨로 STM 아핀 변환을 학습하고 이후 세션에서 테스트합니다. 원형은 STM(Zhang & Liu 2013)입니다.
- **교차 분야**: RPA 회전.
- **프로토타입 퓨샷**: EvoFA(세션 분리, ProtoNet 기준선), SDA-FSL, Bhosale, FACE. 프로토타입을 조정하는 TTA인 T3A도 해당합니다.
- **위협**: 라벨 보정 표본의 클래스 평균으로 프로토타입을 만들거나 섞는 건 퓨샷 학습의 표준입니다.
- **주장 가능한 범위**:
  - 라벨이 표본마다 단 주석이 아니라 감정별 보정 클립의 자극 라벨(클립 수준의 약한 라벨)이라는 점
  - 클립 분리 평가와 20/40 s/전체 예산 곡선
  - "SEED-V에서 40 s 이상일 때만 이득이 있고 SEED에서는 없다"는 조건부 결과
- 같은 초 예산에서 STM, RPA식 회전, ProtoNet을 기준선으로 함께 넣는 것이 안전합니다.

### C-2. 그 밖의 주장별 위협
- **"현실적 비전이적 보정은 거의 연구되지 않았다"는 부분적으로만 맞습니다.** PPDA, Li 2020, Lin 2017/2020, EvoFA가 테스트와 분리된 보정을 씁니다. "처음"이라는 표현 대신 "클립 분리, 감정 포괄, 초 단위 예산을 함께 통제한 체계적 평가"로 한정해야 합니다.
- **"EA·AdaBN·T3A는 이득이 없다"에는 조건을 달아야 합니다.** Lee 2026은 EEG FM에서 T3A만 평균적으로 양의 이득을 냈다고 보고했습니다(감정 미포함, 중심화 없음). 우리 결론은 "중심화 이후, 감정 과제, 현실적 보정 조건에서"라고 명시해야 합니다.
- **퓨샷 문헌과 수치를 바로 비교할 수 없습니다.** FACE 등은 시행 내 무작위 분할입니다. 같은 분할을 재현한 뒤 우리 프로토콜로 바꾸면 얼마나 떨어지는지 보여주면, 그 자체가 기여가 됩니다.
- **Kilgallen 2025를 감정 EEG 함정으로 인용하면 부정확합니다.** 객체 범주 연구이므로, "일반 EEG 디코딩의 RSC"와 "SEED의 클립 반복 구조"를 연결하는 식으로 인용해야 합니다.

### C-3. 리뷰어 방어를 위한 점검 제안
1. **미지 자극 점검.** 테스트 클립을 학습 피험자 데이터에서 빼고 재학습합니다(CLISA의 일반화 시험 방식). 또는 세션마다 클립이 다른 SEED-IV에서 세션 단위로 분리해 시험합니다. 학습 코호트의 보정 영화 반응은 템플릿을 만들 때만 쓰고 분류기 학습에는 쓰지 않습니다.
2. **시간 동기 대조.** 같은 클립의 다른 초나 다른 클립과 짝지어 SLA를 돌려봅니다. 이득이 사라지면 시간 동기화가 핵심이라는 증거가 됩니다. TA2CL의 문제 제기에 대응해 ±1–3 s 지연을 허용하는 실험도 합니다.
3. **상한과 하한 보고.** 보정 없음(학습 통계만), 우리의 비전이적 보정, 테스트 세션 전체 통계를 쓴 전이적 상한을 함께 보고합니다.
4. **모델 선택 명시.** 체크포인트와 하이퍼파라미터를 테스트 피험자로 고르지 않았음을 밝힙니다(Suo 2026).
5. **클립 순서 혼입 점검.** SEED는 모든 피험자·세션에 같은 라벨 순서가 쓰입니다. 데이터셋이 라벨 벡터를 하나만 제공하는 것으로 알지만, 이번 조사에서 문헌으로 다시 확인하지는 않았습니다. 클립 위치별 정확도를 보고, 세션 내 표류와 중심화가 어떻게 상호작용하는지 확인합니다.
6. **비인과 평활 명시.** CLISA를 포함한 많은 SEED 파이프라인이 시행 전체에 LDS 평활을 겁니다. 우리가 이를 쓰지 않았거나 인과 버전을 썼음을 밝힙니다.

## (D) 반드시 인용할 논문

**필수 (위치 설정과 핵심 비교)**
- PPDA (Zhao, Yan, Lu, AAAI 2021)
- Li, Qiu, Shen, Liu, He (TCYB 2020)
- CLISA (TAFFC 2023)
- 층화 정규화 (Front. Neurosci. 2021)
- 전이적 DA 대표로 MS-MDA (2021) 또는 PR-PL (TAFFC 2024)
- 하이퍼정렬 (Haxby, Neuron 2011)과 SRM (Chen, NeurIPS 2015)
- EA (He & Wu, TBME 2020), Zanini (TBME 2018), RPA (TBME 2019), STM (TPAMI 2013)
- 블록 설계 (Li 외, TPAMI 2021), RSC (Kilgallen 2025, 일반 EEG로 인용)
- LaBraM (ICLR 2024), EEG FM TTA 연구 (Lee 외 2026)

**강력 권장**
- Zheng & Lu (IJCAI 2016), Lin & Jung (2017), Lin (JBHI 2020)
- EvoFA (2024), FACE (2025, 분할 방식 대비용)
- Personal-Zscore (TAFFC)
- 비전이적 기준선으로 DResNet (2019) 또는 MAT (2025)
- CL-SSTER (2024), TA2CL (2026), SCORE (2026)
- Ding (TAFFC 2021), Dmochowski (2012)
- Lei (2025), Suo (2026), Apicella (2024) 또는 Kukhilava (2025)

**선택**
- Bhosale (2022), SDA-FSL (2021), DFF-Net (2023), EEGMatch
- Lan (2019), DAN (2018), Zhou 외 (2026)
- Hajlaoui (2018), Imtiaz & Khan (TAFFC, SFDA), LibEER (2024), Li 외 2026 리뷰

## 미확인(UNVERIFIED) 항목
- Personal-Zscore가 테스트 피험자 통계를 어떻게 계산하는지(원문 접근 실패)
- Bhosale 2022의 데이터셋과 정확한 프로토콜
- Wang 2021(ESWA)이 비라벨 타깃을 함께 썼는지
- DOI 미확인 항목: SDA-FSL, DAN 2018, DResNet 2019, Dmochowski 2012, Haxby 2011, Zanini 2018. 서지 정보는 확인했습니다.
- SEED-V의 세션 간 클립 반복 여부와, SEED-IV/V에서 피험자 간 클립 순서가 같은지. SEED-IV는 세션마다 다른 24개 클립(총 72개)을 씁니다.
- 제목과 출처만 확인하고 보정 데이터 양은 확인하지 못해 표에서 뺀 논문:
  - MAML 기반 음악 감정 예측(ACM ICMI 2021 companion, doi:10.1145/3461615.3486569)
  - 연결성 특징과 메타 전이학습(Comput. Biol. Med. 2022)
  - "Towards Subject Agnostic Affective Emotion Recognition"(Jaiswal 외, CIKM 2023 워크숍, arXiv 2310.15189)
- 제목만 확인: Palazzo 외의 블록 설계 반박(arXiv 2012.03849), SPD 도메인 특이 배치 정규화(arXiv 2206.01323, 학회 미확인)
- 웹 검색 한도를 다 써서 2026년 하반기 감정 보정 논문이 빠졌을 수 있습니다.