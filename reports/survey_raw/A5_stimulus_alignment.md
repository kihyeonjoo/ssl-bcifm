# 신규 피험자의 "자극 동기화 정렬(SLA)" 신규성 점검 — 선행연구 조사 (2026-10-05 기준)

## 핵심 요약
- **이미 선행연구가 있어 SLA의 새로운 점으로 주장할 수 없는 것**
  - 공유 자극에 대한 시간 동기 반응으로 신규 피험자를 회전·선형 정렬하는 메커니즘. 기능자기공명영상(fMRI), 피질뇌파(ECoG), 뇌자도(MEG)에서 이미 확립되어 있습니다.
  - 뇌전도(EEG) 딥 임베딩에 대해 "자극 쌍 기반 직교 프로크루스테스 + 항등 행렬 쪽 정규화"를 쓰는 것. SCORE(2026-08, arXiv)가 이미 했습니다.
  - EEG 감정인식에서 같은 영상 타임라인의 훈련 피험자 뇌파를 추론에 쓰는 것. GRN(ICANN 2026)이 SEED에서 이미 했습니다.
- **아직 방어 가능한 것 (검색 범위 내)**
  - 감정인식에서 다음을 모두 결합한 프로토콜에 해당하는 선행연구는 찾지 못했습니다.
    - 학습 시점도 시험 시점도 아닌 **보정 시점**에, 세션 시작의 짧은 영상을 씁니다.
    - **EEG 파운데이션 모델 임베딩의 클립×초 그룹 템플릿**과 대응시켜 사용자별 직교 변환을 추정합니다.
    - 이후 데이터에는 **시험 자극 정보 없이** 그 변환을 그대로 적용합니다.
  - 2026년 EEG 감정인식 피험자 간 일반화 리뷰(83편 분석)도 프로크루스테스, 초정렬, 공유 반응 모델(SRM), 정준상관분석(CCA)을 전혀 언급하지 않습니다.

---

## (A) 논문 표

**약어**
- 초정렬(HA, hyperalignment)
- 공유 반응 모델(SRM) / 강건 공유 반응 모델(RSRM)
- 다중집합 정준상관분석(MCCA, M-CCA)
- 위상 고정값(PLV)
- 피험자 간 상관(ISC)
- 최적수송(OT) / 융합 불균형 그로모프-바서슈타인(FUGW)
- 교차영역 유사도 국소 스케일링(CSLS)
- 접공간 정렬(TSA) / 리만 프로크루스테스 분석(RPA)
- 전이 템플릿 정준상관분석(tt-CCA) / 최소제곱 변환(LST)
- 정상상태 시각유발전위(SSVEP)
- 단일 피험자 제외 교차검증(LOSO)

**관계 등급**: 같음 / 가까움 / 먼 관련. 아래 논문은 모두 검색으로 실재를 확인했습니다. 일부만 확인한 경우 셀 안에 표시했습니다.

| # | 인용 | 연도 | 학회/저널 | 링크 | 모달리티 | 정렬 대상 | 변환 | 신규 피험자에게 필요한 데이터 | SLA와의 관계 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Cui, Kan, Li, Wang, Wu, "SCORE: Subject Coordinate Recovery for Label-Free Cross-Subject EEG-to-Image Retrieval" | 2026 (8월) | arXiv 2608.19134 | https://arxiv.org/abs/2608.19134 | EEG (THINGS-EEG2, Alljoined-1.6M), 이미지 검색 | 처음부터 학습한 EEG 인코더의 딥 임베딩 | **직교 프로크루스테스**. 분석 실험: 신규 피험자를 개별 소스 피험자에 맞춤. 같은 이미지 개념 쌍으로 적합하며 Top-1은 직교 28.2%, 릿지 20.0%, 무정렬 16.9%. 실제 방법: 라벨 없음, 차원별 모멘트 정합 + CSLS 랜드마크 + 신뢰도 가중 + **항등 정규화**(λ=ρ‖X̃ᵀWỸ‖₂) | 분석 실험: 시험 개념의 약 2/3 쌍(3-fold). 방법: 라벨 없는 EEG 배치와 후보 이미지 갤러리 | **가까움, 메커니즘상 가장 가까움.** 차이: 이산 이미지 시행 대 영화 연속 타임라인, 감정이 아님, 파운데이션 모델이 아님, 그룹 템플릿이 아닌 개별 피험자 대상, 보정 프로토콜 없음 |
| 2 | Meng, "Group Resonance Network: Learnable Prototypes and Multi-Subject Resonance for EEG Emotion Recognition" | 2026 | ICANN 2026 (arXiv 2603.11119) | https://arxiv.org/abs/2603.11119 | EEG 감정 (SEED, DEAP) | 원시 채널 동기 특징: 시험 샘플과, **같은 자극 타임라인**의 훈련 피험자 참조 집합 사이의 PLV/코히어런스 | 변환 없음 (동기 텐서를 네트워크에 입력) | 시험 샘플마다 자극·시점 식별이 필요하고, 보정 단계는 없음 | **가까움, 감정 분야에서 가장 가까움.** 차이: 정렬 변환이 없음. 시험 자극 정보를 시험 때 사용하는데, SEED류에서는 클립 식별이 곧 라벨이라 배포 환경에서는 비현실적. SLA는 보정 클립만 사용 |
| 3 | Shen, Liu, Hu, Zhang, Song, CLISA | 2023 (온라인 2022) | IEEE TAC 14(3) | https://arxiv.org/abs/2109.09559 | EEG 감정 (SEED, THU-EP) | 딥 표현 | 비선형. 같은 자극 쌍 대조학습, **학습 시점만** | 없음 (제로샷) | 가까움, 학습 시점 판. 이미 인용 중 |
| 4 | Shen, Tao, Chen, Song, Liu, Zhang, CL-SSTER | 2024 | NeuroImage 301:120890 | https://arxiv.org/abs/2402.14213 | EEG (FACED 감정 영상, 음성) | 시공간 합성곱 표현 | 비선형. "정확히 시간 정렬된" 피험자 쌍을 양성 쌍으로 사용 (학습 시점) | 없음 | 가까움, 학습 시점. 센서 공간 ISC 0.034에서 0.062로 상승 |
| 5 | Xie, Zheng, Xiao, Lu, Liu, TA2CL | 2026 | arXiv 2605.22379 | https://arxiv.org/abs/2605.22379 | EEG 감정 (FACED, SEED, SEED-V) | 딥 표현 | 비선형. 전역 하드 정렬 대신 국소 비동기 매칭 대조학습 | 없음 | 먼 관련~가까움. 학습 시점이며, 지연 허용 아이디어가 있음 |
| 6 | Graves, Clayton, Soh, RSRM을 EEG에 적용 | 2021 | IEEE SIEDS | https://ieeexplore.ieee.org/document/9483745/ | EEG BCI | 센서/특징 | 선형 RSRM | 피험자 데이터 | 가까움. 이미 인용 중 |
| 7 | Zhang, Borst, Kass, Anderson, "Inter-subject alignment of MEG datasets in a common representational space" | 2017 | Hum Brain Mapp 38(9) | https://pmc.ncbi.nlm.nih.gov/articles/PMC6866831/ | MEG | 자극·반응 시점에 고정한 시행 평균 센서 시계열 | 선형 M-CCA | 모든 피험자를 함께 적합 (보류 피험자 절차 없음) | 가까움. 피험자 간 디코딩이 피험자 내 수준에 근접 |
| 8 | de Cheveigné et al., "Multiway CCA of brain data" | 2019 | NeuroImage 186 | doi:10.1016/j.neuroimage.2018.11.026 | EEG/MEG | 원시 센서 | 선형 MCCA | 함께 적합 | 먼 관련 (잡음 제거, ISC) |
| 9 | Katthi & Ganapathy, Deep MCCA | 2021 | ICASSP | https://arxiv.org/abs/2103.06478 | EEG (음성/음악) | 원시 | 비선형 (공유 층 오토인코더) | 함께 적합 | 먼 관련 |
| 10 | Ravishankar, Toneva, Wehbe, 피험자 간 예측으로 단일 시행 MEG 잡음 제거 | 2021 | Front Comput Neurosci 15 | https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.737324/full | MEG | 센서 | 선형 (쌍별 릿지, SRM) | 같은 이야기 자극 | 먼 관련 |
| 11 | Bhattacharjee, Zada, …, Hasson, Goldstein, Nastase, "Aligning brains into a shared space improves their alignment with LLMs" | 2025 온라인 / 2026 | Nat Comput Sci 6:169–178 | https://www.nature.com/articles/s43588-025-00900-y | ECoG | 전극 고감마 | SRM. 정규직교 Wⱼ로 보류 참가자를 기존 공유공간에 "회전" | 같은 팟캐스트의 9/10 구간 | 가까움 (침습 기록, 신규 참가자 회전 정렬) |
| 12 | Heo, Wisniewska, Lee, Lee, 공유공간 정렬을 통한 피험자 간 의미 디코딩 | 2026 | arXiv 2607.19394 | https://arxiv.org/abs/2607.19394 | ECoG | 전극 | SRM. 보류 피험자는 투영만 추정 | 같은 팟캐스트의 학습 구간 | 가까움 |
| 13 | Kneeland, Jiang, Nunes, Scotti, Delorme, Xu, ENIGMA | 2026 | arXiv 2602.10361 | https://arxiv.org/abs/2602.10361 | EEG (THINGS-EEG2, Alljoined) | **백본 임베딩** (184차원) | 피험자별 선형 Nz×Nz 층. 손실은 MSE + InfoNCE, 타깃은 CLIP | 약 15분 (약 4,000 시행, 자극 식별 포함) | 가까움 (임베딩 수준 선형 온보딩). 단, 타깃이 다른 피험자 반응이 아닌 자극 특징이고, 직교가 아니며, 데이터량이 훨씬 많음 |
| 14 | Liu, Liu, Zhou, Lu, Zheng, MindCross | 2026 | AAAI 2026 (arXiv 2511.14196) | https://arxiv.org/abs/2511.14196 | EEG (SEED-DV), fMRI | 피험자별/공유 인코더 | 비선형 (Top-K 협업) | EEG 40–600 샘플 | 먼 관련 |
| 15 | Bleuzé, Mattout, Congedo, Tangent space alignment (TSA) | 2022 | Front Hum Neurosci | https://pmc.ncbi.nlm.nih.gov/articles/PMC9755175/ | EEG BCI (18개 데이터베이스) | 공분산의 접공간 벡터 | 재중심·스케일 후, **클래스 평균** 교차곱의 SVD로 직교 회전. 특이벡터를 잘라 부분 정렬 | 대상 라벨 (클래스 평균) | 가까움. 수학은 같고, 앵커가 클래스 평균이냐 클립×초냐만 다름 |
| 16 | Rodrigues, Jutten, Congedo, Riemannian Procrustes Analysis (RPA) | 2019 | IEEE TBME 66(8) | doi:10.1109/TBME.2018.2889705 | EEG BCI | 공분산 행렬 | 리만 재중심·신축·회전 | 대상 라벨 일부 | 가까움~먼 관련 |
| 17 | Yuan, Chen, Wang, Gao, Gao, tt-CCA | 2015 | J Neural Eng 12(4) | https://iopscience.iop.org/article/10.1088/1741-2560/12/4/046006 | SSVEP | 기존 피험자 템플릿 | CCA 기반 템플릿 전이 | 보정 없음 | 먼 관련 (유발전위 유사 사례) |
| 18 | Chiang, Wei, Nakanishi, Jung, LST | 2021 | J Neural Eng 18(1):016002 | https://par.nsf.gov/biblio/10341426-boosting-template-based-ssvep-decoding-cross-domain-transfer-learning | SSVEP | 원시 다채널 시행을 소스 템플릿으로 매핑 | 선형 최소제곱 (비직교) | 자극별 소수 보정 시행 | 가까움 (유발전위 판 "자극 동기 템플릿 매핑") |
| 19 | Michalke & Rieger, 기능적 피험자 간 정렬이 해부학적 정렬보다 우수 | 2025 | CCN 2025 | https://2025.ccneuro.org/abstract_pdf/Michalke_2025_Functional_Inter-Subject_Alignment_Outperforms_Anatomical_Alignment.pdf | fMRI (오디오 영화) | PCA 800차원 | LOSO에서 보류 피험자를 템플릿에 프로크루스테스 (HA). MCCA/ICA는 회귀 투영 | 같은 영화 | 먼 관련. 다만 논의에서 "공유 기능 템플릿으로 BCI 보정을 웜스타트"하자고 직접 제안함 (동기 부여용 인용 가치) |
| 20 | Tang & Huth, 참가자·자극 모달리티를 넘는 의미 언어 디코딩 | 2025 | Curr Biol 35(5):1023–1032 | doi:10.1016/j.cub.2025.01.024 | fMRI | 복셀 | 선형 변환기 (목표 참가자 복셀에서 참조 참가자 복셀로) | **무성영화 70분**, 언어 라벨 없음 | 가까운 유사 사례 (영화 시청으로 신규 참가자를 정렬한 뒤 디코더 이전) |
| 21 | Thual, Benchetrit, Geilert, Rapin, Makarov, Banville, King, 뇌 기능 정렬로 신규 피험자 디코딩 향상 | 2023 | arXiv 2312.06467 | https://arxiv.org/abs/2312.06467 | fMRI | 피질 꼭짓점 | FUGW 최적수송 | 영화 시청 데이터. 100분 미만에서 단일 피험자 모델보다 우세 | 가까운 유사 사례 |
| 22 | Ferrante, Boccato, Ozcelik, VanRullen, Toschi, "Through their eyes" | 2024 | Imaging Neuroscience | https://doi.org/10.1162/imag_a_00170 | fMRI (NSD) | 복셀 | 릿지(최고), 프로크루스테스, HA 비교 | 공유 이미지. 1/2–1/4 분량에서도 동작 | 유사 사례 |
| 23 | Scotti et al., MindEye2 | 2024 | arXiv 2403.11207 (ICML 2024라고 알고 있으나 학회 표기는 재확인 못 함) | https://arxiv.org/abs/2403.11207 | fMRI | 복셀을 공유 잠재공간으로 | 피험자별 선형 층 | 1시간 | 유사 사례 |
| 24 | Dai et al., MindAligner | 2025 | ICML 2025 | https://arxiv.org/abs/2502.05034 | fMRI | 복셀 | 선형 뇌 전이 행렬, 교차 자극 소프트 정렬 | 1시간 (NSD의 2.5%), 동일 자극 불필요 | 유사 사례 |
| 25 | Ho, Horikawa, Majima, Cheng, Kamitani, 신경 코드 변환 | 2023 | NeuroImage 271 | doi:10.1016/j.neuroimage.2023.120007 | fMRI | 복셀 | 선형 변환기 (동일 이미지 쌍, 라벨 없음) | 동일 이미지 세트 | 유사 사례 |
| 26 | Bazeille, DuPre, Richard, Poline, Thirion, 피험자 간 디코딩으로 본 기능 정렬 평가 | 2021 | NeuroImage 245 | doi:10.1016/j.neuroimage.2021.118683 | fMRI | 복셀 | 조각별/서치라이트 프로크루스테스, OT, SRM | 정렬용 공유 데이터 | 벤치마크 (관심영역에서는 SRM, 전뇌에서는 OT가 최고) |
| 27 | Andreella, Finos, Lindquist, ProMises ("Enhanced hyperalignment via spatial prior information") | 2023 | Hum Brain Mapp 44(4) | https://doi.org/10.1002/hbm.26170 | fMRI | 복셀 | 프로크루스테스 + 행렬 von Mises–Fisher 사전분포. 사후최대 해는 polar(XᵢᵀM + kF) | — | **SLA 수축 단계의 방법론적 선행.** F=I이면 "항등 행렬 쪽 수축"의 원리적인 버전 |

**추가로 확인했으나 먼 관련인 것**
- SATTC (Huang & Zhu, CVPR 2026, arXiv 2603.20738): 라벨 없는 화이트닝 + CSLS. 자극 대응 없음.
- Lopes et al. 2026 (arXiv 2606.16462): 피험자별 인코더, 운동심상.
- Csaky et al. 2023 (Hum Brain Mapp, doi:10.1002/hbm.26500): MEG 피험자 임베딩.
- Richard et al. MultiView ICA (NeurIPS 2020, arXiv 2006.06635): MEG 포함.
- Geirnaert et al. SI-GCCA (arXiv 2401.17841, 저널 미표기).
- Dmochowski, Sajda, Dias, Parra 2012 (Front Hum Neurosci): 감정 영화 EEG의 상관 성분. ISC 피크가 각성 장면과 일치.
- MindAdapter (KDD 2026, arXiv 2605.24679): fMRI, 소수 공유 자극 사용.
- Wang et al. 2025 (Nat Comput Sci): 공유 자극 없는 신경 코드 변환.
- Al-Wasity et al. 2020 (Sci Rep): fMRI 운동심상 HA.
- Zhang et al. 2026 (TASLP, arXiv 2608.22420): MEG/EEG 음성. SRM/프로크루스테스 아님.
- Li et al. 2026 리뷰 (Front Comput Neurosci, https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2026.1865513/full): 위 정렬 기법을 전혀 다루지 않음.
- Haxby et al. 2020 eLife 리뷰 (https://elifesciences.org/articles/56601).

---

## (B) 신규성 판정

### 방어 불가능한 주장 (삭제 권장)
1. **"EEG에 초정렬/공유 반응 모델을 처음 적용"**
   - RSRM-EEG(2021), MEG M-CCA(2017), EEG/MEG MCCA(2019), SCORE(2026)가 이미 있습니다.
2. **"EEG 임베딩에 자극 쌍 기반 직교 프로크루스테스를 처음 적용"**
   - SCORE가 신규 피험자에 대해 이미 했습니다. 직교 방식이 릿지보다 낫다는 것까지 보였습니다.
3. **"같은 영상 타임라인의 다른 피험자 반응을 감정인식에 처음 활용"**
   - GRN이 SEED에서 시험 시점에 이미 사용했습니다.
   - 학습 시점 사용은 CLISA, CL-SSTER, TA2CL이 있습니다.
4. **"항등 행렬 쪽 수축"이나 "상위 k 부분공간"이 새롭다**
   - 수축은 ProMises(2023)와 SCORE(2026)에 이미 있습니다.
   - 부분공간 제한은 TSA의 특이벡터 절단, SRM의 k차원 공유공간에 이미 있습니다.
5. **수식어 없는 "라벨 없는(label-free) 보정"**
   - SEED류에서는 클립이 곧 감정입니다. 그래서 (클립, 초) 대응은 클래스 라벨보다 **더 강한 감독 신호**입니다.
   - "정렬 목적함수에 감정 라벨을 쓰지 않는다"로 한정해서 써야 합니다.

### 방어 가능한 주장 (검색 범위 내)
감정인식에서 다음 네 가지를 결합한 연구는 찾지 못했습니다.
- **보정 시점**에만 공유 자극을 씁니다 (학습 시점도, 시험 시점도 아님).
- 세션 시작의 짧은 영상(20초/40초)을 씁니다.
- **EEG 파운데이션 모델 임베딩의 클립×초 그룹 템플릿**에 직교 변환으로 맞춥니다.
- 그 변환을 **시험 자극의 식별·시점 정보 없이** 이후 모든 데이터에 적용합니다. 시험 데이터를 미리 쓰지도 않습니다.

### 제안 문구 (영문)
- **본문 주장**
  > "To our knowledge, this is the first work to calibrate a new user's EEG foundation-model embeddings for emotion recognition using stimulus-locked shared responses: a short session-start film clip is aligned, second by second, to group templates built from training subjects who watched the same clip, via an orthogonal Procrustes transform estimated without using emotion labels in the alignment objective, and the transform is then applied unchanged to all subsequent data."
- **반드시 함께 쓸 한정 문장**
  > "Hyperalignment-style Procrustes alignment of EEG embeddings has recently been explored for EEG-to-image retrieval (Cui et al., 2026), and stimulus-locked references from other subjects have been used at inference time for emotion recognition (Meng, 2026); SLA differs in using the shared stimulus only at calibration time and never using the identity or timing of test stimuli."
- **더 보수적인 대안 (리뷰어가 강하게 문제 삼을 경우)**
  > "We adapt hyperalignment—Procrustes alignment on time-locked responses to a shared stimulus (Haxby et al., 2011)—as a lightweight calibration step for an EEG foundation model in cross-subject emotion recognition."

### 가장 가까운 선행연구로 반드시 인용할 목록
- **최우선**: SCORE(2026), GRN(2026)
- **학습 시점 유사 연구**: CLISA, CL-SSTER, (선택) TA2CL
- **초정렬·공유 반응 모델 계열**: Haxby 2011/2020, Chen 2015 SRM, Jiahui 2020
- **다른 모달리티의 신규 피험자 정렬**: Bhattacharjee 2025/26 (ECoG, 신규 참가자 회전), Zhang 2017 (MEG), de Cheveigné 2019
- **EEG 쪽 기존 정렬**: RSRM-EEG 2021, TSA 2022 (+RPA 2019), LST 2021 (+tt-CCA 2015)
- **fMRI 유사 사례**: Tang & Huth 2025, Thual 2023
- **수축 단계 선행**: ProMises 2023
- **선택**: Michalke & Rieger 2025 (BCI 웜스타트 제안), ENIGMA 2026 (임베딩 선형 온보딩)

### "자극 동기화가 핵심"이라는 주장을 지키는 데 필요한 대조 실험
1. **시간 셔플 대응**: 같은 클립 안에서 무작위 초와 짝짓기. 이것이 SLA와 비슷하게 나오면 "자극 동기" 주장은 "클래스 앵커 회전"으로 약해집니다.
2. **클래스 평균 앵커 프로크루스테스**: TSA 방식, 같은 보정 창 사용.
3. **대응 없는 프로크루스테스**: SCORE 방식, 라벨 없음.
4. **재중심만**: 유클리드 정렬(EA) 방식.
5. **릿지 (비직교) 사상**.

---

## (C) 선행연구에서 얻는 SLA 개선 아이디어
1. **ProMises식 사후최대 수축**
   - R = polar(XᵀT + κI)로 두고, κ = ρ‖XᵀT‖₂로 잡습니다. SCORE와 같은 방식으로 ρ가 단위 없는 값이 됩니다.
   - 직교성이 유지되고, 보정 길이(20초/40초/전체)에 자동으로 적응합니다.
   - 현재의 (1−α)I + αR 보간은 직교 행렬이 아니게 되므로 재직교화가 필요합니다.
2. **신뢰도 가중 또는 선택 ("적지만 좋은 초" 사용)**
   - 각 (클립, 초) 대응에 템플릿의 피험자 간 신뢰도(훈련 피험자 간 반분 상관 또는 ISC)로 가중치를 줍니다. SCORE의 마진 기반 신뢰도 가중도 참고할 수 있습니다.
   - 근거: EEG의 ISC는 매우 낮습니다(CL-SSTER: 0.03–0.06). 그래서 초 단위 템플릿은 잡음이 큽니다. 반면 Dmochowski 2012는 ISC 피크가 각성 장면에 몰린다고 보고했습니다.
   - SEED에서 이득이 거의 0인 원인에 대한 가설로도 쓸 수 있습니다.
3. **지연 허용**
   - 클립별로 ±1–2초 시차를 탐색하거나 소프트 동적 시간 정합(DTW)을 씁니다.
   - 근거: TA2CL이 보고한 감정 반응의 피험자 간 비동기성.
4. **템플릿 개선**
   - 단순 평균 대신 훈련 피험자들끼리 일반화 프로크루스테스 분석(GPA)이나 SRM으로 반복 정렬한 공유 템플릿을 씁니다.
   - 사용자와 비슷한 훈련 피험자 부분집합으로 템플릿을 만듭니다. Zhang, Gobbini, Haxby, Feilong (bioRxiv 2025, https://www.biorxiv.org/content/10.1101/2025.02.19.639148v1)에서 사용자와 맞는 템플릿이 더 우수했습니다.
5. **혼합 앵커**
   - 시간 앵커와 클래스 평균 앵커(TSA 방식)를 같은 프로크루스테스에 함께 쌓습니다.
   - SEED처럼 시간 동기의 이득이 없는 데이터셋에 대비하는 방법입니다.
6. **연결성/하이브리드 정렬 목표**
   - 참고: Jiahui et al., eLife 2023 (https://doi.org/10.7554/eLife.86037), Busch et al., NeuroImage 2021 (https://www.sciencedirect.com/science/article/pii/S1053811921002524).
   - 훈련 피험자가 보지 않은 영화로 보정해도 정렬할 수 있어 배포 유연성이 커집니다. 짧은 보정의 안정화에도 도움이 될 수 있습니다.
7. **임베딩 이방성 처리**
   - 정규화 초정렬(Xu et al., IEEE SSP 2012, doi:10.1109/SSP.2012.6319668)로 CCA와 프로크루스테스 사이를 보간하거나, SCORE식 차원별 모멘트 정합을 한 뒤 회전합니다.
8. **직교 대 릿지**
   - EEG 임베딩에서 수백 쌍 정도일 때는 직교가 릿지보다 낫습니다(SCORE). fMRI처럼 데이터가 많을 때는 릿지가 최고였습니다(Ferrante).
   - 20초/40초 조건은 직교를 유지하고, "전체" 조건에서는 릿지나 저랭크 릿지를 대조군으로 보고하는 것을 권장합니다.

---

## 남은 위험
- 세션 공용 웹 검색 한도(200회)가 소진되어 다음은 확인하지 못했습니다(UNVERIFIED).
  - Turek et al.의 준지도 SRM (ICASSP 2017)
  - Guntupalli et al. 2018의 연결성 초정렬
  - 중국어권 학술지 (예: "超对齐 脑电 情绪")
- SCORE가 2026-08-19에 나온 것처럼 이 분야는 매우 빠르게 바뀝니다. 투고 직전에 arXiv에서 다음 검색어로 다시 확인하는 것을 권장합니다: "Procrustes EEG emotion", "hyperalignment EEG foundation model", "calibration clip EEG".
