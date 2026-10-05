# EEG 파운데이션 모델의 신규 피험자·세션 적응: 2023–2026 문헌 조사 (2026-10-05 기준)

**검증 방법.** 모든 논문의 제목, 저자, 연도, 학회 또는 arXiv ID는 arXiv, 학회 페이지, Crossref에서 확인했습니다. 핵심 수치는 가능한 한 PDF 원문에서 직접 확인했습니다.
- 웹 검색 한도(200회)가 후반에 소진됐습니다. 이후 검증은 arXiv API, Crossref, PDF 원문으로 했습니다.
- 확인하지 못한 항목은 **UNVERIFIED**로 표시했습니다.

**용어.**
- **피험자 내 분할**: 같은 피험자의 다른 시행(trial)으로 테스트합니다. 학습과 테스트에 같은 피험자가 들어갑니다.
- **피험자 분리**: 테스트 피험자를 학습에 쓰지 않습니다. 피험자 단일 제외 교차검증(LOSO)도 여기에 속합니다.

---

## (A) 논문 표

### A-1. 주요 EEG 파운데이션 모델(FM) 원 논문과 감정 평가 방식

| 인용 | 연도 | 학회 | 링크 | 평가 방식 (신규 피험자 데이터 양) | 방법 한 줄 | 감정 데이터셋 결과 | 우리 연구와의 관련성 |
|---|---|---|---|---|---|---|---|
| LaBraM (Jiang, Zhao, Lu) | 2024 | ICLR | [arXiv:2405.18765](https://arxiv.org/abs/2405.18765) | SEED-V(20명 판). 세션별 15시행을 5:5:5로 나누고 전 피험자를 합칩니다. **피험자 내 분할**이라 신규 피험자가 없습니다. | 채널 패치, 벡터양자화 신경 스펙트럼 예측, 약 2,500시간 사전학습 | SEED-V 정확도 0.4095(Base)~0.4102(Huge) | 우리 백본입니다. 원 논문에 LOSO 감정 평가가 없어 우리 수치와 직접 비교할 수 없습니다. |
| EEGPT (Wang G. et al.) | 2024 | NeurIPS | [DOI 10.52202/079017-1239](https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf) | 하류 과제는 BCIC-2A/2B, Sleep-EDFx, ERN, P300, TUAB, TUEV입니다. 감정은 하류 평가가 없고 SEED는 사전학습에만 썼습니다(원문 Table 1). 선형 프로빙 중심입니다. | 마스크 기반 이중 자기지도와 시공간 표현 정렬 | 원 논문에는 없습니다. 벤치마크에서는 SEED 피험자 내 70.83, 피험자 분리 49.37(아래 참조) | 선형 프로빙이 미세조정보다 나은 유일한 FM입니다(Compass). |
| BIOT (Yang, Westover, Sun) | 2023 | NeurIPS | [arXiv:2305.10351](https://arxiv.org/abs/2305.10351) | 발작, 임상 EEG, 심전도, 활동인식. 감정은 없습니다. | 채널별 토큰을 하나의 "문장"으로 이어 붙이는 트랜스포머 | CBraMod 표 기준 FACED 0.5118, SEED-V 0.3837 | 기준선 모델입니다. |
| CBraMod (Wang J. et al.) | 2025 | ICLR | [arXiv:2412.07236](https://arxiv.org/abs/2412.07236) | FACED: 피험자 1–80 학습, 81–100 검증, 101–123 테스트(**피험자 분리**). SEED-V(공개 16명): 시행 5:5:5(**피험자 내**). | 공간과 시간을 나눠 보는 교차형(criss-cross) 어텐션 | 균형 정확도(BAcc): FACED 0.5509, SEED-V 0.4091 | FACED 분할이 FM 감정 평가의 사실상 표준이 됐습니다. 우리 방법을 일반화 검증할 1순위 데이터셋입니다. |
| NeuroLM (Jiang W.-B. et al.) | 2025 | ICLR | [arXiv:2409.00101](https://arxiv.org/abs/2409.00101) | SEED: 15시행을 시간순 9:3:3, 세션을 합칩니다(피험자 내). | 텍스트와 정렬한 토크나이저와 대형 언어모델(LLM) 기반 다중과제 지시조정 | SEED BAcc: NeuroLM-XL 0.6034, LaBraM-Base 0.7318 | 모델을 키우고 언어모델화해도 감정 성능은 오히려 떨어졌습니다. |
| Gram (Li Z., Zheng, Lu B.-L. 등) | 2025/2026 | ICASSP 2025, IEEE TAFFC 2026 | [DOI 10.1109/ICASSP49660.2025.10890831](https://ieeexplore.ieee.org/document/10890831/), [TAFFC DOI 10.1109/TAFFC.2025.3638592](https://ieeexplore.ieee.org/document/11271181/) | 감정 분류를 포함한다고 보고되지만 분할 방식은 **UNVERIFIED**입니다. | 원시 신호 패치 양자화와 다중 관점 마스크 오토인코더, 약 7,000시간 | **UNVERIFIED** | LaBraM과 같은 SJTU 연구실 계열 FM입니다. |
| REVE (El Ouahidi et al.) | 2025 | NeurIPS | [arXiv:2510.21585](https://arxiv.org/abs/2510.21585) | FACED 피험자 분리(위와 같은 분할). 소수샷: 운동상상 데이터(BCI IV-2a)에서 세션 안 클래스당 1–20개 표본, 최근접 클래스 평균(NCM) 분류기 | 4차원 위치부호화, 92개 데이터셋·6만 시간·2.5만 명 사전학습 | FACED BAcc 0.5646 | **동결 임베딩과 클래스 평균 분류기**를 FM에 쓴 선례입니다(운동상상). 우리 "보정 클래스 평균 프로토타입"과 같은 계열입니다. |
| CSBrain (Zhou Y. et al.) | 2025 | NeurIPS | [arXiv:2506.23075](https://arxiv.org/abs/2506.23075) | CBraMod와 같은 분할 | 여러 시공간 스케일을 묶는 토큰화와 희소 어텐션 | FACED 0.5752, SEED-V 0.4197 | 비교 기준값입니다. |
| CodeBrain (Ma J. et al.) | 2026 | ICLR | [arXiv:2506.09110](https://arxiv.org/abs/2506.09110) | 같은 분할 | 시간·주파수를 분리한 토크나이저와 다중 스케일 상태공간 모델 | FACED 0.5941, SEED-V 0.4137 | 2026년 시점 FACED 최고 수준입니다. |
| mdJPT (Zhang Q., Zhong, Li Z., Shen X., Liu Q.) | 2025 | NeurIPS | [arXiv:2510.22197](https://arxiv.org/abs/2510.22197) | 데이터셋 단위 제외 평가. 소수샷은 대상 데이터셋 피험자 1/4로 분류기를 학습하고 나머지 3/4로 테스트합니다(개인 보정이 아닙니다). 제로샷도 평가합니다. | 감정 전용 다중 데이터셋 사전학습. 공분산 정렬 손실과 **피험자 간 정렬 손실(ISA)** 사용. ISA는 서로 다른 피험자의 **같은 자극, 같은 시각** 구간을 양성쌍으로 묶는 대조학습입니다. | 일반 FM(LaBraM, EEGPT) 대비 소수샷 AUROC +4.57%, 새 데이터셋 제로샷 정확도 +11.92%. 시각을 맞추지 않은 양성쌍으로 바꾸면 성능이 크게 떨어집니다(부록 E.2). | **자극 고정 정렬(SLA)의 가장 가까운 선행 연구**입니다. "같은 클립·같은 초"를 기준점으로 쓴다는 발상이 같고, 사전학습 손실로 쓴다는 점이 다릅니다. |

참고로 LUNA(NeurIPS 2025, [arXiv:2510.22257](https://arxiv.org/abs/2510.22257))도 SEED-V 피험자 내 분할에서 0.3918로 CBraMod보다 낮았습니다. 모델을 키워도 감정 성능이 오르지 않는 또 하나의 사례입니다.

### A-2. 벤치마크와 비판 연구

| 인용 | 연도 | 학회 | 링크 | 평가 방식 | 방법 | 감정 결과 | 관련성 |
|---|---|---|---|---|---|---|---|
| EEG-FM-Bench (Xiong W. et al.) | 2025/26 | **ICML 2026** | [arXiv:2508.17742](https://arxiv.org/abs/2508.17742) | 14개 데이터셋. 감정은 "기존 수치와 비교하기 위해" 피험자 내 분할(SEED 9:3:3, SEED-V 1:1:1)을 씁니다. SEED-VII만 피험자를 8:1:1로 분리합니다. | FM 7개 × (전체 미세조정, 동결, 저랭크 적응(LoRA)) | 단일과제 전체 미세조정 BAcc: SEED는 EEGPT 70.83, CSBrain 70.23, LaBraM 61.59. SEED-V는 CSBrain 38.23, LaBraM 24.35. **SEED-VII(피험자 분리, 7클래스, 우연 14.3%)는 EEGPT 26.42, LaBraM 20.39**. 다중과제 SEED에서 LaBraM 동결 52.13, 전체 67.71. | 피험자를 분리하면 감정 성능이 거의 우연 수준입니다. 원문은 SEED-V를 "notoriously difficult"라고 씁니다. |
| EEG-FM-Compass (Liu D. et al., 교신 Wu D.) | 2026 | **National Science Review** | [arXiv:2601.17883](https://arxiv.org/abs/2601.17883) | FM 12개와 특화모델 8개. 두 시나리오: LOSO(보정 없음), 그리고 **피험자 내 소수샷(SEED는 클래스당 영상 1개로 미세조정, 같은 피험자의 나머지 영상으로 테스트)** | 전체 미세조정, 선형 프로빙, 부록에 LoRA | SEED BAcc(3클래스). LOSO: CBraMod 53.61, ShallowConv 53.41, LaBraM 52.23, EEGNet 48.57. 소수샷: Neuro-GPT 55.90, CBraMod 55.82, Conformer 55.67, **LaBraM 47.00** | **보정량이 우리 설정과 거의 같습니다**(클래스당 1클립). 다만 그들은 피험자 내 미세조정만 하고, 우리는 모집단 미세조정에 보정을 더합니다. 원문 인용: 소수샷에서는 특화모델이 대부분 1위였고, *"their benefit diminishes when only limited calibration data are available"*, *"minimal calibration or even calibration-free adaptation remains a critical requirement… open and pressing challenge"*. LoRA 효과도 일정하지 않았습니다. |
| AdaBrain-Bench (Wu J., Ren Z., Wang J. et al.) | 2025 | arXiv(학회 미확인) | [arXiv:2507.09882](https://arxiv.org/abs/2507.09882) | 세 설정: 피험자 분리 / 다중 피험자(같은 피험자의 다른 시행·세션) / 소수샷(하류 학습 데이터의 2–50%) | FM 4개 전체 미세조정과 선형 프로빙 | SEED 피험자 분리: LaBraM 55.78, CBraMod 51.11, 특화모델 최고 LDMA 53.34. **SEED 다중 피험자: LaBraM 70.90**. SEED-IV: 피험자 분리 40.98, 다중 피험자 47.63 | 원문: *"inter-subject variability poses a greater challenge… than the inter-trial or session variability"*. 우리 가설을 직접 지지합니다. 데이터가 적을 때 사전학습 이점이 큽니다(SEED에서 학습 데이터 5–10%일 때 CBraMod가 무사전학습 대비 +10~12%p). |
| EEG-Arena, "Benchmarking EEG FMs at Scale" (Chen Z. et al.) | 2026(9월) | arXiv | [arXiv:2609.32743](https://arxiv.org/abs/2609.32743) | FM 30개, 지도학습 기준선 25개, 57개 과제. 피험자 내와 피험자 분리를 함께 보고합니다. 피험자 내 분할의 세부 정의는 **UNVERIFIED**입니다. | 대규모 통일 평가 | SEED BAcc(피험자 내 / 피험자 분리): REVE-base 89.32/58.20, ST-EEGFormer-small 94.09/56.76, LaBraM 68.80/48.25, mdJPT 64.61/53.21. 지도학습 최고(피험자 분리)는 SleepEEGNet 57.32 | **피험자 내 약 90%, 피험자 분리 약 50–58%로 30–40%p 차이**입니다. 사전학습 데이터를 600시간에서 30,000시간으로 늘려도 이득이 피험자 내 +6.69%p, 피험자 분리 +4.22%p로 분리 쪽이 더 작습니다. |
| NeuroAdapt-Bench (Lee G.J., Pradeepkumar, Sun J.) | 2026 | **MLHC 2026** | [arXiv:2604.16926](https://arxiv.org/abs/2604.16926) | 라벨 없는 대상 데이터로 하는 테스트 시점 적응(TTA, 테스트 데이터 사용). 모델은 CBraMod, TFM-Tokenizer, REVE. 데이터는 임상, 수면, 귀 EEG이고 **감정은 없습니다**. | Tent, SHOT, T3A 비교 | 감정 없음 | 원문: *"TTA methods yield inconsistent gains and often degrade performance… optimization-free methods showing greater stability"*. 우리에게서 T3A와 AdaBN이 거의 0이었던 결과와 맞습니다. |
| The Identity Trap (Lin J.-Y., Wu Y.C., Jung T.-P.) | 2026 | arXiv | [arXiv:2606.06647](https://arxiv.org/abs/2606.06647) | LaBraM, CBraMod, REVE. 휴지기 위주 4개 데이터셋(감정 없음). 피험자 축 소거는 **학습 fold에서만** 맞추므로 테스트 데이터를 쓰지 않습니다. | FMScope 진단: 분산 분해, 최소제곱 개념 소거(LEACE)로 피험자 축 제거, 1/f 제거, 층별 프로빙, 피험자 내 방향 일관성 | 피험자 분산이 무작위 기준의 13–89배(12쌍 모두)입니다. **미세조정하면 +10~+63%p 더 커집니다.** 피험자 축을 제거하면 라벨 판별이 +6~+12%p 오릅니다. 공통 표지가 없는 과제에서는 피험자마다 대비 방향이 공유되지 않습니다. | **우리 결과 (1)과 (3)을 직접 뒷받침합니다.** 피험자 성분이 하나의 선형 축이라 평균 중심화로 상당 부분 제거되고, 남는 오차는 피험자별 방향 차이(회전)라는 해석과 맞습니다. 이들의 진단 지표를 그대로 가져다 쓰길 권합니다. |
| Kuruppu, Wagh, Kremen, Varatharajah | 2026 | J. Neural Eng. 23:021001 | [DOI 10.1088/1741-2552/ae4455](https://iopscience.iop.org/article/10.1088/1741-2552/ae4455) ([arXiv:2507.11783](https://arxiv.org/abs/2507.11783)) | 리뷰 | 초기 FM 10개 분석 | — | 평가 방식이 제각각이고 실사용 유용성 평가가 부족하다고 지적합니다. 현실적 보정 프로토콜의 근거로 쓸 수 있습니다. |
| Lee N., Barmpas et al., "Are Large Brainwave FMs Capable Yet?" | 2025 | ICML (PMLR 267) | [arXiv:2507.01196](https://arxiv.org/abs/2507.01196) | 뇌-컴퓨터 인터페이스(BCI) 과제(기억, 수면 등). 감정 없음 | 미세조정과 LoRA | 전통 딥러닝 대비 이득이 0.9–1.2%p에 그칩니다. | "FM이 단순 기준선을 크게 못 이긴다"는 비판 계열입니다. |
| 기타 2026 비판: NeuroAtlas / Zare(음성 대조) / Zhou & Roy(운동상상 감사) | 2026 | arXiv | [2605.14698](https://arxiv.org/abs/2605.14698), [2607.24519](https://arxiv.org/abs/2607.24519), [2609.23924](https://arxiv.org/abs/2609.23924) | 임상과 운동상상. 피험자 분리 | — | 감정 없음. EEG 전용 FM이 일반 시계열 FM을 꾸준히 이기지 못합니다. 임상 데이터 CAUEEG에서는 고전 특징(0.734)이 CBraMod(0.669)와 REVE(0.568)를 이깁니다. 4클래스 운동상상에서는 지도학습 모델 전부가 FM 설정 전부를 이깁니다. | FM이 단순 기준선에 지는 사례 모음입니다. |

### A-3. FM을 신규 피험자에 적응시키는 방법

| 인용 | 연도 | 학회 | 링크 | 평가 방식 (신규 피험자 데이터 양) | 방법 | 결과 | 관련성 |
|---|---|---|---|---|---|---|---|
| Tao X., Chen K., "Separating personal from population gains when calibrating EEG FMs for new users" | 2026(9/28) | arXiv | [arXiv:2609.34801](https://arxiv.org/abs/2609.34801) | 동결 CBraMod, REVE, LaBraM. 운동상상 3개 데이터셋, 235명. 세션 앞 절반의 라벨로 개인 어댑터를 맞춥니다(PhysioNet 22시행, 다른 데이터셋 80–120시행). 소수 라벨은 5–40시행. | 개인 어댑터(특징별 가산 오프셋 FiLM, LoRA)를 모집단 모델과 **"교환" 어댑터(남의 어댑터)**와 비교합니다. 모집단 학습량을 4배로 늘린 조건도 봅니다. | 개인 어댑터는 모집단 대비 +1.5~5.4%p, 교환 대비 +2.3~7.3%p입니다. **모집단 학습량을 4배로 늘리면 중앙값 +1.0~2.0%p로 줄어듭니다.** 소수 라벨 보정은 1개 데이터셋에서만 일관됐습니다. CBraMod에서는 라벨 없는 문맥도 메타학습 초기화도 대조군을 넘지 못했습니다. 유클리드 정렬(EA)은 PhysioNet에서 성능을 떨어뜨렸습니다. | **사용자 가설에 대한 가장 강한 반론이자 보완입니다.** 보정 이득은 약한 모집단 모델 때문에 생긴 착시일 수 있습니다. 교환 보정과 모집단 규모를 바꾸는 대조 실험이 필요합니다. 가산 오프셋 FiLM은 도메인 중심화에 라벨을 쓴 일반화판이라고 볼 수 있습니다. |
| Stacked LoRA (Sarhane, …, Lys, Lioi) | 2026 | IEEE WCCI 2026 | [arXiv:2607.03094](https://arxiv.org/abs/2607.03094) | REVE, LaBraM, LUNA. 운동상상과 임상 데이터. 보정량 **UNVERIFIED** | 전역 LoRA 위에 피험자별 LoRA를 쌓습니다. | 큰 코호트에서는 공유 어댑터로 충분했습니다. 세션 변동이 큰 임상 데이터에서는 피험자별 어댑터가 필수였습니다. | "모집단 + 개인" 구조의 운동상상 선례입니다. |
| STEM (An S., Kim S., Park S.H.) | 2026 | MICCAI | [MICCAI 2026 paper 1006](https://papers.miccai.org/miccai-2026/1006-Paper2756.html) | 운동상상과 수면. 신규 피험자 소수샷(양 **UNVERIFIED**) | 메트릭 메타학습과 대조학습으로 피험자 성분과 과제 성분을 분리합니다. | 감정 없음 | 사전학습 단계에서 피험자 성분을 떼어내는 접근으로, 우리 사후 중심화와 대비됩니다. |
| SCOPE (Ma J. et al.) | 2026 | arXiv | [arXiv:2602.17251](https://arxiv.org/abs/2602.17251) | LaBraM 등 FM 5개, 6개 과제(SEED 이진 분류 포함). 라벨 있는 피험자 5–50%와 라벨 없는 데이터 | 코호트 프로토타입, 신뢰도 가중 의사라벨, 경량 어댑터 | SEED(라벨 30%) LaBraM AUC 75.23 | FM이 과신하고, 예측이 한쪽으로 쏠리고, 표현이 표류한다고 보고합니다. 우리 의사라벨 회전이 거의 0이었던 결과와 맞습니다. |
| NeuroTTT (Wang S. et al.) | 2025 | arXiv | [arXiv:2509.26301](https://arxiv.org/abs/2509.26301) | CBraMod, LaBraM. 상상 발화, 스트레스, 운동상상. 테스트 샘플마다 학습하고 엔트로피를 줄입니다(정규화 통계 갱신, 테스트 데이터 사용). | 과제 관련 자기지도 미세조정과 테스트 시점 학습 | 감정 없음 | 테스트 데이터를 쓰는 대안입니다. 우리 설정은 테스트 데이터를 쓰지 않습니다. |
| FUSED (Gong P. et al.) | 2026 | arXiv | [arXiv:2605.00857](https://arxiv.org/abs/2605.00857) | 원천 데이터 없이 하는 영역 적응. 라벨 없는 대상 피험자 전체를 씁니다(테스트 데이터 사용). 운동상상, 감정, SSVEP. 감정 데이터셋과 수치는 **UNVERIFIED** | FM과 소형 특화모델의 이중 분기 공동 적응, 의사라벨 | 최고 성능 주장(수치 미확인) | 테스트 데이터를 쓰는 접근의 대표 사례로 대비용입니다. |
| ECHO (Liu C. et al.) | 2025 | arXiv | [arXiv:2509.22556](https://arxiv.org/abs/2509.22556) | 지지 샘플을 문맥에 넣는 문맥 내 학습(ICL). 데이터셋 세부 **UNVERIFIED** | 디코더 중심 seq2seq 대형 EEG 모델 | 미확인 | EEG FM에서 ICL을 쓴 거의 유일한 사례입니다. 신규 피험자 감정 보정에 ICL을 쓴 연구는 찾지 못했습니다. |
| SCORE (Cui Z., Kan, Li S., Wang Z., Wu D.) | 2026 | arXiv | [arXiv:2608.19134](https://arxiv.org/abs/2608.19134) | EEG로 이미지를 찾는 검색 과제(EEG→이미지 검색). 신규 피험자의 라벨 없는 테스트 데이터를 씁니다. 인코더는 동결합니다. | 랜드마크를 매칭한 뒤 **직교 변환으로 피험자 좌표계를 복원**합니다. | THINGS-EEG2 Top-1 53.23%(+17.45%p) | **우리 결과 (3)과 SLA에 가장 가까운 동시대 연구**입니다. 원문: *"different subjects preserve similar relationships among concepts but express them along different coordinate directions"*. |
| SATTC (Huang Q., Zhu W.) | 2026 | **CVPR 2026** | [arXiv:2603.20738](https://arxiv.org/abs/2603.20738) | EEG→이미지 검색, LOSO. 테스트 집합의 구조를 씁니다. | 피험자별 임베딩 중심화·백색화, CSLS(교차 도메인 유사도 지역 스케일링, 검색에서 허브 편향을 줄이는 보정), 상호 최근접 이웃 | 감정 없음 | 임베딩 수준 피험자 중심화의 동시대 선례입니다. 다만 테스트 데이터를 씁니다. |
| Channel Adaptation for EEG FMs (Kokate, Aristimunha, Truong, Delorme) | 2026 | arXiv | [arXiv:2604.23091](https://arxiv.org/abs/2604.23091) | FM 5개 × 5개 과제. 감정 없음 | 리만 재중심화를 포함한 4가지 채널 적응 비교 | 최적 방법이 모델 구조마다 다릅니다. | 입력 공간 재중심화가 FM에서 일정하게 듣지 않습니다. 우리에게서 EA가 거의 0이었던 결과와 맞습니다. |

### A-4. FM 이전 선행 연구 (참신성 위협 판단에 필수)

| 인용 | 연도 | 학회 | 링크 | 핵심 |
|---|---|---|---|---|
| CLISA (Shen X., Liu X., Hu X., Zhang D., Song S.) | 2023 | IEEE TAFFC | [DOI 10.1109/TAFFC.2022.3164516](https://arxiv.org/abs/2109.09559) | **같은 감정 자극을 본 다른 피험자의 EEG**를 양성쌍으로 쓰는 대조학습(SEED, THU-EP). 학습 단계에서 씁니다. |
| Haxby et al., hyperalignment | 2011 | Neuron 72:404–416 | [DOI 10.1016/j.neuron.2011.08.026](https://doi.org/10.1016/j.neuron.2011.08.026) | 같은 영화를 본 시점별 반응으로 피험자 사이 **Procrustes 회전**을 구합니다(fMRI). SLA와 개념적으로 같습니다. |
| Rodrigues, Jutten, Congedo, 리만 프로크루스테스 분석(RPA) | 2019 | IEEE TBME 66:2390–2401 | [DOI 10.1109/TBME.2018.2889705](https://ieeexplore.ieee.org/document/8588384/) | 재중심화, 스케일 조정, 회전. 회전은 **클래스 평균(라벨)**으로 구합니다. |
| Zanini et al., 리만 재중심화 | 2018 | IEEE TBME 65:1107–1116 | [DOI 10.1109/TBME.2017.2742541](https://doi.org/10.1109/TBME.2017.2742541) | 세션별 기준 공분산으로 재중심화합니다. 라벨 없는 보정 데이터로 세션을 중심화하는 고전 선례입니다. |
| He & Wu, 유클리드 정렬(EA) / Wu, "Revisiting EA" | 2020 / 2025 | IEEE TBME 67:399–410 / J. Neural Eng. 22:031005 | [DOI 10.1109/TBME.2019.2913914](https://doi.org/10.1109/TBME.2019.2913914), [arXiv:2502.09203](https://arxiv.org/abs/2502.09203) | EA 원전과 13개 패러다임 리뷰입니다. |
| Fdez et al., 계층 정규화 | 2021 | Front. Neurosci. | [DOI 10.3389/fnins.2021.626277](https://www.frontiersin.org/articles/10.3389/fnins.2021.626277/full) | 피험자·세션별 정규화입니다. |
| Zhao, Yan, Lu, Plug-and-Play DA | 2021 | AAAI 35:863–870 | [DOI 10.1609/aaai.v35i1.16169](https://ojs.aaai.org/index.php/AAAI/article/view/16169) | 신규 피험자의 **라벨 없는 짧은 보정 데이터**로 개인 성분만 맞춥니다. 감정 분야에서 "보정 블록" 프로토콜의 선례입니다. |
| FACED 데이터셋 (Chen J., …, Shen X., Zhang D.) | 2023 | Sci. Data | [DOI 10.1038/s41597-023-02650-w](https://doi.org/10.1038/s41597-023-02650-w) | 123명 전원이 같은 28개 클립을 봤습니다. SLA 검증에 이상적입니다. |

---

## (B) 2024–2026 경향과 가설 판정

**질문: 최근 연구는 EEG FM에서 피험자 보정·적응이 병목이라고 보는가?**
**답: "피험자 이동이 병목"이라는 점은 강하게 지지됩니다.** 하지만 "보정만 잘하면 된다"는 강한 형태는 반론이 있습니다. 그리고 감정 분야에서 현실적인 보정 블록을 FM에 직접 평가한 연구는 사실상 없습니다(빈자리입니다).

1. **감정 FM의 대표 수치는 대부분 피험자 내 분할에서 나왔고, 피험자를 분리하면 크게 떨어집니다.**
   - LaBraM, CBraMod, CSBrain, CodeBrain, LUNA, NeuroLM, EEG-FM-Bench는 SEED와 SEED-V를 시행 분할로 평가합니다.
   - 피험자를 분리한 SEED 성적은 Compass, AdaBrain, EEG-Arena 모두 BAcc 48–58%입니다. 같은 모델의 피험자 내 성적은 70–94%입니다.
   - SEED-VII 피험자 분리는 7클래스에서 20–26%입니다.
   - AdaBrain은 "피험자 간 변동이 세션 간 변동보다 어렵다"고 명시합니다.

2. **FM 표현은 피험자 정체성이 지배하고, 미세조정이 이를 키웁니다(Identity Trap).** 피험자 성분은 하나의 선형 축이라 간단한 조작으로 제거됩니다. 남는 문제는 피험자마다 대비 방향이 다르다는 점입니다. SCORE도 다른 과제(EEG→이미지)에서 "피험자 차이의 상당 부분은 좌표계 회전"이라고 보였습니다. 두 결과 모두 우리 (1) 중심화가 대부분의 이득, (3) 잔차는 회전이라는 결과와 일치합니다.

3. **모델과 데이터를 키워도 피험자 분리 격차는 잘 안 줄어듭니다.**
   - Compass: 모델 크기와 성능 사이에 관계가 없습니다.
   - EEG-Arena: 사전학습 데이터를 50배 늘려도 피험자 분리는 +4.22%p에 그칩니다.
   - NeuroAtlas, Lee 등(ICML 2025): 기준선 대비 이득이 미미합니다.

   그 결과 2026년에 FM 적응 연구가 급증했습니다: Stacked LoRA, 개인 어댑터, STEM, SCOPE, NeuroTTT, NeuroOnline, FUSED, NeuroAdapt-Bench. Compass는 "최소 보정 또는 무보정 적응"을 미해결 핵심 과제로 꼽습니다.

4. **무거운 적응은 불안정합니다. 가볍고 최적화 없는 보정이 더 안정적입니다.**
   - 테스트 시점 적응은 자주 성능을 떨어뜨리고, 최적화 없는 방법(예: T3A)만 안정적이었습니다(NeuroAdapt-Bench).
   - 선형 프로빙은 부족하고 LoRA는 결과가 엇갈립니다(Compass, EEG-FM-Bench).
   - 소수 시행으로 피험자 내 미세조정하면 FM이 특화모델에 집니다(Compass의 SEED에서 LaBraM 47.00로 LOSO 52.23보다 낮음).
   - 그러니 "FM을 보정 데이터로 다시 학습"하기보다 "모집단 모델에 가벼운 보정"을 얹는 쪽이 합리적입니다. 우리 접근이 정확히 이 틈에 있습니다.

5. **반론: 보정 이득은 모집단 모델이 얼마나 강한지에 달려 있습니다.**
   - Tao & Chen 2026: 모집단 학습을 4배로 늘리면 개인 이득이 1–2%p로 줄었습니다. 적은 라벨로 하는 보정과 메타학습 초기화는 대조군을 넘지 못했습니다.
   - mdJPT: 감정 전용 사전학습(자극 정렬 포함)이 일반 FM을 이겼습니다. 표현(사전학습) 품질도 여전히 중요합니다.
   - AdaBrain, EEG-Arena: 데이터가 적을수록 사전학습 이점이 큽니다.
   - 결론적으로 정확한 진술은 "보정만 중요"가 아니라 "강한 모집단 표현 + 피험자 이동만 겨냥한 가벼운 보정"입니다.

6. **평가 방식의 빈틈.** 신규 사용자에게 짧은 보정 블록만 쓰고 테스트 데이터를 정규화·적응에 쓰지 않는 평가를 FM과 감정 조합에서 한 논문은 찾지 못했습니다. 가장 가까운 것은 Compass의 "클래스당 영상 1개" 피험자 내 미세조정과 Tao & Chen(운동상상)입니다.

---

## (C) 참신성 위협과 포지셔닝

**1. 도메인 중심화 (보정 블록 평균 빼기)**
- **개념상 선행이 많습니다.**
  - 입력·공분산 공간: Zanini 재중심화(2018), EA(2020), 계층 정규화(2021), AdaBN.
  - 라벨 없는 신규 피험자 보정: Plug-and-Play DA(2021).
  - FM 임베딩 공간:
    - Identity Trap의 피험자 축 소거: 학습 피험자로 맞추며, 세션별 중심화는 아닙니다.
    - SATTC의 피험자별 임베딩 백색화: 테스트 데이터를 쓰며, 이미지 검색 과제입니다.
    - Tao & Chen의 가산 오프셋 FiLM: 라벨을 쓰며, 운동상상입니다.
- **찾지 못한 것.** 다음을 보인 논문은 이번 검색 범위에서 찾지 못했습니다.
  - FM과 피험자 분리 감정 인식에서, 테스트 데이터 없이, **보정 블록 평균 중심화만으로 이득 대부분이 나온다**는 결과
  - EA, AdaBN, T3A, 의사라벨 회전을 위에 얹어도 **거의 0**이라는 결과

  이 두 가지와 "잔차 = 회전" 진단이 주장 가능한 기여입니다. 다만 "최초"라고 쓸 때는 "우리가 아는 한"으로 한정하세요.

**2. SLA (자극 고정 정렬)**
- **개념적 참신성은 제한적입니다.**
  - Hyperalignment(2011): 같은 영화의 시점별 반응으로 Procrustes 회전. 우리와 같은 원리입니다.
  - CLISA(2023)와 mdJPT ISA(2025): 같은 자극·같은 시각 양성쌍. mdJPT는 시각을 맞추는 것이 필수라는 것까지 보였습니다.
  - SCORE(2026): 신규 피험자 좌표를 직교 변환으로 복원.
  - RPA(2019): 클래스 평균으로 회전.
- **차별점.** 사전학습이나 테스트 데이터 랜드마크가 아니라, **배치 시점에 학습 없이, 테스트 데이터 없이, 짧은 자극 보정 블록으로 FM 임베딩을 회전**합니다. 거기에 "SEED-V에서는 효과(+0.061), SEED에서는 없음"이라는, 언제 회전이 필요한지에 대한 경험적 결과가 더해집니다.
- **권장 표현:** "EEG FM 배치를 위한 hyperalignment".

**3. 라벨 누출 지적에 대비해야 합니다.**
- SEED 계열에서는 클립이 정해지면 감정도 정해집니다. 그래서 SLA가 "감정 라벨 없음"이라고 주장하면 리뷰어가 사실상 라벨이라고 볼 수 있습니다.
- 도메인 중심화도 "감정별로 1클립씩 균형 잡힌 보정"이라는 점에서 라벨 정보를 씁니다. 프로토타입 혼합은 명시적으로 라벨을 씁니다.
- **권장:** "label-free"가 아니라 **"알려진 자극으로 하는 보정(known-stimulus calibration)"**으로 표현하세요. 일반 BCI 보정에서 큐를 아는 것과 같은 위치입니다.
- **추가 실험 제안:** 중립 클립 하나만, 휴지기만, 또는 감정이 불균형한 보정 블록으로 중심화해 보세요. 그래도 효과가 유지되면 정말로 라벨이 필요 없다는 주장이 됩니다.

**4. 범위 설정 제안**
- **필수 대조군(Tao & Chen 권고):**
  - 교환 보정: 다른 사용자나 다른 세션의 보정 평균으로 중심화
  - 모집단 규모 변화: 학습 피험자 수 증감
  - 무작위 클립 매칭 SLA: 자극 고정이 효과의 원인인지 확인
- **진단 지표(Identity Trap 차용):** 중심화 전후의 피험자 분산 비율, 그리고 SLA 전후의 피험자 간 클래스 대비 벡터 코사인
- **양 끝 기준선:** Compass 방식 두 가지(LOSO 무보정, 클래스당 영상 1개로 피험자 내 미세조정)를 같은 보정량으로 함께 보고하세요. 이 비교가 가장 설득력 있는 그림이 될 것입니다.
- **일반화:** FACED(123명, 공통 28클립, FM 표준 피험자 분리 분할)와 FM 하나 이상(CBraMod 또는 REVE, 둘 다 공개)에서 재현하세요. 벤치마크와 비교되도록 클립 정확도와 함께 구간(window) 단위 BAcc도 보고하세요.

---

## (D) 반드시 인용할 논문

- **FM 원전:** LaBraM(ICLR'24), CBraMod(ICLR'25), EEGPT(NeurIPS'24), BIOT(NeurIPS'23), NeuroLM(ICLR'25), REVE(NeurIPS'25), CSBrain(NeurIPS'25). CodeBrain(ICLR'26)은 선택입니다.
- **감정 특화와 자극 정렬:** mdJPT(NeurIPS'25), CLISA(TAFFC'23), FACED(Sci Data'23)
- **벤치마크와 비판:** EEG-FM-Bench(ICML'26), EEG-FM-Compass(NSR'26), AdaBrain-Bench(arXiv'25), EEG-Arena(arXiv'26), NeuroAdapt-Bench(MLHC'26), The Identity Trap(arXiv'26), Kuruppu 등(JNE'26), Lee 등(ICML'25)
- **보정과 적응:** Tao & Chen(arXiv'26, 필수), Stacked LoRA(WCCI'26), SCORE(arXiv'26), SATTC(CVPR'26)
- **고전 정렬 선행:** Hyperalignment(Neuron'11), RPA(TBME'19), Zanini(TBME'18), EA(TBME'20)와 Wu(JNE'25), 계층 정규화(Front. Neurosci.'21), Plug-and-Play DA(AAAI'21)

**UNVERIFIED 목록:**
- Gram의 감정 분할과 수치
- FUSED와 ECHO의 데이터셋과 수치
- STEM과 Stacked LoRA의 보정량
- EEG-Arena의 피험자 내 분할 정의
- AdaBrain-Bench의 학회 게재 여부(arXiv만 확인)
- BrainWave(arXiv:2402.10251)의 감정 평가 방식. 프로토타입 기반 소수샷 분류는 검색 요약에서만 봤습니다.

**확인 필요:** 웹 검색 한도(200회)가 이 세션에서 소진됐습니다. 다른 하위 조사도 웹 검색이 필요하면 `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` 값을 올려야 합니다.
