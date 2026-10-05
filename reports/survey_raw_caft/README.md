# CAFT 관련 연구 조사 — 원 보고서 (2026-10-05)

종합본은 [../CAFT_RELATED_WORK.pdf](../CAFT_RELATED_WORK.pdf) (`make_caft_survey_pdf.py`).  여기 다섯 파일은 갈래별 조사 에이전트가 낸
원 보고서다 (논문 표 · 링크 · 가장 가까운 선행 정밀 비교 · 리뷰어가 요구할 실험 · 확인 못 한 항목).  지난 캘리브레이션 조사
([../survey_raw/](../survey_raw/README.md), A1~A5) 의 확인된 인용을 재사용했고, 그 조사의 정정 사항 (예: CLISA 게재처) 은 B3 머리에 있다.

| 파일 | 갈래 |
|---|---|
| B1_adaptation_aware_training.md | 배포 때의 정규화를 학습 때도 똑같이 하는 학습법 — ARM-BN, MetaBN, 음성 SAT · CMVN, 근전도 다중 스트림 AdaBN (Du 2017), 잠재 정렬, 층화 정규화, 짧은 이론 명제 |
| B2_train_like_you_test.md | '시험처럼 학습' 원리 — Matching Networks, FLYP, TTT, BCI 메타학습, Tao & Chen 2026 (학습형 문맥 조건화 무효과), 이름 충돌 |
| B3_shared_stimulus_alignment.md | 학습 단계의 같은 자극 사람 간 정렬 — CLISA → CL-SSTER → mdJPT → MSHCL → TA2CL, 정렬 손실의 위험과 ② 개선 단서, 자극 지름길 (Gerster 2026) |
| B4_fm_finetuning.md | EEG 파운데이션 모델 파인튜닝 · 피험자 적응, 벤치마크 프로토콜, 동시 투고 (PACE, SCC, SEAM, NeuroContext-TTA) |
| B5_emotion_cross_subject.md | 교차 피험자 감정 인식의 학습 중 정규화 · 불변 학습, 캘리브레이션 계열, SEED-V 문헌 수치와 공정 비교 |

주의
- 일부 사실은 2026-10-05 기준 공개 코드로 확인했다 (mdJPT 샘플러 · 정규화, DMMR · MS-MDA · DAEST 평가 코드, PACE 설정).  보고 수치가 그 코드 경로로 나왔는지는 확인하지 못했다.
- ICLR 2027 투고작은 OpenReview 본문이 막혀 초록 기준이다 (PACE 만 익명 공개 코드로 보강).  심사 중이라 내용이 바뀔 수 있다.
- 에이전트가 찾아 확인한 내용이므로, 논문에 인용하기 전에 원문을 한 번 더 본다.
