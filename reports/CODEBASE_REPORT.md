# 현재 상태 보고서 — EA + LaBraM LOSO 파이프라인

작성 2026-09-23. 코드 수정 없이 읽기만 해서 정리한 것.
다음 단계(피험자 간 클래스 prototype 정렬 loss, 그 전 특징 공간 진단)를 위한 사전 조사.

모든 줄 번호는 이 문서 작성 시점 기준.

---

## 1. 데이터

| 항목 | 값 | 근거 |
|---|---|---|
| 데이터셋 | SEED (`Preprocessed_EEG`) | [seed_raw_dataset.py:380](../data/seed_raw_dataset.py#L380), 설정 `root: /mnt/data/original/SEED` |
| 피험자 | 15명 | [seed_raw_dataset.py:107](../data/seed_raw_dataset.py#L107) `n_subjects = 15`, [:156](../data/seed_raw_dataset.py#L156) |
| 세션 | 1인당 3회 | [seed_raw_dataset.py:157](../data/seed_raw_dataset.py#L157) 기본 `[1,2,3]` |
| 클래스 | 3 (negative/neutral/positive → 0/1/2) | 라벨 시퀀스 [:30](../data/seed_raw_dataset.py#L30), 매핑 [:353](../data/seed_raw_dataset.py#L353) `_SEED_LABEL_SEQ[clip-1] + 1` |
| 클래스별 trial | **세션당 5개씩 균등** (15클립 = 5+5+5). 피험자당 15개/클래스, 전체 225개/클래스 | 라벨 시퀀스 + 저장된 메타로 확인 |
| 채널 | 62개 | [seed_raw_dataset.py:51-61](../data/seed_raw_dataset.py#L51-L61) |
| 샘플링 레이트 | 200 Hz (**선언만**, 코드가 리샘플링하지 않음) | [seed_raw_dataset.py:109](../data/seed_raw_dataset.py#L109) `FS = 200` |
| trial 길이 | 측정값 **184–264초, 평균 224.5초** (세그먼트 개수 기반이라 실제는 최대 +4초) | `logits/calibdiag_S1.npz` 메타에서 계산 |

피험자 1명당 4초 창 **2,526개** (3세션 합), 클래스별 834 / 822 / 870.

채널 목록 (LaBraM 10-20 명명):

```
FP1 FPZ FP2 | AF3 AF4 | F7 F5 F3 F1 FZ F2 F4 F6 F8 | FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8
T7 C5 C3 C1 CZ C2 C4 C6 T8 | TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8
P7 P5 P3 P1 PZ P2 P4 P6 P8 | PO7 PO5 PO3 POZ PO4 PO6 PO8 | CB1 O1 OZ O2 CB2
```

### LaBraM 입력 맞추기가 일어나는 곳

- **리샘플링: 없음.** SEED의 `Preprocessed_EEG`를 그대로 읽음 ([:340](../data/seed_raw_dataset.py#L340) `sio.loadmat`). 필터·다운샘플 코드가 저장소 어디에도 없음.
- **채널 매핑**: 이름 기준. [finetune_labram_hemi_aux.py:86-87](../finetune_labram_hemi_aux.py#L86-L87) `_get_input_chans`가 SEED 62개 이름을 LaBraM의 130여 개 1020 어휘 인덱스로 변환하고 앞에 CLS 자리 `0`을 붙임. 모델에 [:194](../finetune_labram_hemi_aux.py#L194)에서 `self.input_chans`로 들어가 LaBraM의 공간 임베딩 조회에 쓰임. 채널 **재배열 없음**, 62개 순서 유지.
- **패치 분할**: [seed_raw_dataset.py:238](../data/seed_raw_dataset.py#L238) `eeg.reshape(62, n_patches, 200)`. 설정은 `segment_length: 800, patch_size: 200` → **4초 창 = 4패치**. 슬라이딩 [:359-376](../data/seed_raw_dataset.py#L359-L376), `step: 800`이라 **겹침 없음**.

---

## 2. 전처리 순서

실제 적용 순서는 [`__getitem__`](../data/seed_raw_dataset.py#L216-L248) 안에 전부 있음.

```
.mat 로드 (필터·리샘플 없음)          :340
  → 4초 창으로 자르기 (겹침 없음)       :370-376
  → EA 적용   W @ X                    :221     ← ea=True일 때만
  → norm                               :224-235 ← EA일 때 건너뜀 (:227 pass)
  → (62, 4, 200) reshape               :238
```

**대역통과·리샘플링: 확인 불가.** 저장소 코드에 없음. SEED가 배포 시점에 적용한 것을 그대로 사용. [alignment.py:28](../data/alignment.py#L28) 주석이 "band-passed but not artifact-rejected"라고만 적고 있어 차단 주파수와 원 샘플링 레이트는 코드로 확인 불가. 데이터셋 배포 문서를 인용해야 함.

### 스케일 처리

- `ea=False`, `norm='scale100'`: [:231](../data/seed_raw_dataset.py#L231) `eeg / 100.0` — µV → 0.1 mV. LaBraM 공식 전처리와 같은 절대 진폭.
- `ea=True` (현재 설정): **`norm`을 통째로 건너뜀** ([:224-227](../data/seed_raw_dataset.py#L224-L227)). 진폭은 [alignment.py:140](../data/alignment.py#L140) `self.scale * W`의 α = `ea_scale: 0.2`가 결정. 생성자에 경고도 있음 ([:164-171](../data/seed_raw_dataset.py#L164-L171)).

> **정정 (2026-09-23 측정 후)**: 이 자리에 처음 적었던 "diag 화이트닝 후 채널 std가 정확히 α가 되므로 입력이 기준의 1.5배"는 **틀렸다**. 실측하면 α=0.2일 때 채널 std 중앙값은 0.1444로, scale100 기준값 0.1278의 **1.13배**다. α가 그대로 출력 std가 되지 않는 이유는 R̄가 소수의 초고진폭 세그먼트에 끌려 typical 세그먼트보다 1.4–1.9배 부풀려지기 때문이다(아래 참조). 따라서 [:96-98](../data/seed_raw_dataset.py#L96-L98) 주석의 "0.2가 0.131과 맞는다"는 서술은 **표현이 혼란스러울 뿐 선택 자체는 타당하다** — α가 아니라 α를 통과시킨 결과가 0.131 근처에 온다는 뜻. 측정 근거는 [measure_ea_scale.py](../measure_ea_scale.py), 결과는 [S0_REPORT.md](S0_REPORT.md).

---

## 3. EA 구현

| 질문 | 답 | 근거 |
|---|---|---|
| R̄ 단위 | **세션별** — `(subj, sess)` | [:251-252](../data/seed_raw_dataset.py#L251-L252) `_group_key`, 설정 `ea_scope: session` |
| 어떤 trial로 | **그 데이터셋 객체가 가진 전부** | [:254-273](../data/seed_raw_dataset.py#L254-L273) `_fit_alignment` |
| 학습 피험자 | 각자 자기 세션 전체 (그 세션 15클립 전부) | 위와 동일 |
| 테스트 피험자 | **3세션 각각, 해당 세션 전체 trial로 한 번에** | [run_cv:867](../finetune_labram_hemi_aux.py#L867)이 test_ds를 `sessions=[1,2,3]`으로 따로 만들고, 그 안에서 세션별 fit |
| 역제곱근 | `eigh` 후 고유값 바닥 적용 | [alignment.py:49-61](../data/alignment.py#L49-L61) |
| 정규화 | **고유값 바닥 `eps·λmax` (eps=1e-6) + 상위 5% 파워 세그먼트 제외**, 그 외 shrinkage 없음 | [alignment.py:59-60](../data/alignment.py#L59-L60), [:78-85](../data/alignment.py#L78-L85) |

**diag 모드** (현재 실행 중 `ea_mode: diag`): [alignment.py:134-137](../data/alignment.py#L134-L137)에서 `W = diag(R̄)^(-1/2)`. 채널별 진폭만 맞추고 **채널 간 상관은 그대로 둠**. `full`은 [:139](../data/alignment.py#L139) `inverse_sqrt(R)`.

### EA 후 단위행렬 확인 방법 (실행하지 않음)

이미 도구가 있음. [seed_raw_dataset.py:321](../data/seed_raw_dataset.py#L321) `alignment_report()` → [alignment.py:150-168](../data/alignment.py#L150-L168) `report()`. 그룹별로 `(key, ‖R̄_after/α² − I‖_F, 채널 std 중앙값)`을 반환. 내부에서 300개 세그먼트를 **균등 간격으로** 추출 ([:162](../data/alignment.py#L162)) — 앞부분만 쓰면 클립 몇 개에 치우쳐 화이트닝 품질이 과소평가되기 때문.

```python
ds = SEEDRawDataset(root=..., subjects=[1], sessions=[1,2,3], ea=True, ea_mode='diag', ...)
for key, dev, med in ds.alignment_report():
    print(key, dev, med)
```

**단, diag 모드에는 이 지표를 그대로 쓰면 안 됨.** diag는 대각만 1로 만들고 비대각은 손대지 않으므로 `‖R̄−I‖_F`가 0 근처로 갈 이유가 없음. diag를 검증하려면 **`diag(R̄)/α²`가 1 벡터에 가까운지**를 봐야 하고, 비대각 노름은 "EA가 무엇을 남겨뒀는지"를 재는 별도 값으로 읽어야 함. `report()`는 이 구분을 하지 않음.

---

## 4. LOSO 프로토콜

[run_cv:845-870](../finetune_labram_hemi_aux.py#L845-L870)

- **테스트**: 피험자 1명, 3세션 전부
- **검증**: 테스트 피험자 **다음 2명** (순환 순서, RNG 없음) — [:855-859](../finetune_labram_hemi_aux.py#L855-L859). `n_val_subjects: 2`
- **학습**: 나머지 12명 — [:860](../finetune_labram_hemi_aux.py#L860). **피험자 분리 맞음.**
- **모델 선택**: 검증 macro-F1만 — [:548-553](../finetune_labram_hemi_aux.py#L548-L553). 테스트는 매 에폭 계산되지만 ([:544](../finetune_labram_hemi_aux.py#L544)) 선택에 쓰이지 않음. 보고 숫자는 "최고 val-F1 에폭의 test" — [:640](../finetune_labram_hemi_aux.py#L640).
- **시드**: **없음.** `manual_seed` / `np.random.seed` 호출이 학습 스크립트·데이터셋·모델 어디에도 없음 (grep 0건).
- **반복**: fold당 **1회**.

### 누수 가능성

**(a) EA의 transductive 성격 — 라벨 누수는 아니지만 프로토콜 제약.**
테스트 피험자의 화이트닝 행렬이 그 피험자의 **테스트 데이터 전체**로 추정됨 ([:867](../finetune_labram_hemi_aux.py#L867) → [:254-273](../data/seed_raw_dataset.py#L254-L273)). 라벨은 안 쓰지만 추론 시점에 그 피험자의 전체 녹화가 이미 있어야 함. 코드 자체가 명시 ([alignment.py:16-20](../data/alignment.py#L16-L20)). 현재 돌리는 캘리브레이션 스윕이 이 제약을 얼마나 줄일 수 있는지를 재는 실험.

**(b) 시드 부재 — 누수는 아니지만 결론을 위협.**
DataLoader 셔플, dropout, drop_path(0.1), head 초기화가 모두 통제 안 됨. fold당 1회 실행이므로 **작은 개선을 실행 간 잡음과 구분 불가**. diag EA의 이득이 +0.064인데 fold 표준편차가 0.068.

**(c) 창 겹침 — 현재 설정에서는 문제 없음.**
`step: 800 = segment_length` → 겹침 0. (기본값 `step=200`이면 75% 겹침이라 subject_dependent 프로토콜에서 문제가 되지만, LOSO는 피험자로 나누므로 무관.)

**(d) 검증 피험자가 fold마다 다름.** 순환 규칙이라 결정론적이지만 각 fold의 학습 데이터가 12명으로 다름. 의도된 설계 ([:852-854](../finetune_labram_hemi_aux.py#L852-L854) 주석).

---

## 5. Fine-tuning 방식

`configs/calib_diag.yaml`의 `use_lora: false` → **전체 fine-tuning**.

- 학습 파라미터: LaBraM 백본 **5,819,936개 전부** + main_head + asym_head ([:188-191](../finetune_labram_hemi_aux.py#L188-L191), 로그로 확인)
- 옵티마이저: AdamW + **layer-wise LR decay** — [build_layer_decay_optimizer:312-336](../finetune_labram_hemi_aux.py#L312-L336)
- 계층 id: 0 = patch_embed / cls_token / pos_embed / time_embed, 1–12 = 트랜스포머 블록, 13 = norm·head — [:295-309](../finetune_labram_hemi_aux.py#L295-L309)

| 하이퍼파라미터 | 값 |
|---|---|
| lr | 5e-4 (layer 13), 최저 1.85e-6 |
| layer_decay | 0.65 |
| weight_decay | 0.05 (1D 파라미터·bias는 0) |
| epochs / warmup | 50 / 5 |
| 스케줄 | 코사인, **스텝 단위** ([:339-346](../finetune_labram_hemi_aux.py#L339-L346), [:473](../finetune_labram_hemi_aux.py#L473)) |
| batch_size | 64 |
| clip_grad | 3.0 |
| label_smoothing | 0.1 |
| dropout | 0.0 |
| AMP | bf16 ([:349-358](../finetune_labram_hemi_aux.py#L349-L358)) |

---

## 6. 특징과 분류 head

[forward:224-270](../finetune_labram_hemi_aux.py#L224-L270)

```
all_tokens = labram.forward_features(eeg, input_chans, return_all_tokens=True)  :237
    (B, 1 + 62*4, 200)
├── cls_token   = all_tokens[:, 0]      (B, 200)       :243  ──► main_head ──► logits_main
└── patch_tokens= all_tokens[:, 1:]     (B, 248, 200)  :244
      reshape (B, 62, 4, 200)                          :250
      mean over patches → channel_feats (B, 62, 200)   :252
      ├─ LEFT_IDX (27ch)  mean → z_L (B,200)           :254
      └─ RIGHT_IDX(27ch)  mean → z_R (B,200)           :255
          concat [z_L; z_R; z_L−z_R; z_L⊙z_R] (B,800)  :258  ──► asym_head ──► logits_asym
```

- **특징 차원 200** (`embed_dim`, [:154](../finetune_labram_hemi_aux.py#L154))
- **main_head** ([:207-213](../finetune_labram_hemi_aux.py#L207-L213)): `LayerNorm(200) → Linear(200,200) → GELU → Dropout → Linear(200,3)`
- **asym_head** ([:216-222](../finetune_labram_hemi_aux.py#L216-L222)): `LayerNorm(800) → Linear(800,200) → GELU → Dropout → Linear(200,3)`
- 예측은 두 head의 앙상블 `logits_main + λ_asym·logits_asym` ([:373](../finetune_labram_hemi_aux.py#L373))인데 **현재 설정은 `lambda_asym: 0.0`**. 즉 asym_head는 CE 기울기를 전혀 못 받고 예측에도 0으로 곱해짐 — **지금 돌아가는 모델은 사실상 CLS 단일 head**.

### Prototype 계산에 가장 좋은 지점

**`out["z"]` = CLS 토큰** ([:266](../finetune_labram_hemi_aux.py#L266)).
이미 forward 반환 dict에 있어 코드 추가 없이 접근 가능하고, 분류 head가 실제로 보는 벡터와 동일. `main_head`의 첫 `LayerNorm` 전이라 스케일이 정규화되어 있지 않다는 점만 유의 — prototype 거리를 재려면 LayerNorm 후 또는 L2 정규화 후가 더 안정적.

대안으로 `channel_feats`(B,62,200)를 쓰면 채널별 prototype이 가능하지만 forward 반환값에 없어 [:252](../finetune_labram_hemi_aux.py#L252)에 한 줄 추가 필요.

진단용 특징 추출은 [probe_subject_invariance.py:127-168](../probe_subject_invariance.py#L127-L168) `extract()`가 이미 CLS / z_all / z_L / z_R를 전부 뽑아 캐시에 저장. 재사용이 가장 빠름.

---

## 7. 배치 구성

**피험자 ID: 배치에 이미 있음.** [seed_raw_dataset.py:240-247](../data/seed_raw_dataset.py#L240-L248)

```python
return {"eeg": ..., "label": ..., "subject": ..., "session": ...}
```

**clip은 없음.** `_seg_meta`에는 있지만 ([:152](../data/seed_raw_dataset.py#L152)) `__getitem__`이 내보내지 않음.

**샘플링: 균등 무작위 셔플뿐.** [:397-400](../finetune_labram_hemi_aux.py#L397-L400) `DataLoader(shuffle=True, drop_last=True)`. 피험자·클래스 층화 샘플러 없음.

배치 하나(64개)의 기대 구성:

- 피험자 12명 → 피험자당 평균 **5.3개**
- (피험자 × 클래스) 36칸 → 칸당 평균 **1.8개**

클래스는 거의 균등(각 5클립)이라 배치에 세 클래스가 다 들어올 확률은 높지만, **특정 피험자의 특정 클래스가 배치에서 통째로 빠지는 일은 흔함**.

### 피험자 ID를 loss까지 전달하려면

이미 배선되어 있음. E6(적대적 판별기) 경로를 그대로 따라가면 됨.

| 단계 | 위치 |
|---|---|
| 피험자 번호 → 0-based 인덱스 | [models/adversarial.py](../models/adversarial.py) `SubjectIndexer`, 생성 [:439](../finetune_labram_hemi_aux.py#L439) |
| 배치에서 꺼내기 | [:498-499](../finetune_labram_hemi_aux.py#L498-L499) `subj_index(batch["subject"])` |
| loss로 넘기기 | [:503-504](../finetune_labram_hemi_aux.py#L503-L504) `compute_loss(..., subject_idx=subj_idx)` |
| loss에서 받기 | [:272-273](../finetune_labram_hemi_aux.py#L272-L273) `compute_loss(self, out, label, criterion, subject_idx=None)` |

즉 **prototype loss를 위해 새로 배선할 것은 없음.** `lambda_adv`와 무관하게 `subj_index`가 만들어지도록 [:432](../finetune_labram_hemi_aux.py#L432)의 조건만 풀면 됨.

---

## 8. 학습 루프 구조

- **loss 계산 위치**: [compute_loss:272-281](../finetune_labram_hemi_aux.py#L272-L281)

```python
loss = loss_main + self.lambda_asym * loss_asym          # :276
if "logits_subj" in out and subject_idx is not None:     # :277
    loss = loss + criterion(out["logits_subj"], subject_idx)   # :280
```

- **호출**: [:501-504](../finetune_labram_hemi_aux.py#L501-L504), bf16 autocast 안
- **추가 loss 항을 넣기 가장 좋은 곳**: `compute_loss` 안 [:280](../finetune_labram_hemi_aux.py#L280) 바로 뒤. 여기서 `out["z"]`(CLS), `label`, `subject_idx`가 모두 손에 있고 이미 `subject_idx`를 받는 시그니처. 에폭 의존 가중치가 필요하면 `model.adv_lambda`와 같은 방식으로 [:489-493](../finetune_labram_hemi_aux.py#L489-L493) 패턴을 복사.
- **설정 시스템**: YAML 단일 파일, `yaml.safe_load` [:938](../finetune_labram_hemi_aux.py#L938). 블록은 `data:` / `labram:` / `model:` / `training:` / `finetune:`. `model:`과 `training:`은 지금 경로에서 쓰이지 않음(사전학습용 잔재). 새 하이퍼파라미터는 `finetune:` 아래에 넣고 `fc.get("이름", 기본값)`으로 읽는 것이 이 파일의 관례.

---

## 9. 현재 결과

지표: **window 단위 accuracy / macro-F1**, 최고 val-F1 에폭의 test 값. 우연 수준 0.333.

| 피험자 | EA 없음 | full EA | **diag EA** |
|---|---|---|---|
| 1 | 0.5241 | 0.6089 | 0.6006 |
| 2 | 0.6105 | 0.5269 | 0.5998 |
| 3 | 0.5934 | 0.7094 | 0.6671 |
| 4 | 0.5819 | 0.4755 | 0.5780 |
| 5 | 0.5669 | 0.5986 | 0.5780 |
| 6 | 0.4794 | 0.4968 | 0.4818 |
| 7 | 0.4727 | 0.6006 | 0.6405 |
| 8 | 0.6793 | 0.7375 | 0.6983 |
| 9 | 0.4854 | 0.6283 | 0.6789 |
| 10 | 0.5067 | 0.6045 | 0.5970 |
| 11 | 0.6370 | 0.6730 | 0.7031 |
| 12 | 0.4929 | 0.5693 | 0.5598 |
| 13 | 0.4691 | 0.6211 | 0.5946 |
| 14 | 0.5364 | 0.5625 | 0.5396 |
| 15 | 0.6508 | 0.6322 | 0.7245 |
| **평균** | **0.5524 ± 0.0701** | **0.6030 ± 0.0719** | ~~0.6161 ± 0.0679~~ |
| **평균 (시드 3개)** | (미측정) | (미측정) | **0.5881 ± 0.0799** |

> **2026-09-25 정정.** diag EA 열의 0.6161 은 **1회 실행**이며, 시드 3개로 다시 재면
> **0.5881 ± 0.0799** 다 (−0.028). EA 없음 대비 이득도 +0.064 → **+0.0357**
> (p=0.0145, 11/15) 로 줄어든다. 실행 잡음(시드 간 sd)은 window **0.0284**,
> clip **0.0634**, 최소 감지 효과는 **+0.0386**.
> EA 없음과 full EA 열도 1회 실행이라 같은 편향이 있을 수 있다 — 공정한 비교를 하려면
> 그쪽도 시드 3개로 다시 재야 한다. 근거는 [S0_REPORT.md](S0_REPORT.md) 의 A팔 절.
>
> **2026-09-27 재정정.** EA 없음을 시드 3개로 재측정했다
> ([S7_EA_X_CENTERING.md](S7_EA_X_CENTERING.md)). 이득은 위의 +0.0357 이 아니라
> clip **+0.0499** (p=0.0115, 13/15), window **+0.0407** (p=0.0215, 10/15) 로
> **커진다**. +0.0357 은 한쪽만 시드 3개였던 비대칭 비교였다.

diag EA는 full EA와 달리 **어느 fold도 3pp 이상 악화시키지 않음** (full EA는 S2 −0.084, S4 −0.106).

### 저장 위치

- 로그: `loso_baseline_gpu{2,3,3b}.log`, `loso_ea_gpu{0,3}.log`, [loso_ea_diag.log](../loso_ea_diag.log)
- 창 단위 logits + 메타(subject/session/clip): `logits/c0diag_S*.npz`, `logits/calibdiag_S*.npz` — [저장 코드 :604-616](../finetune_labram_hemi_aux.py#L604-L616)
- 집계 도구: [aggregate_logits.py](../aggregate_logits.py) (clip 단위, causal-Ns, balanced accuracy)
- 종합 메모: [NEXT.md](NEXT.md)
- 진행 중: [loso_calib_diag_8_15.log](../loso_calib_diag_8_15.log) (캘리브레이션 길이 스윕, fold 8–15)

---

## 클래스 정렬 loss / 특징 진단의 걸림돌

### 1. 배치가 prototype을 만들기엔 너무 얇음 — 가장 큰 문제

배치 64 ÷ 피험자 12 ÷ 클래스 3 = 칸당 1.8개. 층화 샘플러가 없고 ([:399](../finetune_labram_hemi_aux.py#L399)) 단순 셔플뿐이라 배치 내 피험자별 클래스 prototype은 극도로 불안정. 셋 중 하나가 필요:
(a) 피험자·클래스 층화 샘플러 신규 작성, (b) EMA / memory bank prototype, (c) 훨씬 큰 배치 (메모리 여유 있음 — 현재 6 GB / 49 GB).

### 2. 시드가 없어 효과를 검증할 수 없음

`manual_seed` 호출 0건, fold당 1회 실행. diag EA의 +0.064조차 fold 표준편차 0.068 안에 있음. prototype loss가 그보다 작은 이득을 내면 **현재 설정으로는 잡음과 구분 불가**. 새 loss를 평가하기 전에 시드 고정 또는 시드별 반복을 먼저 넣어야 함.

### 3. EA의 단위와 prototype의 단위가 어긋남

EA는 `(피험자, 세션)`별로 화이트닝 ([:251-252](../data/seed_raw_dataset.py#L251-L252), `ea_scope: session`). 같은 피험자라도 세션 3개가 **서로 다른 변환**을 거침. "피험자별 클래스 prototype"의 단위를 피험자로 잡을지 (피험자, 세션)으로 잡을지 먼저 정해야 하고, 피험자로 잡으면 이미 세션별로 정렬된 것을 다시 뭉개는 셈.

### 4. 유효 표본 수가 겉보기보다 훨씬 적음

피험자당 창 2,526개지만 한 클립 안의 ~56개 창은 같은 라벨에 강하게 상관된 사실상 중복. **(피험자, 클래스)당 실질 표본은 창 842개가 아니라 클립 15개.** prototype 분산 추정과 유의성 계산을 창 단위로 하면 표본 수를 50배 과대평가.

### 5. `clip`이 배치에 없음

[:241-248](../data/seed_raw_dataset.py#L241-L248)이 `subject`와 `session`만 내보냄. 클립 단위 prototype이나 클립 내 중복 제어를 하려면 `_seg_meta[idx][2]` 추가 필요 (한 줄).

### 6. 진단할 모델 상태를 꺼내기 어려움

`best_state`는 [`_train_and_eval` 안 :552](../finetune_labram_hemi_aux.py#L552)에만 존재하고 반환되지 않음 ([:639-645](../finetune_labram_hemi_aux.py#L639-L645)). `save_backbone`을 켜도 저장되는 건 [`model.labram`만](../finetune_labram_hemi_aux.py#L626)이고 **head는 빠짐**. "EA 이후 FM 특징 공간"을 fine-tuning된 상태에서 보려면 현재는 백본만 복원하고 head는 다시 학습해야 함. 사전학습 상태에서 보는 거라면 [probe_subject_invariance.py](../probe_subject_invariance.py)를 그대로 사용 가능.

### 7. asym_head가 죽어 있음

`lambda_asym: 0.0`이라 [:276](../finetune_labram_hemi_aux.py#L276)에서 CE 기울기가 0. prototype loss를 `z_fused`(z_L/z_R 기반)에 걸면 **CE를 전혀 받지 않는 벡터를 정렬시키게 됨.** CLS(`out["z"]`)에 거는 게 맞음.

### 8. 평가가 앙상블 logits

[:373](../finetune_labram_hemi_aux.py#L373)이 `logits_main + λ_asym·logits_asym`을 사용. 현재 λ=0이라 무해하지만, prototype loss 실험에서 λ를 건드리면 평가식도 같이 바뀜.

### 9. EA 적용이 워커에서 매 샘플 62×62 행렬곱

[:221](../data/seed_raw_dataset.py#L221) `W @ X`. diag 모드에서는 대각 행렬이므로 브로드캐스트 곱이면 충분. 속도 문제일 뿐 정확도와 무관하지만, 배치를 크게 키울 계획이면 병목이 될 수 있음.

### 10. 대역통과 필터 설정을 코드로 확인 불가

"EA 이후 특징 공간"을 논문에 쓰려면 원 신호의 필터 대역과 샘플링 레이트를 명시해야 하는데 현재 저장소에 근거 없음. SEED 배포 문서를 인용해야 함.

---

## 권고 순서

진단(prototype 어긋남 측정)부터 시작한다면 **2번(시드)과 4번(유효 표본)을 먼저 정리**할 것. 이 둘이 안 잡히면 진단 결과의 크기를 해석할 기준이 없음.
