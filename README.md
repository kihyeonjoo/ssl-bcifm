# ssl-bcifm

**Brain-Hemisphere Asymmetry-Based Self-Supervised Learning for EEG Emotion Recognition**

---

## Overview

EEG 기반 감정 인식 연구에서 레이블된 데이터 부족 문제를 해결하기 위해, 뇌 반구 비대칭(Brain Hemisphere Asymmetry)을 inductive bias로 활용하는 self-supervised pretraining 프레임워크.

기존 EEG Foundation Model 연구들은 대규모 레이블 데이터를 전제하지만, 실제 환경에서는 충분한 EEG 데이터를 확보하기 어렵다. 본 연구는 좌·우 반구 간의 생리학적 비대칭 구조를 pretext task의 핵심 신호로 활용하여, 레이블 없이 의미 있는 EEG representation을 학습한다.

---

## Method

### Pretraining Tasks

```
EEG (62ch)
    │
    ├─ Channel split ──► Left hemisphere  (27ch, midline 제거)
    │                    Right hemisphere (27ch, midline 제거)
    │
    └─ STFT Band Extraction ──► (C_h, 4, T)  per hemisphere
                                  [θ | α | β | γ]
```

#### Main Pretext Task — Cross-Hemisphere Masked Band Prediction

1. **Band-selective masking**: 매 iteration마다 한쪽 반구와 주파수 대역(θ/α/β/γ) 하나를 랜덤 선택. **Gaussian-smoothed soft mask**를 적용하여 band 경계에서 스펙트럴 누수(spectral leakage) 방지.

   ```
   mask_weight[b] = 1 − exp(−(b − b_masked)² / 2σ²),  σ = 0.7
   → b_masked: 0 %,  ±1 band: 36 %,  ±2 bands: 98 % 보존
   ```

2. **Cross-hemisphere reconstruction**: 마스킹된 반구의 시퀀스 z̃ₛₑq를 Query, intact 반구의 시퀀스 zₛₑq를 Key/Value로 하는 Cross-attention 디코더 D_ψ가 마스킹된 대역 복원.

3. **ℒ_main** = MSE(복원값, 원본 masked band), 마스킹된 대역 인덱스에서만 계산.

#### Auxiliary Pretext Task — Temporal Delta Asymmetry Prediction

같은 trial 내 두 시간 구간 (t₁, t₂)을 샘플링하여 비대칭 변화량 Δa를 예측.

```
a(t) = z_L(t) − z_R(t)           ← asymmetry vector
Δa   = a(t₂) − a(t₁)            ← temporal delta (ground truth)
Δâ   = P_θ([z_L(t₁); z_R(t₁)])  ← predictor output

ℒ_aux = MSE(Δa, Δâ) + λ_cos · (1 − cos_sim(Δa, Δâ))
```

λ_aux는 linear warm-up: 0.1 → 0.5 (20 epoch).

#### Downstream Task — Emotion Classification

Pretrained encoder E_φ의 좌우 CLS embedding을 asymmetry-aware fusion 후 분류 헤드 부착.

```
z = [ z_L ; z_R ; z_L − z_R ; z_L ⊙ z_R ]  ∈ R^{4·d_model}
```

Leave-One-Subject-Out (LOSO) cross-validation으로 평가.

---

### Architecture

```
Input: (B, C_h=27, 4, T=16) per hemisphere
           │
    ┌──────┴───────┐
    │ SpectroSpatial│  Conv2d(1→32, kernel=(1,4))  ← spectral fusion
    │     CNN       │  Conv2d(32→64, (3,1))
    │               │  Conv2d(64→128, (3,1))
    │               │  Conv2d(128→256, (3,1))
    │               │  AdaptiveAvgPool → (B, T, 256)
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │   Temporal    │  [CLS] + SinusoidalPE
    │  Transformer  │  4× Pre-LN TransformerEncoderLayer
    │               │  → CLS token (B, 256)
    └──────┬───────┘
           │
    z_left / z_right  (B, 256)      ← shared weights E_φ
    z_joint = cat     (B, 512)      ← downstream input
```

**Weight sharing**: `DualStreamEncoder`는 `HemisphereEncoder` 단일 인스턴스를 좌·우에 두 번 호출 — 파라미터 복제 없음.

---

## Project Structure

```
ssl-bcifm/
├── configs/
│   └── seed.yaml                  # 하이퍼파라미터 (pretrain + finetune)
├── data/
│   ├── preprocessing.py           # 채널 분리 + BandSTFT (STFT → 4-band log-power)
│   ├── seed_dataset.py            # SEEDDataset (sliding window, hemisphere split)
│   └── temporal_pair_dataset.py   # TemporalPairDataset (같은 trial 내 t₁,t₂ 쌍)
├── models/
│   ├── encoder.py                 # SpectroSpatialCNN + TemporalTransformer
│   │                              # HemisphereEncoder + DualStreamEncoder (E_φ)
│   └── classifier.py             # AsymmetryFusionClassifier (downstream head)
├── tasks/
│   ├── cross_hemisphere.py        # CrossHemisphereMaskedPrediction (ℒ_main)
│   └── temporal_delta_asymmetry.py # TemporalDeltaAsymmetry (ℒ_aux)
├── pretrain.py                    # Pretraining loop (ℒ_main + λ·ℒ_aux)
├── finetune.py                    # LOSO cross-validation fine-tuning
├── utils/
├── requirements.txt
└── CLAUDE.md
```

---

## Dataset

**SEED** (SJTU Emotion EEG Dataset)

| 항목 | 내용 |
|---|---|
| 출처 | BCMI Lab, Shanghai Jiao Tong University |
| 피험자 | 15명 × 3 세션 |
| 채널 | 62ch (10-20 system), 200 Hz |
| 자극 | 15개 영화 클립 (긍정/중립/부정) |
| 레이블 | +1 / 0 / −1 → 0 / 1 / 2 (remapped) |

채널 분할 (midline 8ch 제거):

```
Left  (27ch): FP1, AF3, F7, F5, F3, F1, FT7, FC5, FC3, FC1, T7, C5, C3, C1,
              TP7, CP5, CP3, CP1, P7, P5, P3, P1, PO7, PO5, PO3, CB1, O1
Right (27ch): 좌우 대칭 전극 (FP2, AF4, …, CB2, O2)
Midline (8ch, discarded): FPZ, FZ, FCZ, CZ, CPZ, PZ, POZ, OZ
```

---

## Data Pipeline

```
.mat file (62, T) @ 200 Hz
    │
    ├─ Sliding window (800 samples = 4 s, step 200 = 1 s)
    │
    ├─ Hemisphere split → left (27, L),  right (27, L)
    │
    └─ BandSTFT  n_fft=200, hop=40
         → (27, 4, 16) per hemisphere
              │
              [0] θ  4 – 8  Hz
              [1] α  8 – 13 Hz
              [2] β  13 – 30 Hz
              [3] γ  30 – 45 Hz
```

---

## Setup

### 1. 가상환경 생성

```bash
conda create -n bcifm python=3.11 -y
conda activate bcifm
pip install -r requirements.txt
```

### 2. 데이터 경로 설정

[configs/seed.yaml](configs/seed.yaml)의 `data.root`를 SEED 데이터셋 경로로 수정:

```yaml
data:
  root: /mnt/data/original/SEED   # Preprocessed_EEG/ 폴더가 있는 경로
```

SEED 데이터 구조:
```
SEED/
  Preprocessed_EEG/
    1_20131027.mat    # Subject 1, Session 1
    1_20131030.mat    # Subject 1, Session 2
    ...
    15_20131105.mat   # Subject 15, Session 3
```

---

## How to Run

### Step 1. Pretraining

```bash
# 기본 실행
python pretrain.py --config configs/seed.yaml

# wandb 로깅 포함
python pretrain.py --config configs/seed.yaml --wandb_project ssl-bcifm

# 백그라운드 실행 (터미널 닫아도 유지)
nohup python -u pretrain.py --config configs/seed.yaml --wandb_project ssl-bcifm > pretrain.log 2>&1 &

# 로그 실시간 확인
tail -f pretrain.log
```

Checkpoint는 `checkpoints/` 폴더에 매 10 epoch마다 저장.

### Step 2. Fine-tuning (LOSO-CV)

`configs/seed.yaml`의 `finetune.pretrained_ckpt`에 사용할 checkpoint 경로 지정:

```yaml
finetune:
  pretrained_ckpt: checkpoints/pretrain_epoch100.pt
```

```bash
# 기본 실행
python finetune.py --config configs/seed.yaml

# wandb 로깅 포함
python finetune.py --config configs/seed.yaml --wandb_project ssl-bcifm-ft

# 백그라운드 실행
nohup python -u finetune.py --config configs/seed.yaml --wandb_project ssl-bcifm-ft > finetune.log 2>&1 &
```

15명 LOSO-CV 완료 후 accuracy / macro-F1 (mean ± std) 출력.

### 프로세스 관리

```bash
# 실행 중인 프로세스 확인
pgrep -af "pretrain.py\|finetune.py"

# 프로세스 종료
kill <PID>
```

### 빠른 테스트 (subject 일부만)

```yaml
data:
  subjects: [1, 2, 3]    # 3명만 로드
training:
  epochs: 5              # 적은 epoch으로 테스트
```

---

## Configuration

[configs/seed.yaml](configs/seed.yaml)에서 모든 하이퍼파라미터 관리:

```yaml
data:
  root: /mnt/data/original/SEED
  segment_length: 800       # 4 s window @ 200 Hz
  step: 200                 # 1 s stride (75 % overlap)
  n_fft: 200                # 1 Hz/bin frequency resolution
  hop_length: 40            # 0.2 s STFT hop

model:
  d_model: 256
  cnn_channels: [32, 64, 128]
  n_heads: 8
  n_layers: 4

training:                   # pretraining
  batch_size: 64
  lr: 1.0e-3
  epochs: 100
  lambda_aux: 0.5           # ℒ_aux final weight
  lambda_aux_init: 0.1      # ℒ_aux warm-up start

finetune:                   # downstream LOSO-CV
  pretrained_ckpt: checkpoints/pretrain_epoch100.pt
  lr: 5.0e-4
  encoder_lr: 1.0e-5        # pretrained encoder용 낮은 LR
  epochs: 30
  freeze_encoder: false
```

---

## Results

*(실험 완료 후 채울 것)*

---

## Citation

*(논문 출판 후 채울 것)*
