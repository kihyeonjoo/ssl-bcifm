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

#### Auxiliary Pretext Task — Temporal Delta Asymmetry Prediction *(예정)*

좌우 반구 간 band power 시간 차분(Δ)의 비대칭 패턴 예측.

#### Downstream Task — Emotion Classification

Pretrained encoder E_φ의 좌우 CLS embedding을 concat한 `z_joint (B, 2d)`에 분류 헤드 부착 후 fine-tuning.

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
│   └── seed.yaml               # 하이퍼파라미터
├── data/
│   ├── preprocessing.py        # 채널 분리 + BandSTFT (STFT → 4-band features)
│   └── seed_dataset.py         # SEEDDataset (sliding window, hemisphere split)
├── models/
│   └── encoder.py              # SpectroSpatialCNN + TemporalTransformer
│                               # HemisphereEncoder + DualStreamEncoder (E_φ)
├── tasks/
│   └── cross_hemisphere.py     # BandSelectiveMasking + CrossHemisphereDecoder
│                               # CrossHemisphereMaskedPrediction (ℒ_main)
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

## Requirements

```bash
pip install -r requirements.txt
```

```
torch>=1.9.0
torchaudio>=0.9.0
numpy>=1.21
scipy>=1.7
pyyaml>=6.0
```

---

## Quick Start

```python
from data.seed_dataset import SEEDDataset
from models.encoder import DualStreamEncoder
from tasks.cross_hemisphere import CrossHemisphereMaskedPrediction

# Dataset
dataset = SEEDDataset(root="/path/to/SEED", segment_length=800, step=200)
# item['left'], item['right']: (27, 4, 16)  /  item['label']: scalar

# Encoder (shared weights)
encoder = DualStreamEncoder(
    n_channels_per_hemi=27, n_bands=4, d_model=256,
    cnn_channels=(32, 64, 128), n_heads=8, n_layers=4,
)

# Pretext task
task = CrossHemisphereMaskedPrediction(encoder=encoder, d_model=256)

# Forward
out = task(batch["left"], batch["right"])
loss     = out["loss"]       # ℒ_main (MSE)
z_left   = out["z_left"]    # (B, 256)
z_right  = out["z_right"]   # (B, 256)
```

---

## Configuration

[configs/seed.yaml](configs/seed.yaml) 에서 모든 하이퍼파라미터 관리:

```yaml
data:
  segment_length: 800   # 4 s window @ 200 Hz
  step: 200             # 1 s stride  (75 % overlap)
  n_fft: 200            # 1 Hz/bin frequency resolution
  hop_length: 40        # 0.2 s STFT hop

model:
  d_model: 256
  cnn_channels: [32, 64, 128]
  n_heads: 8
  n_layers: 4

training:
  batch_size: 64
  lr: 1.0e-3
  epochs: 100
```

---

## Results

*(실험 완료 후 채울 것)*

---

## Citation

*(논문 출판 후 채울 것)*
