# ssl-bcifm

## Overview
This project investigates ... (연구 주제 한 줄 요약)

## Motivation
1. 지금까지는 EEG-Foundation Model을 구축하여 downstream task의 성능을 향상시키는 연구들이 진행되어 왔다.
2. 그러나 실생활에서는 충분한 EEG 데이터를 확보하기 어려우며, 연구적으로 공개된 데이터셋 역시 한계가 존재한다.
3. 따라서 개인화된 소규모 데이터셋만으로도 Foundation Model을 구축하고 downstream task의 성능을 향상시키는 것이 중요하다.
4. 이를 위해 Self-Supervised Learning을 활용하여 pretext task에 대한 사전학습을 수행하고, 새로운 데이터로 fine-tuning하여 downstream task의 성능을 향상시키고자 한다.

## Method
Pretext Task -> Pretraining
Pretrained Model -> Downstream Task -> Finetuning



(사용하는 방법론, 알고리즘, 모델 등)

## Dataset
(사용하는 데이터셋 이름 및 출처)

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py --config configs/default.yaml
```

## Results
(실험 결과 표나 수치 — 나중에 채울 것)

## Citation
```
@article{...}
```
