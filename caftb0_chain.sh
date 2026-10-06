#!/bin/bash
# CAFT ⓑ0 대조 연결 (2026-10-06 준비 — GPU 승인 뒤 실행): 학습 (SEED-V 16 fold, 시드 0) → 특징 캐시 → CPU 분석.
# ⓑ0 = ⓑ 와 같은 구조 배치, 배치 안 중심화만 끔 (configs/caftb0_seedv_noea.yaml).  결과 results/caftb0_seedv_noea_*.npz.
#   스모크 (먼저, 약 1분):  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=G python finetune_labram_hemi_aux.py \
#                             --config configs/smoke_caftb0.yaml --eval_mode loso --folds 1
#   본 실행:               setsid nohup bash caftb0_chain.sh G > caftb0_chain.log 2>&1 &
G=${1:?GPU 번호}
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/home/kihyeonjoo/miniconda3/envs/bcifm/bin/python
echo "$(date +%m-%d_%H:%M) CAFT ⓑ0 시드 0 학습 시작 (GPU$G)"
CUDA_VISIBLE_DEVICES=$G $P finetune_labram_hemi_aux.py --config configs/caftb0_seedv_noea.yaml --eval_mode loso \
    > caftb0_seedv.log 2>&1; rc=$?
echo "$(date +%m-%d_%H:%M) 학습 종료 (exit $rc) — 체크포인트 $(ls checkpoints_s0/caftb0_seedv_noea_S*_seed0.pt 2>/dev/null | wc -l)"
CUDA_VISIBLE_DEVICES=$G DATASET=seedv $P cache_ft_features.py --no_ea --ckpt_glob "checkpoints_s0/caftb0_seedv_noea_S*_seed0.pt" \
    --out_dir cache_ft_seedv_caftb0_noea > cache_ft_seedv_caftb0_noea.log 2>&1; rc=$?
echo "$(date +%m-%d_%H:%M) 캐시 종료 (exit $rc) — 파일 $(ls cache_ft_seedv_caftb0_noea 2>/dev/null | wc -l)개"
bash caft_analyze.sh caftb0_seedv_noea cache_ft_seedv_caftb0_noea caftb0_seedv_noea_ > caft_analyze_b0.log 2>&1
echo "$(date +%m-%d_%H:%M) 분석 종료 — $(tail -1 caft_analyze_b0.log)"
