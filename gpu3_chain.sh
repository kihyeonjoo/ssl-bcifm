#!/bin/bash
# GPU3 순서 (2026-10-05 사용자 결정 "Ada 두 장"): CAFT ⓑ 학습 (SEED-V 16 fold, 시드 0) → 그 특징 캐시 → DEAP 남은 fold 학습.
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/home/kihyeonjoo/miniconda3/envs/bcifm/bin/python
echo "$(date +%m-%d_%H:%M) CAFT ⓑ 학습 시작 (GPU3)"
$P finetune_labram_hemi_aux.py --config configs/caftb_seedv_noea.yaml --eval_mode loso > caftb_seedv.log 2>&1; rc=$?
echo "$(date +%m-%d_%H:%M) CAFT ⓑ 학습 종료 (exit $rc) — 체크포인트 $(ls checkpoints_s0/caftb_seedv_noea_S*_seed0.pt 2>/dev/null | wc -l)"
DATASET=seedv $P cache_ft_features.py --no_ea --ckpt_glob "checkpoints_s0/caftb_seedv_noea_S*_seed0.pt" --out_dir cache_ft_seedv_caftb_noea > cache_ft_seedv_caftb_noea.log 2>&1; rc=$?
echo "$(date +%m-%d_%H:%M) CAFT ⓑ 캐시 종료 (exit $rc) — 파일 $(ls cache_ft_seedv_caftb_noea 2>/dev/null | wc -l)개"
$P finetune_labram_hemi_aux.py --config configs/deap_noea_resume_gpu3.yaml --eval_mode loso --folds 4,5,6,7,8,9,10,11,12,13,14,15,16 > deap_noea_resume_gpu3.log 2>&1; rc=$?
echo "$(date +%m-%d_%H:%M) DEAP 이어 돌리기 종료 (exit $rc)"
