#!/bin/bash
# DEAP no-EA LOSO 학습 — GPU0 (nvidia-smi 번호, PCI 순서 고정), 시험 피험자 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32, 시드 0 (2026-10-05 사용자 승인: GPU0 + GPU3, 시드 1개).
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/kihyeonjoo/miniconda3/envs/bcifm/bin/python finetune_labram_hemi_aux.py --config configs/deap_noea_gpu0.yaml --eval_mode loso --folds 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32
