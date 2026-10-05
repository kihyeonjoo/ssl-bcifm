#!/bin/bash
# DEAP no-EA LOSO 학습 — GPU3 (nvidia-smi 번호, PCI 순서 고정), 시험 피험자 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 시드 0 (2026-10-05 사용자 승인: GPU0 + GPU3, 시드 1개).
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/kihyeonjoo/miniconda3/envs/bcifm/bin/python finetune_labram_hemi_aux.py --config configs/deap_noea.yaml --eval_mode loso --folds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
