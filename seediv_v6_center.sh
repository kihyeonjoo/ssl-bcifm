#!/bin/bash
# V6 추가 사전 등록 — SEED-IV 중심화 재현 (CPU).  인자는 V6 문서 15:02 고정분 그대로.
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATASET=seediv
P=/home/kihyeonjoo/miniconda3/envs/bcifm/bin/python
echo "시작 $(date '+%m-%d %H:%M')"
$P analyze_calib_protocol.py --cache cache_ft_seediv_noea --prefix seediv_noea_ --seconds 20 40 --n_rep 10 \
    --out results/seediv_calib_protocol_noea_r10.npz
echo "중심화 재현 종료 (exit $?) $(date '+%m-%d %H:%M')"
