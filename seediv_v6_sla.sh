#!/bin/bash
# V6 — SEED-IV: 회귀 기준 (analyze_calib_rotation) → SLA 스모크 (center 일치·유한성) → SLA 전체 (CPU).
# 스모크가 실패하면 전체를 돌리지 않는다.
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATASET=seediv
P=/home/kihyeonjoo/miniconda3/envs/bcifm/bin/python
echo "시작 $(date '+%m-%d %H:%M')"
$P analyze_calib_rotation.py --cache cache_ft_seediv_noea --out results/seediv_calib_rotation_noea.npz \
    > seediv_calib_rotation.log 2>&1 || { echo "회귀 기준 실패 $(date '+%m-%d %H:%M')"; exit 1; }
echo "회귀 기준 종료 $(date '+%m-%d %H:%M')"
$P analyze_sla.py --cache cache_ft_seediv_noea --smoke > seediv_sla_smoke.log 2>&1 \
    || { echo "스모크 실패 — 전체 실행 안 함 $(date '+%m-%d %H:%M')"; exit 1; }
echo "스모크 통과 $(date '+%m-%d %H:%M')"
$P analyze_sla.py --cache cache_ft_seediv_noea --out results/seediv_sla_noea.npz > seediv_sla.log 2>&1
echo "SLA 종료 (exit $?) $(date '+%m-%d %H:%M')"
