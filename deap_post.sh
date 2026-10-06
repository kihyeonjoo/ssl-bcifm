#!/bin/bash
# DEAP 특징 저장 (deap_final.sh 의 'DEAP 캐시 종료') 이 끝나면 분석을 CPU 로 잇는다 — 2026-10-06.
# analyze_deap.py (현실적 캘리브레이션, 4사분면 + 이진 V/A) → analyze_misalignment.py (어긋남 c) → summarize_deap.py.
# 2 fold 시험 실행 (--max_files 2) 통과 확인 뒤 건다.
#   setsid nohup bash deap_post.sh > deap_post.log 2>&1 &
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATASET=deap
P=/home/kihyeonjoo/miniconda3/envs/bcifm/bin/python
echo "대기 시작 $(date '+%m-%d %H:%M')"
until grep -q "DEAP 캐시 종료" deap_final.log 2>/dev/null; do sleep 60; done
grep "DEAP 캐시 종료" deap_final.log
N=$(ls cache_ft_deap_noea/S*_seed*.npz 2>/dev/null | wc -l)
[ "$N" -eq 32 ] || { echo "캐시 $N/32 — 분석 안 함 $(date '+%m-%d %H:%M')"; exit 1; }
$P analyze_deap.py --cache cache_ft_deap_noea --out results/deap_calib_noea.npz > deap_analyze.log 2>&1
echo "분석 종료 (exit $?) $(date '+%m-%d %H:%M')"
$P analyze_misalignment.py --cache cache_ft_deap_noea --out results/deap_misalignment_noea.npz > deap_misalignment.log 2>&1
echo "어긋남 종료 (exit $?) $(date '+%m-%d %H:%M')"
$P summarize_deap.py --res results/deap_calib_noea.npz --mis results/deap_misalignment_noea.npz > deap_summary.log 2>&1
echo "요약 종료 (exit $?) $(date '+%m-%d %H:%M')"
echo "끝 $(date '+%m-%d %H:%M')"
