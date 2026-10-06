#!/bin/bash
# 교환 캘리브레이션 대조 (2026-10-06): ⓐ 시드 0·1·2, ⓑ 시드 0·1·2, ⓒ 시드 0 — CPU, 팔마다 병렬.
#   bash calib_swap_all.sh > calib_swap_all.log 2>&1
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATASET=seedv
P=/home/kihyeonjoo/miniconda3/envs/bcifm/bin/python
run() {   # 캐시 폴더, 결과 태그
    $P analyze_calib_swap.py --cache $1 --ref results/$2_sla.npz --out results/$2_calib_swap.npz > calib_swap_$2.log 2>&1
    echo "$2 종료 (exit $?) — 회귀 $(grep -c '최대 차이 0.00e+00' calib_swap_$2.log)/6"
}
for s in 0 1 2; do run cache_ft_seedv_noea_seed$s seedv_noea_seed$s & done
run cache_ft_seedv_caftb_noea caftb_seedv_noea &
run cache_ft_seedv_caftb_noea_s1 caftb_seedv_noea_s1 &
run cache_ft_seedv_caftb_noea_s2 caftb_seedv_noea_s2 &
run cache_ft_seedv_caftc_noea caftc_seedv_noea &
wait
echo "끝 $(date '+%H:%M:%S')"
