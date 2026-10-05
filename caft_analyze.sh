#!/bin/bash
# CAFT 비교 분석 (CPU) — 한 팔의 특징 캐시로: 현실적 캘리브레이션 r10 → 회귀 기준 (calib_rotation) → 자극 시점 정렬 스모크 → 전체 → 어긋남 c.
# 인자는 SEED-IV V6 실행 (seediv_v6_center.sh · seediv_v6_sla.sh) 과 같다.  회귀 기준은 같은 캐시로 만들어 --v4 로 넘긴다
# (기본 자동 경로는 시드 3개 캐시의 결과라 시드 0 하나뿐인 캐시와 피험자별 값이 다르다).
#   setsid nohup bash caft_analyze.sh seedv_noea_seed0 cache_ft_seedv_noea_seed0 seedv_noea_ > caft_analyze_base.log 2>&1 &
#   (CAFT: caftb_seedv_noea cache_ft_seedv_caftb_noea caftb_seedv_noea_ / caftc_... 같은 꼴)
TAG=${1:?태그}; CACHE=${2:?캐시}; PFX=${3:?체크포인트 접두어}
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DATASET=seedv
P=/home/kihyeonjoo/miniconda3/envs/bcifm/bin/python
N=$(ls $CACHE/S*_seed*.npz 2>/dev/null | wc -l)
[ "$N" -eq 16 ] || { echo "[$TAG] 캐시 $N/16 — 중단 $(date '+%m-%d %H:%M')"; exit 1; }
echo "[$TAG] 시작 (캐시 $N개) $(date '+%m-%d %H:%M')"
$P analyze_calib_protocol.py --cache $CACHE --prefix $PFX --seconds 20 40 --n_rep 10 \
    --out results/${TAG}_calib_protocol_r10.npz > ${TAG}_calib_protocol.log 2>&1 \
    || { echo "[$TAG] 캘리브레이션 실패 $(date '+%m-%d %H:%M')"; exit 1; }
echo "[$TAG] 캘리브레이션 종료 $(date '+%m-%d %H:%M')"
$P analyze_calib_rotation.py --cache $CACHE --out results/${TAG}_calib_rotation.npz > ${TAG}_calib_rotation.log 2>&1 \
    || { echo "[$TAG] 회귀 기준 실패 $(date '+%m-%d %H:%M')"; exit 1; }
echo "[$TAG] 회귀 기준 종료 $(date '+%m-%d %H:%M')"
$P analyze_sla.py --cache $CACHE --v4 results/${TAG}_calib_rotation.npz --smoke > ${TAG}_sla_smoke.log 2>&1 \
    || { echo "[$TAG] 스모크 실패 — 전체 실행 안 함 $(date '+%m-%d %H:%M')"; exit 1; }
grep -q "불일치" ${TAG}_sla_smoke.log && { echo "[$TAG] 스모크 회귀 불일치 — 전체 실행 안 함"; exit 1; }
echo "[$TAG] 스모크 통과 $(date '+%m-%d %H:%M')"
$P analyze_sla.py --cache $CACHE --v4 results/${TAG}_calib_rotation.npz --out results/${TAG}_sla.npz > ${TAG}_sla.log 2>&1 \
    || { echo "[$TAG] SLA 실패 $(date '+%m-%d %H:%M')"; exit 1; }
echo "[$TAG] SLA 종료 $(date '+%m-%d %H:%M')"
$P analyze_misalignment.py --cache $CACHE --out results/${TAG}_misalignment.npz > ${TAG}_misalignment.log 2>&1
echo "[$TAG] 어긋남 종료 (exit $?) $(date '+%m-%d %H:%M')"
echo "[$TAG] 분석 종료 $(date '+%m-%d %H:%M')"
