#!/bin/bash
# 두 GPU 연결 (gpu3_chain PID 2143082, gpu0_chain PID 2143089) 이 모두 끝나면 DEAP 체크포인트 32개를 확인하고
# GPU3 에서만 DEAP 특징 캐시를 만든다.  그 뒤 GPU0 · GPU3 모두 반납 (더 띄우지 않음).
cd /home/kihyeonjoo/ssl-bcifm || exit 1
while kill -0 2143082 2>/dev/null || kill -0 2143089 2>/dev/null; do sleep 60; done
nck=$(ls checkpoints_s0/deap_noea_S*_seed0.pt 2>/dev/null | wc -l)
echo "$(date +%m-%d_%H:%M) 두 연결 종료 — DEAP 체크포인트 ${nck}"
if [ "$nck" -ne 32 ]; then
  echo "체크포인트가 32개가 아니다 — 캐시를 만들지 않고 멈춘다 (사람이 확인할 것)"; exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DATASET=deap /home/kihyeonjoo/miniconda3/envs/bcifm/bin/python cache_ft_features.py --no_ea > cache_ft_deap_noea.log 2>&1
echo "$(date +%m-%d_%H:%M) DEAP 캐시 종료 (exit $?) — 파일 $(ls cache_ft_deap_noea 2>/dev/null | wc -l)개"
