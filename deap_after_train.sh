#!/bin/bash
# DEAP: 두 GPU 학습 (GPU3 PID 2118541 = 시험 피험자 1~16, GPU0 PID 2118546 = 17~32) 이 모두 끝나면
# 체크포인트 32개를 확인하고 GPU3 에서만 특징 캐시를 만든다.  GPU0 은 학습이 끝나는 즉시 반납된다 (여기서 쓰지 않음).
cd /home/kihyeonjoo/ssl-bcifm || exit 1
while kill -0 2118541 2>/dev/null || kill -0 2118546 2>/dev/null; do sleep 60; done
n1=$(python3 -c 'import csv; print(len(list(csv.DictReader(open("results/deap_noea.csv")))))' 2>/dev/null)
n2=$(python3 -c 'import csv; print(len(list(csv.DictReader(open("results/deap_noea_part2.csv")))))' 2>/dev/null)
nck=$(ls checkpoints_s0/deap_noea_S*_seed0.pt 2>/dev/null | wc -l)
echo "$(date +%m-%d_%H:%M) 학습 종료 — CSV 행 GPU3 ${n1} / GPU0 ${n2}, 체크포인트 ${nck}"
if [ "$nck" -ne 32 ]; then
  echo "체크포인트가 32개가 아니다 — 캐시를 만들지 않고 멈춘다 (사람이 확인할 것)"; exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DATASET=deap /home/kihyeonjoo/miniconda3/envs/bcifm/bin/python cache_ft_features.py --no_ea > cache_ft_deap_noea.log 2>&1
echo "$(date +%m-%d_%H:%M) 캐시 종료 (exit $?) — 파일 $(ls cache_ft_deap_noea 2>/dev/null | wc -l)개"
