#!/bin/bash
# CAFT ⓑ 시드 N 연결 (2026-10-06 사용자 승인): [선택: GPU 가 빌 때까지 대기] → 학습 (SEED-V 16 fold) → 특징 캐시 → CPU 분석.
# 기준선 ⓐ 는 시드 0·1·2 가 이미 있으므로 ⓑ 시드 1·2 만 더하면 시드 3개 짝 비교가 된다.
#   setsid nohup bash caftb_seed_chain.sh 1 0 > caftb_s1_chain.log 2>&1 &            (시드 1, GPU0, 바로 시작)
#   WAIT_FREE=1 setsid nohup bash caftb_seed_chain.sh 2 3 > caftb_s2_chain.log 2>&1 & (시드 2, GPU3, 다른 사용자 작업이 끝나면 시작)
S=${1:?시드}; G=${2:?GPU 번호}
cd /home/kihyeonjoo/ssl-bcifm || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/home/kihyeonjoo/miniconda3/envs/bcifm/bin/python
if [ -n "$WAIT_FREE" ]; then
    echo "$(date +%m-%d_%H:%M) GPU$G 에 다른 프로세스가 없어질 때까지 대기"
    until [ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do sleep 300; done
fi
echo "$(date +%m-%d_%H:%M) CAFT ⓑ 시드 $S 학습 시작 (GPU$G)"
CUDA_VISIBLE_DEVICES=$G $P finetune_labram_hemi_aux.py --config configs/caftb_seedv_noea_s$S.yaml --eval_mode loso \
    > caftb_seedv_s$S.log 2>&1; rc=$?
echo "$(date +%m-%d_%H:%M) 학습 종료 (exit $rc) — 체크포인트 $(ls checkpoints_s0/caftb_seedv_noea_S*_seed$S.pt 2>/dev/null | wc -l)"
CUDA_VISIBLE_DEVICES=$G DATASET=seedv $P cache_ft_features.py --no_ea --ckpt_glob "checkpoints_s0/caftb_seedv_noea_S*_seed$S.pt" \
    --out_dir cache_ft_seedv_caftb_noea_s$S > cache_ft_seedv_caftb_noea_s$S.log 2>&1; rc=$?
echo "$(date +%m-%d_%H:%M) 캐시 종료 (exit $rc) — 파일 $(ls cache_ft_seedv_caftb_noea_s$S 2>/dev/null | wc -l)개"
bash caft_analyze.sh caftb_seedv_noea_s$S cache_ft_seedv_caftb_noea_s$S caftb_seedv_noea_ > caft_analyze_b_s$S.log 2>&1
echo "$(date +%m-%d_%H:%M) 분석 종료 — $(tail -1 caft_analyze_b_s$S.log)"
