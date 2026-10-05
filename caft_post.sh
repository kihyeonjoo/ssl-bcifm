#!/bin/bash
# CAFT 두 팔의 학습 · 특징 저장이 끝나면 (gpu3_chain.log · gpu0_chain.log 의 '캐시 종료') 분석을 CPU 로 잇는다 — 2026-10-05.
# 분석 인자는 기준선 ⓐ 시드 0 과 같다 (caft_analyze.sh).  캐시가 16개가 아니면 caft_analyze.sh 가 스스로 멈춘다.
#   setsid nohup bash caft_post.sh > caft_post.log 2>&1 &
cd /home/kihyeonjoo/ssl-bcifm || exit 1
echo "대기 시작 $(date '+%m-%d %H:%M')"
until grep -q "캐시 종료" gpu3_chain.log 2>/dev/null && grep -q "캐시 종료" gpu0_chain.log 2>/dev/null; do sleep 120; done
echo "캐시 확인 $(date '+%m-%d %H:%M')"; grep "캐시 종료" gpu3_chain.log gpu0_chain.log
bash caft_analyze.sh caftb_seedv_noea cache_ft_seedv_caftb_noea caftb_seedv_noea_ > caft_analyze_b.log 2>&1 &
bash caft_analyze.sh caftc_seedv_noea cache_ft_seedv_caftc_noea caftc_seedv_noea_ > caft_analyze_c.log 2>&1 &
wait
tail -1 caft_analyze_b.log; tail -1 caft_analyze_c.log
echo "분석 종료 $(date '+%m-%d %H:%M')"
