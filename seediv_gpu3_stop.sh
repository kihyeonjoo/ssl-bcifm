#!/bin/bash
# GPU3 의 SEED-IV 학습(PID 820708)을 피험자 8 · 시드 2 까지만 하고 멈춘다 — 피험자 9~15 는 GPU0 가 맡는다.
# CSV 행은 run 이 끝나 체크포인트를 저장한 뒤에 쓰이므로, 그 행이 보이면 끊어도 잃는 것이 없다.
cd /home/kihyeonjoo/ssl-bcifm
while kill -0 820708 2>/dev/null; do
  if [ -f results/seediv_noea.csv ] && python3 -c '
import csv, sys
rows = list(csv.DictReader(open("results/seediv_noea.csv")))
sys.exit(0 if any(r["subject"] == "8" and r["seed"] == "2" for r in rows) else 1)'; then
    kill 820708
    echo "$(date +%m-%d_%H:%M) S8 seed2 완료 확인 → GPU3 학습 종료 (PID 820708)"
    break
  fi
  sleep 20
done
