"""subject-dependent (SD) 결과 요약 — 논문 표 1 (위치 잡기) 의 SD 행.

입력: results/a02_sd.csv (SEED), results/seedv_a02_sd.csv (SEED-V).
한 행 = (피험자, 시드) 한 run.  지표는 run 안에서 분할의 결정을 모아 한 번 계산한
값이다 (finetune_labram_hemi_aux.py 의 SD 분기).  여기서는 피험자마다 시드 평균을
낸 뒤 피험자 사이 평균 ± 표준편차를 보고한다 (LOSO 표와 같은 방식).

    python summarize_sd.py
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

SETS = (("SEED", "results/a02_sd.csv", 15, 3, 30),
        ("SEED-V", "results/seedv_a02_sd.csv", 16, 3, 30))
METRICS = ("accuracy", "window_balanced", "f1_macro",
           "clip_accuracy", "clip_balanced", "clip_f1_macro")


def summarize(name, path, n_subj, n_seed, max_epoch):
    if not os.path.exists(path):
        print(f"\n[{name}] {path} 없음 — 아직 실행 전")
        return
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in METRICS + ("selected_epoch", "seconds", "n_clips"):
            if r.get(m) not in (None, ""):
                by[int(r["subject"])][m].append(float(r[m]))
    done = len(rows)
    print(f"\n[{name}] {done}/{n_subj * n_seed} runs, 피험자 {len(by)}/{n_subj}"
          + ("" if done == n_subj * n_seed else "  ← 미완료, 부분 결과"))
    for m in METRICS:
        v = np.array([np.mean(by[s][m]) for s in sorted(by) if by[s][m]])
        if len(v):
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            print(f"  {m:<16} {v.mean():.4f} ± {sd:.4f}  (피험자 {len(v)}명, 시드 평균)")
    # 시드 사이 흔들림 — 피험자 안에서
    spread = [np.std(by[s]["clip_accuracy"], ddof=1)
              for s in sorted(by) if len(by[s]["clip_accuracy"]) > 1]
    if spread:
        print(f"  피험자 안 시드 간 sd (clip): 평균 {np.mean(spread):.4f}, 최대 {np.max(spread):.4f}")
    ep = np.concatenate([by[s]["selected_epoch"] for s in by]) if by else np.array([])
    if len(ep):
        # selected_epoch 은 분할 평균이다.  상한 근처에 몰리면 epoch 을 올려야 한다
        # (config 주석: "선택 epoch 이 30 에 몰리면 올린다").
        near = float(np.mean(ep >= max_epoch - 3))
        print(f"  선택 epoch (분할 평균): 중앙 {np.median(ep):.1f}, 최대 {ep.max():.1f}, "
              f"{max_epoch - 3} 이상 비율 {near:.0%}"
              + ("  ← 상한에 몰림, epoch 상향 검토" if near > 0.25 else ""))
    sec = np.concatenate([by[s]["seconds"] for s in by])
    print(f"  run 시간: 중앙 {np.median(sec):.0f}초, 합 {sec.sum() / 3600:.2f}시간")
    nc = sorted({int(x) for s in by for x in by[s]["n_clips"]})
    print(f"  피험자당 평가 클립 수: {nc}")


if __name__ == "__main__":
    for args in SETS:
        summarize(*args)
