"""테스트 세션의 감정 방향이 학습 prototype 과 얼마나 어긋나 있는가 (V6 의 예측 변수).

  c(ts) = mean_{세션 a} mean_{감정 k} cos( Q[ts,a,k], P[k] )

Q 는 analyze_session_geometry.session_protos (세션 평균으로 중심화한 감정별 클립 평균, L2),
P 는 analyze_stratnorm.prototypes(…, train, "center") — V4·V5 와 같은 학습 prototype 이다.
테스트 세션의 감정 라벨을 쓰는 **진단값**이다 (방법이 아니다).  피험자별 시드 평균.

V5 에서 탐색적으로 잰 값: SEED 0.754, SEED-V 0.322 (no-EA 팔).  이 스크립트가 그 값을 재현해야 한다.

    DATASET=seed  python analyze_misalignment.py --cache cache_ft_noea
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from dataset_config import get as _cfg_root
from analyze_centering import roles_for_fold, clip_index, l2
from analyze_session_geometry import session_protos
import analyze_stratnorm as SN

CFG = _cfg_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CFG.cache_dir)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.cache, "S*_seed*.npz"))):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
        z = np.load(f)
        Z = l2(z["cls"]).astype(np.float32); meta = z["meta"].astype(int); lab = z["lab"].astype(int)
        cidx = clip_index(meta)
        train, _, _ = roles_for_fold(ts)
        P = l2(SN.prototypes(Z, meta, lab, cidx, train, "center"))
        Q = session_protos(Z, meta, lab, cidx, ts)
        R[ts].append(float(np.nanmean([np.sum(Q[a] * P, 1) for a in Q])))
    subs = sorted(R)
    c = np.array([np.mean(R[s]) for s in subs])
    print(f"{CFG.name}: 테스트 세션 prototype 과 학습 prototype 의 코사인 c = {c.mean():.3f} "
          f"(피험자 {len(c)}명, 범위 {c.min():.3f}~{c.max():.3f})")
    if args.out:
        np.savez(args.out, subjects=np.array(subs), c=c)


if __name__ == "__main__":
    main()
