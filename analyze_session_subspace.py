"""V3 2a: 보지 못한 피험자의 세션 간 어긋남은 체계적인가 (사전 등록은 V3 문서).

세션별 중심화 prototype 의 세션 간 차이 ΔQ 가 공유 부분공간에 모이는지 본다.
검증 피험자(기울기 없음)에서 뽑은 방향이 테스트 피험자의 차이를 설명하면 체계적이다.
"""
from __future__ import annotations

import glob, os, re, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from dataset_config import get as _cfg_root
from analyze_centering import roles_for_fold, clip_index, l2
from analyze_session_geometry import session_protos

CFG = _cfg_root()
KS = [1, 2, 4, 8]


def deltas(protos):
    s = sorted(protos)
    D = [protos[s[i]] - protos[s[j]] for i in range(len(s)) for j in range(i + 1, len(s))]
    D = np.concatenate(D, 0)
    return D[~np.isnan(D).any(1)]


def top_k(D, k):
    _, _, Vt = np.linalg.svd(D - 0.0, full_matrices=False)   # 원점 기준 (차이 벡터)
    return Vt[:k].T


def captured(D, U):
    return float(np.sum((D @ U) ** 2) / np.sum(D ** 2))


def main():
    R = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(CFG.cache_dir, "S*_seed*.npz")))
    for f in files:
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
        z = np.load(f)
        Z = l2(z["cls"]); meta = z["meta"].astype(int); lab = z["lab"].astype(int)
        cidx = clip_index(meta)
        train, val, _ = roles_for_fold(ts)
        Dv = np.concatenate([deltas(session_protos(Z, meta, lab, cidx, s)) for s in val])
        Dt = np.concatenate([deltas(session_protos(Z, meta, lab, cidx, s)) for s in train])
        Dx = deltas(session_protos(Z, meta, lab, cidx, ts))
        for k in KS:
            R[f"val_k{k}"][ts].append(captured(Dx, top_k(Dv, k)))
            R[f"train_k{k}"][ts].append(captured(Dx, top_k(Dt, k)))
            R[f"ceil_k{k}"][ts].append(captured(Dx, top_k(Dx, k)))
    arr = lambda key: np.array([np.mean(R[key][s]) for s in sorted(R[key])])
    d = Z.shape[1]
    print(f"{CFG.name}: 테스트 피험자 ΔQ 에너지 포착률 (피험자 평균, 시드 평균)")
    print(f"  {'k':>3}{'검증 방향':>11}{'학습 방향':>11}{'상한(자기)':>12}{'무작위 k/d':>12}"
          f"{'검증/상한':>11}{'검증/무작위':>12}")
    for k in KS:
        v, t, c = arr(f"val_k{k}").mean(), arr(f"train_k{k}").mean(), arr(f"ceil_k{k}").mean()
        print(f"  {k:>3}{v:>11.3f}{t:>11.3f}{c:>12.3f}{k/d:>12.3f}{v/c:>11.2f}{v/(k/d):>12.1f}")
    np.savez(f"results/{CFG.prefix('session_subspace')}.npz",
             **{key: arr(key) for key in R})


if __name__ == "__main__":
    main()
