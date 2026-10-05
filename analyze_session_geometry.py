"""V3 1단계 진단: 세션 기하 — 사전 등록 reports/V3_SESSION_GEOMETRY_PREREG.md.

각 fold 모델에서 피험자를 학습/검증/테스트 역할로 나누고, 세션별로 중심화한 뒤의
감정 prototype 이 세션 사이에서 얼마나 일치하는지 잰다.

  Q_{s,a,c} = l2( mean_{clips k of class c in session a}
                   ( mean_{windows of k} z  −  μ_{s,a} ) )
  cos_within(s)  = mean_{c, a<b} cos(Q_{s,a,c}, Q_{s,b,c})       같은 피험자, 다른 세션
  cos_between    = mean_{c, s≠t, a, b} cos(Q_{s,a,c}, Q_{t,b,c})  다른 피험자

z 는 L2 정규화한 CLS (조건4 경로와 같은 표현).
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
from dataset_config import get as _cfg_root
from analyze_centering import N_CLS, roles_for_fold, clip_index, l2

CFG = _cfg_root()


def session_protos(Z, meta, lab, cidx, subj):
    """{session: (N_CLS, d) array of L2-normalised centred class prototypes}"""
    out = {}
    for ss in sorted({int(m) for m in meta[meta[:, 0] == subj, 1]}):
        m = (meta[:, 0] == subj) & (meta[:, 1] == ss)
        mu = Z[m].mean(0)
        P = []
        for c in range(N_CLS):
            ks = [k for k in cidx if k[0] == subj and k[1] == ss
                  and int(lab[cidx[k][0]]) == c]
            if not ks:
                P.append(np.full(Z.shape[1], np.nan)); continue
            P.append(np.mean([Z[cidx[k]].mean(0) - mu for k in ks], axis=0))
        out[ss] = l2(np.stack(P))
    return out


def cos_within(protos):
    vals = []
    sess = sorted(protos)
    for i in range(len(sess)):
        for j in range(i + 1, len(sess)):
            A, B = protos[sess[i]], protos[sess[j]]
            vals.extend(np.sum(A * B, axis=1).tolist())
    return float(np.nanmean(vals))


def cos_between(pa, pb):
    vals = []
    for A in pa.values():
        for B in pb.values():
            vals.extend(np.sum(A * B, axis=1).tolist())
    return float(np.nanmean(vals))


def main():
    R = defaultdict(lambda: defaultdict(list))   # metric -> test subj -> [per seed]
    files = sorted(glob.glob(os.path.join(CFG.cache_dir, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
        z = np.load(f)
        Z = l2(z["cls"]); meta = z["meta"].astype(int); lab = z["lab"].astype(int)
        cidx = clip_index(meta)
        train, val, _ = roles_for_fold(ts)
        P = {s: session_protos(Z, meta, lab, cidx, s) for s in list(train) + list(val) + [ts]}

        cw_train = np.mean([cos_within(P[s]) for s in train])
        cw_val = np.mean([cos_within(P[s]) for s in val])
        cw_test = cos_within(P[ts])
        # 다른 피험자 사이 (학습 피험자끼리) — 세션 변동과 피험자 변동을 같은 눈금에서
        tr = list(train)
        cb_train = np.mean([cos_between(P[tr[a]], P[tr[b]])
                            for a in range(len(tr)) for b in range(a + 1, len(tr))])
        cb_test = np.mean([cos_between(P[ts], P[s]) for s in train])

        for k, v in (("cw_train", cw_train), ("cw_val", cw_val), ("cw_test", cw_test),
                     ("cb_train", cb_train), ("cb_test_to_train", cb_test)):
            R[k][ts].append(v)
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  within: train {cw_train:.3f} "
              f"val {cw_val:.3f} test {cw_test:.3f}   between: train {cb_train:.3f} "
              f"test→train {cb_test:.3f}", flush=True)

    arr = lambda k: np.array([np.mean(R[k][s]) for s in sorted(R[k])])
    out = {k: arr(k) for k in R}
    np.savez(f"results/{CFG.prefix('session_geometry')}.npz", **out,
             subjects=np.array(sorted(R["cw_test"])))
    print(f"\n[저장] results/{CFG.prefix('session_geometry')}.npz")


if __name__ == "__main__":
    main()
