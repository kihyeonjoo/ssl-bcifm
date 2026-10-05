"""V3 2b: 검증 피험자에서 뽑은 공유 세션 방향을 사영으로 빼면 정확도가 오르는가.

같은 난수·같은 클립으로 analyze_stratnorm.run_one 을 두 번 돌린다 (사영 없음/있음).
U 는 fold 마다 검증 피험자 2명의 세션 간 prototype 차이에서만 뽑는다.
"""
from __future__ import annotations

import glob, os, re, sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
from dataset_config import get as _cfg_root
from analyze_centering import roles_for_fold, clip_index, l2, bootstrap_ci
from analyze_session_geometry import session_protos, cos_within
from analyze_session_subspace import deltas, top_k
import analyze_stratnorm as SN

CFG = _cfg_root()
K = 4


def main():
    R = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(CFG.cache_dir, "S*_seed*.npz")))
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
        z = np.load(f)
        meta = z["meta"].astype(int); lab = z["lab"].astype(int)
        Z = l2(z["cls"]); cidx = clip_index(meta)
        _, val, _ = roles_for_fold(ts)
        U = top_k(np.concatenate([deltas(session_protos(Z, meta, lab, cidx, s))
                                  for s in val]), K)
        Zp = Z - (Z @ U) @ U.T
        for tag, X in (("base", Z), ("proj", Zp)):
            rng = np.random.default_rng(1000 * ts + sd)     # 두 쪽 같은 클립
            r = SN.run_one(X, meta, lab, ts, [20, 40], 10, rng)
            for k in ("center_T20|clip", "center_T40|clip", "center_Tfull|clip",
                      "center_T20|win", "center_Tfull|win", "none|clip"):
                R[f"{tag}:{k}"][ts].append(r[k])
            R[f"{tag}:cw_test"][ts].append(
                cos_within(session_protos(l2(X), meta, lab, cidx, ts)))
        print(f"  [{j}/{len(files)}] S{ts} seed{sd}  Tfull clip "
              f"{R['base:center_Tfull|clip'][ts][-1]:.3f} -> "
              f"{R['proj:center_Tfull|clip'][ts][-1]:.3f}   cw_test "
              f"{R['base:cw_test'][ts][-1]:.3f} -> {R['proj:cw_test'][ts][-1]:.3f}",
              flush=True)
    arr = lambda key: np.array([np.mean(R[key][s]) for s in sorted(R[key])])
    np.savez(f"results/{CFG.prefix('session_projection')}.npz",
             **{key.replace(":", "__").replace("|", "__"): arr(key) for key in R})
    print(f"\n{CFG.name}  (k={K}, 검증 피험자 방향 제거)")
    for k in ("center_T20|clip", "center_T40|clip", "center_Tfull|clip",
              "center_T20|win", "center_Tfull|win", "cw_test"):
        a, b = arr(f"proj:{k}"), arr(f"base:{k}")
        d = a - b; lo, hi = bootstrap_ci(d); p = stats.wilcoxon(a, b).pvalue
        print(f"  {k:18s} 기존 {b.mean():.4f}  사영 {a.mean():.4f}  Δ{d.mean():+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} {int((d>0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
