"""V5 gate 0 (성립성 진단, 방법 평가 아님).  gate 0 교차검증: 캘리브레이션 클립(감정당 1개, 클립 전체)의 시점 대응으로 맞춘 회전이
**다른 클립**에서 공유 신호(동적 ISC, 같은 클립 정적 유사도)를 늘리는가.  라벨·정확도 미사용.
회전: 템플릿 PCA 상위 k 부분공간 안의 Procrustes, 단위행렬 쪽 수축 β."""
import glob, os, re, sys
import numpy as np
from collections import defaultdict
from scipy import stats
sys.path.insert(0, os.getcwd())
from analyze_centering import roles_for_fold, clip_index, l2, N_CLS

def fcos(A, B):
    a, b = A.ravel(), B.ravel(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def sub_procrustes(X, T, k, beta):
    d = X.shape[1]
    U = np.linalg.svd(np.concatenate([X, T]) - 0, full_matrices=False)[2][:k].T   # d x k
    A, B = X @ U, T @ U
    u, _, vt = np.linalg.svd(A.T @ B, full_matrices=False)
    Rk = u @ vt
    W = np.eye(d) + U @ (Rk - np.eye(k)) @ U.T
    return (1 - beta) * np.eye(d) + beta * W

cache = sys.argv[1]; CONF = [(10, 1.0), (20, 1.0), (20, 0.5), (50, 0.5)]
R = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(os.path.join(cache, "S*_seed*.npz"))):
    ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
    z = np.load(f); Z = l2(z["cls"]).astype(np.float64); meta = z["meta"].astype(int); lab = z["lab"].astype(int)
    cidx = clip_index(meta); train, _, _ = roles_for_fold(ts); tr = list(train)
    Zc = np.empty_like(Z)
    for d in {(int(m[0]), int(m[1])) for m in meta}:
        msk = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1]); Zc[msk] = Z[msk] - Z[msk].mean(0)
    keys = sorted({(k[1], k[2]) for k in cidx if k[0] == ts})
    T = {k: np.mean([Zc[cidx[(u,) + k]] for u in tr], 0) for k in keys}
    rng = np.random.default_rng(7000 + 100 * ts + sd)
    for rep in range(3):
        for a in sorted({k[0] for k in keys}):
            ks = [k for k in keys if k[0] == a]
            by = defaultdict(list)
            for k in ks: by[int(lab[cidx[(ts,) + k][0]])].append(k)
            cal = [by[c][rng.integers(len(by[c]))] for c in sorted(by)]
            ev = [k for k in ks if k not in cal]
            X = np.concatenate([Zc[cidx[(ts,) + k]] for k in cal]); Tc = np.concatenate([T[k] for k in cal])
            def score(W):
                dyn, st = [], []
                for k in ev:
                    Xk = Zc[cidx[(ts,) + k]] @ W
                    dyn.append(fcos(Xk - Xk.mean(0), T[k] - T[k].mean(0))); st.append(fcos(Xk.mean(0), T[k].mean(0)))
                return np.mean(dyn), np.mean(st)
            b = score(np.eye(Z.shape[1])); R["dyn|none"][ts].append(b[0]); R["st|none"][ts].append(b[1])
            for (k, beta) in CONF:
                s = score(sub_procrustes(X, Tc, k, beta))
                R[f"dyn|k{k}b{beta}"][ts].append(s[0]); R[f"st|k{k}b{beta}"][ts].append(s[1])
A = {k: np.array([np.mean(v[s]) for s in sorted(v)]) for k, v in R.items()}
for m in ("dyn", "st"):
    base = A[f"{m}|none"]; print(f"  [{ '동적 ISC' if m=='dyn' else '정적 같은클립'}] 회전 없음 {base.mean():+.4f}")
    for (k, beta) in CONF:
        a = A[f"{m}|k{k}b{beta}"]; d = a - base
        print(f"     k={k:<3} β={beta:<4} {a.mean():+.4f}  Δ {d.mean():+.4f}  p={stats.wilcoxon(a, base).pvalue:.3g}  {int((d>0).sum())}/{len(d)}")

if len(sys.argv) > 2:          # 결과 저장 (METHOD.pdf 가 읽는다)
    np.savez(sys.argv[2], **{k.replace("|", "__"): v for k, v in A.items()})
    print(f"[저장] {sys.argv[2]}")
