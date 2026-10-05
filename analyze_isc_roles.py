"""V5 gate 0 (성립성 진단, 방법 평가 아님).  gate 0 보강: 동적 ISC 를 역할별로 — 학습(LOO 템플릿) / 검증(기울기 없음) / 테스트.
그리고 '어긋났지만 남아 있는가': 테스트 피험자 세션마다 시점 대응 전체로 맞춘 oracle 직교 회전
(평가 데이터 자신에 맞춤, 상한) 뒤의 ISC.  진단 전용."""
import glob, os, re, sys
import numpy as np
from collections import defaultdict
from scipy import stats
sys.path.insert(0, os.getcwd())
from analyze_centering import roles_for_fold, clip_index, l2

def fcos(A, B):
    a, b = A.ravel(), B.ravel(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def procrustes(X, T):
    U, _, Vt = np.linalg.svd(X.T @ T, full_matrices=False); return U @ Vt

cache = sys.argv[1]
R = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(os.path.join(cache, "S*_seed*.npz"))):
    ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)", os.path.basename(f)).groups())
    z = np.load(f); Z = l2(z["cls"]).astype(np.float64); meta = z["meta"].astype(int)
    cidx = clip_index(meta); train, val, _ = roles_for_fold(ts)
    Zc = np.empty_like(Z)
    for d in {(int(m[0]), int(m[1])) for m in meta}:
        msk = (meta[:, 0] == d[0]) & (meta[:, 1] == d[1]); Zc[msk] = Z[msk] - Z[msk].mean(0)
    keys = sorted({(k[1], k[2]) for k in cidx if k[0] == ts})
    tr = list(train)
    def dyn_isc(s, template_subs):
        v = []
        for (a, c) in keys:
            X = Zc[cidx[(s, a, c)]]; T = np.mean([Zc[cidx[(u, a, c)]] for u in template_subs], 0)
            v.append(fcos(X - X.mean(0), T - T.mean(0)))
        return float(np.mean(v))
    R["train"][ts].append(np.mean([dyn_isc(s, [u for u in tr if u != s]) for s in tr[:4]]))
    R["val"][ts].append(np.mean([dyn_isc(s, tr) for s in val]))
    R["test"][ts].append(dyn_isc(ts, tr))
    # oracle: 세션마다 (클립 평균 포함) 시점 대응 전체로 직교 회전 → 동적 ISC
    v = []
    for a in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == a]
        X = np.concatenate([Zc[cidx[(ts, a, c)]] for (_, c) in ks])
        T = np.concatenate([np.mean([Zc[cidx[(u, a, c)]] for u in tr], 0) for (_, c) in ks])
        W = procrustes(X, T)
        for (_, c) in ks:
            Xk = Zc[cidx[(ts, a, c)]] @ W; Tk = np.mean([Zc[cidx[(u, a, c)]] for u in tr], 0)
            v.append(fcos(Xk - Xk.mean(0), Tk - Tk.mean(0)))
    R["test_oracle_rot"][ts].append(float(np.mean(v)))
A = {k: np.array([np.mean(v[s]) for s in sorted(v)]) for k, v in R.items()}
for k in ("train", "val", "test", "test_oracle_rot"):
    print(f"  {k:<16} 동적 ISC {A[k].mean():+.4f} ± {A[k].std(ddof=1):.4f}")
for a, b in (("train", "test"), ("val", "test"), ("test_oracle_rot", "test")):
    d = A[a] - A[b]; print(f"  {a} − {b}: {d.mean():+.4f}  p={stats.wilcoxon(A[a], A[b]).pvalue:.3g}  {int((d>0).sum())}/{len(d)}")

if len(sys.argv) > 2:          # 결과 저장 (METHOD.pdf 가 읽는다)
    np.savez(sys.argv[2], **{k.replace("|", "__"): v for k, v in A.items()})
    print(f"[저장] {sys.argv[2]}")
