"""
분석 1: EA 는 클래스 방향의 전이를 돕는가?

가설: EA 의 이득은 특징 공간의 세션 오프셋을 줄여서가 아니라, fine-tuning 이 만든
클래스 방향이 처음 보는 피험자에게 이어지는 정도를 높여서 생긴다.

S7 에서 EA 가 도메인 간 분산을 줄이지 않는다는 음성 결과가 나왔으므로, 남은 설명
후보는 "방향의 전이" 다.  세 측정을 EA 있음/없음 캐시에 똑같이 적용하고 피험자 단위로
짝짓는다.

  role_train / role_val / role_test   중심화 후 clip 단위 교차 prototype 정확도
                                      (S2 결과 3 방식, 역할별)
  ceil_subject                        테스트 피험자 자기 도메인 상한
                                      (피험자 단위 LOCO, 최근접 prototype)
  gap = ceil_subject - role_test      전이 격차.  EA 로 줄면 가설 지지.
  cos_k                               학습 도메인 평균 prototype 과 테스트 도메인
                                      prototype 의 클래스별 코사인

모두 중심화된 표현에서 재므로, 중심화가 이미 제거하는 1차 모멘트 차이는 여기에
섞이지 않는다 — 측정되는 것은 방향뿐이다.  GPU 없음.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
# 데이터셋은 환경변수 DATASET 으로 고른다.  기본 출력 경로를 설정에서
# 끌어오는 이유: --out 을 한 번 빠뜨리면 SEED 결과 파일을 덮어쓴다.
from dataset_config import get as _cfg_root
CFG = _cfg_root()

from scipy import stats

from analyze_centering import (N_CLS, roles_for_fold, clip_index, l2,
                               clip_reduce, all_domains, cross_for,
                               bootstrap_ci)


def _centred_clips(cls, meta, lab, subjects):
    """Per-domain-centred clip embeddings for a set of subjects."""
    cidx = clip_index(meta)
    keys = sorted(k for k in cidx if k[0] in subjects)
    Z = l2(cls)
    E = clip_reduce(Z, cidx, keys)
    y = np.array([lab[cidx[k][0]] for k in keys])
    dom = np.array([(k[0], k[1]) for k in keys])
    for d in {tuple(x) for x in dom}:
        m = (dom[:, 0] == d[0]) & (dom[:, 1] == d[1])
        E[m] -= E[m].mean(0)
    return E, y, dom


def _prototypes(E, y, dom):
    """Class prototypes: per-domain class mean, then averaged over domains.

    Averaging over domains first keeps a domain with more clips of one class
    from dominating that class's direction."""
    return np.stack([np.nanmean(
        [E[(dom[:, 0] == d[0]) & (dom[:, 1] == d[1]) & (y == k)].mean(0)
         for d in {tuple(x) for x in dom}], axis=0) for k in range(N_CLS)])


def ceil_subject(cls, meta, lab, test_subj):
    """Subject-level leave-one-clip-out, nearest prototype.

    Each session is centred on its own mean first, so the session offset is
    removed exactly as in the cross-subject path; what is left is how well the
    subject's OWN class directions separate its own clips.  This is the ceiling
    the cross-subject transfer is chasing."""
    E, y, dom = _centred_clips(cls, meta, lab, [test_subj])
    hit = 0
    for i in range(len(E)):
        o = np.arange(len(E)) != i
        P = np.stack([E[o & (y == k)].mean(0) for k in range(N_CLS)])
        hit += int((l2(E[i:i + 1]) @ l2(P).T).argmax(1)[0] == y[i])
    return hit / len(E)


def class_cosines(cls, meta, lab, test_subj):
    """Cosine between the train-domain mean prototype and the test subject's
    own prototype, per class."""
    train, _, _ = roles_for_fold(test_subj)
    Etr, ytr, dtr = _centred_clips(cls, meta, lab, train)
    Ete, yte, dte = _centred_clips(cls, meta, lab, [test_subj])
    P, Q = _prototypes(Etr, ytr, dtr), _prototypes(Ete, yte, dte)
    return (l2(P) * l2(Q)).sum(1)          # per class


def collect(cache_dir, arm):
    R = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(cache_dir, "S*_seed*.npz")))
    if not files:
        raise SystemExit(f"no cache in {cache_dir}")
    for j, f in enumerate(files, 1):
        ts, sd = map(int, re.search(r"S(\d+)_seed(\d+)",
                                    os.path.basename(f)).groups())
        z = np.load(f)
        cls, meta, lab = z["cls"], z["meta"].astype(int), z["lab"].astype(int)
        train, val, _ = roles_for_fold(ts)
        r_tr = float(np.nanmean([cross_for(cls, meta, lab, [s],
                                           [t for t in train if t != s])
                                 for s in train]))
        r_va = float(np.nanmean([cross_for(cls, meta, lab, [s], train)
                                 for s in val]))
        r_te = cross_for(cls, meta, lab, [ts], train)
        ceil = ceil_subject(cls, meta, lab, ts)
        cos = class_cosines(cls, meta, lab, ts)
        R["role_train"][ts].append(r_tr)
        R["role_val"][ts].append(r_va)
        R["role_test"][ts].append(r_te)
        R["ceil"][ts].append(ceil)
        R["gap"][ts].append(ceil - r_te)
        for k in range(N_CLS):
            R[f"cos{k}"][ts].append(float(cos[k]))
        R["cos_mean"][ts].append(float(cos.mean()))
        print(f"  [{arm} {j}/{len(files)}] S{ts} seed{sd}  "
              f"train {r_tr:.3f} val {r_va:.3f} test {r_te:.3f}  "
              f"ceil {ceil:.3f} gap {ceil - r_te:+.3f}  cos {cos.mean():.3f}",
              flush=True)
    return R


def arr(R, key):
    return np.array([np.mean(R[key][s]) for s in sorted(R[key])])


def paired(a, b, label):
    d = a - b
    lo, hi = bootstrap_ci(d)
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    zero = "  0포함" if lo <= 0 <= hi else "       "
    print(f"  {label:<40} Δ{d.mean():+.4f}  CI[{lo:+.4f},{hi:+.4f}]{zero}"
          f"  p={p:.4f}  {int((d > 0).sum())}/{len(d)}")
    return d.mean(), lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea_cache", default=CFG.cache_dir)
    ap.add_argument("--noea_cache", default=CFG.cache_dir + "_noea")
    ap.add_argument("--out", default=f"results/{CFG.prefix('ea_transfer')}.npz")
    args = ap.parse_args()

    print("[EA 있음] A팔")
    EA = collect(args.ea_cache, "EA")
    print("\n[EA 없음]")
    NO = collect(args.noea_cache, "noEA")

    W = 94
    print(f"\n{'='*W}\n역할별 정렬과 전이 격차  (n={CFG.n_subjects}, 시드 평균 평균, 중심화 clip 단위)\n{'='*W}")
    print(f"{'':<22}{'EA 없음':>18}{'EA 있음 (A팔)':>20}{'차이':>12}")
    for key, nm in (("role_train", "학습 도메인"), ("role_val", "검증 도메인"),
                    ("role_test", "테스트 도메인"), ("ceil", "자기 도메인 상한"),
                    ("gap", "격차 (상한−테스트)"), ("cos_mean", "클래스 코사인 평균")):
        n_, e_ = arr(NO, key), arr(EA, key)
        print(f"  {nm:<20} {n_.mean():>10.4f} ± {n_.std(ddof=1):.4f} "
              f"{e_.mean():>10.4f} ± {e_.std(ddof=1):.4f} {e_.mean()-n_.mean():>+12.4f}")

    print(f"\n{'='*W}\n짝지은 비교 (EA 있음 − EA 없음)\n{'='*W}")
    print("  가설 지지 조건: 테스트 정렬↑, 격차↓, 코사인↑")
    paired(arr(EA, "role_test"), arr(NO, "role_test"), "테스트 도메인 정렬")
    paired(arr(EA, "gap"), arr(NO, "gap"), "전이 격차  (음수면 가설 지지)")
    paired(arr(EA, "cos_mean"), arr(NO, "cos_mean"), "클래스 코사인 평균")
    print()
    paired(arr(EA, "ceil"), arr(NO, "ceil"), "자기 도메인 상한 (교란 확인)")
    paired(arr(EA, "role_train"), arr(NO, "role_train"), "학습 도메인 정렬 (교란 확인)")

    print(f"\n{'='*W}\n클래스별 코사인\n{'='*W}")
    names = [f"{nm}({i})" for i, nm in enumerate(CFG.class_names)]
    for k in range(N_CLS):
        n_, e_ = arr(NO, f"cos{k}"), arr(EA, f"cos{k}")
        d = e_ - n_
        lo, hi = bootstrap_ci(d)
        print(f"  {names[k]:<10} EA없음 {n_.mean():.4f}  EA있음 {e_.mean():.4f}  "
              f"Δ{d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]"
              f"{'  0포함' if lo <= 0 <= hi else ''}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **{f"ea_{k}": arr(EA, k) for k in EA},
             **{f"noea_{k}": arr(NO, k) for k in NO})
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
