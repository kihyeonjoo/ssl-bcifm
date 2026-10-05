"""
테스트 시점 적응 방법들 — 전부 같은 캐시, 같은 프로토콜, 같은 판정 기준.

모두 **결정 규칙만 바꾸는** 방법이다 (백본·head 재학습 없음).  따라서 판정은 그
비교의 짝지은 검정으로 하고, 재학습 비교용 최소 감지 효과 +0.0386 은 쓰지 않는다.

구현된 것
---------
mean          단순 평균 중심화 (= S8/S9 의 조건2·4).  기준선.
M1            감정 편향 제거 기준점 추정.  캘리브레이션 창의 감정 구성이 치우쳐
              있으면 단순 평균은 도메인 오프셋에 그 감정의 클래스 방향을 섞어
              가진다.  클래스 확률로 그 성분을 빼면서 μ 를 다시 추정한다.
M2            pseudo-label Procrustes 회전.  중심화 후에도 남는 클래스 방향의
              회전을, 학습 prototype 이 이루는 저차원 부분공간 안에서만 맞춘다.
adaBN         도메인별 평균·분산 정규화 (BN 통계 교체의 특징 공간 대응).
LA            Latent Alignment — 도메인별 특징 표준화.
T3A           확신도 높은 테스트 특징으로 prototype 을 점진 갱신.

용어
----
raw CLS   head 가 실제로 보는 벡터 (LayerNorm 전).  조건2 경로가 쓴다.
z CLS     창별 L2 정규화된 벡터.  조건4(prototype) 경로가 쓴다.
"""

from __future__ import annotations

import numpy as np

from analyze_centering import N_CLS, l2, head_probs

EPS = 1e-8


# ── 학습 쪽 재료 ─────────────────────────────────────────────────────────────

def train_prototypes_raw(cls, meta, lab, cidx, train_subjects):
    """중심화된 **raw** CLS 공간의 클래스 prototype (L2 정규화 없이, 크기 유지).

    M1 은 μ 를 raw 공간에서 추정하므로 prototype 도 raw 여야 한다.  L2 로
    정규화하면 크기 정보가 사라져 ``z - Σ p·P`` 의 스케일이 맞지 않는다.
    """
    keys = sorted(k for k in cidx if k[0] in train_subjects)
    doms = sorted({(k[0], k[1]) for k in keys})
    E = np.stack([cls[cidx[k]].mean(0) for k in keys])      # clip mean, raw
    y = np.array([lab[cidx[k][0]] for k in keys])
    dm = np.array([(k[0], k[1]) for k in keys])
    for d in doms:                                          # centre each domain
        m = (dm[:, 0] == d[0]) & (dm[:, 1] == d[1])
        E[m] -= E[m].mean(0)
    return np.stack([np.nanmean(
        [E[(dm[:, 0] == d[0]) & (dm[:, 1] == d[1]) & (y == c)].mean(0)
         for d in doms], axis=0) for c in range(N_CLS)])


# ── M1 ──────────────────────────────────────────────────────────────────────

def m1_mean(Xc, P_raw, tau=1.0, n_iter=3, gamma=0.5, return_trace=False):
    """감정 편향을 뺀 기준점 μ.

    Xc  (n, d) 한 도메인의 캘리브레이션 창 raw CLS
    P_raw (3, d) 학습 도메인 raw prototype

    단순 평균 μ0 = mean(Xc) 는 그 창들이 담은 감정의 클래스 방향을 함께 갖는다.
    감정이 균형이면 클래스 방향들이 서로 상쇄되어 문제가 없지만, 한 감정만
    들어 있으면 μ0 는 도메인 오프셋 + 그 클래스 방향이 되고, 그것을 빼면 그
    클래스 신호까지 지워진다 (S8: 단일 감정 캘리브레이션 15/15 실패).

    반복:
      (1) 현재 μ 로 중심화
      (2) prototype 과의 코사인을 온도 τ 로 소프트맥스 -> p_ic
      (3) μ <- mean_i( x_i − Σ_c p_ic P_c )

    τ -> ∞ 는 p 가 균등해져 μ -> mean(Xc) − mean(P) 가 되고, prototype 평균이
    0 에 가까우므로 단순 평균으로 되돌아간다.  즉 τ 는 "보정을 얼마나 믿는가" 다.

    **γ (축소) 가 없으면 이 보정은 해롭다.**  P_c 는 학습 도메인 평균이고
    fine-tuning 이 그 도메인들의 클래스 방향을 증폭시켰으므로, 측정해 보면
    ‖P_c‖ ≈ 0.70 인데 테스트 피험자 자신의 클래스 편차는 ‖δ_c‖ ≈ 0.33 이다
    (S2 의 "학습 0.96 대 테스트 0.755" 와 같은 과적합).  코사인이 0.85 여도 두 배
    긴 벡터를 빼면 ‖δ−P‖/‖δ‖ = 1.7 로 **보정 안 한 것보다 오차가 커진다.**
    γ 는 그 길이 차이를 흡수하며, 검증으로 고른 값이 예측값 ‖δ‖/‖P‖ ≈ 0.47 에
    가깝게 나온다.
    """
    mu = Xc.mean(0)
    trace = [mu.copy()]
    Pn = l2(P_raw)
    for _ in range(n_iter):
        A = Xc - mu
        logits = l2(A) @ Pn.T / max(tau, EPS)
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        mu = (Xc - gamma * (p @ P_raw)).mean(0)
        trace.append(mu.copy())
    return (mu, trace) if return_trace else mu


# ── M2 ──────────────────────────────────────────────────────────────────────

def _subspace(P, extra, k_extra):
    """Orthonormal basis of span(prototypes) plus ``k_extra`` PCA axes of ``extra``.

    Fitting a full d=200 rotation from 3 noisy class means is hopeless, so the
    rotation is confined to the few directions that carry the class signal."""
    B = P.T                                                  # (d, 3)
    if k_extra > 0 and extra is not None and len(extra) > k_extra:
        R = extra - extra.mean(0)
        # remove the prototype span first so the PCA axes add new directions
        Q0, _ = np.linalg.qr(B)
        R = R - (R @ Q0) @ Q0.T
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        B = np.concatenate([B, Vt[:k_extra].T], axis=1)
    Q, _ = np.linalg.qr(B)
    return Q                                                 # (d, r)


def m2_rotation(A, pred, P, extra=None, k_extra=2, beta=0.5, min_per_class=3):
    """Orthogonal rotation aligning the test class means to the train prototypes.

    A     (n, d) centred test features (clip or window level)
    pred  (n,)   pseudo-labels
    P     (3, d) train prototypes in the same space
    beta         shrinkage toward the identity: W = (1-beta) I + beta R.
                 beta=0 disables the rotation, beta=1 applies it fully.

    Returns a (d, d) operator to apply as ``A @ W``.  If a class has too few
    confident members the rotation is not identifiable, so the identity is
    returned rather than a transform fitted from one point.
    """
    d = A.shape[1]
    I = np.eye(d, dtype=np.float32)
    cls_means, tgt = [], []
    for c in range(N_CLS):
        m = pred == c
        if m.sum() < min_per_class:
            return I, False
        cls_means.append(A[m].mean(0))
        tgt.append(P[c])
    M = np.stack(cls_means)                                  # (3, d)
    T = np.stack(tgt)
    Q = _subspace(P, extra, k_extra)                         # (d, r)
    Ms, Ts = M @ Q, T @ Q                                    # (3, r)
    # Procrustes in the subspace: min ||Ms R - Ts||, R orthogonal
    U, _, Vt = np.linalg.svd(Ms.T @ Ts)
    Rsub = U @ Vt
    if np.linalg.det(Rsub) < 0:                              # keep it a rotation
        U[:, -1] *= -1
        Rsub = U @ Vt
    # lift back: act as Rsub inside the subspace, identity outside
    W = I - Q @ Q.T + Q @ Rsub @ Q.T
    return ((1 - beta) * I + beta * W).astype(np.float32), True


# ── 기존 방법 ────────────────────────────────────────────────────────────────

def adabn(X, calib):
    """도메인별 평균·분산 정규화.

    BN 통계를 테스트 도메인 것으로 바꾸는 adaBN 을 특징 공간에서 한 것.  차원별
    평균과 표준편차를 모두 캘리브레이션 창에서 추정한다.  중심화가 평균만 쓰는
    데 비해 스케일까지 바꾼다."""
    mu, sd = calib.mean(0), calib.std(0) + EPS
    return (X - mu) / sd


def latent_align(X, calib):
    """Latent Alignment 방식 — 도메인별 특징 표준화.

    **구현 가정**: 원 논문들은 도메인별 1·2차 모멘트를 맞추되 대상 통계를
    소스 통계로 되돌린다.  여기서는 캘리브레이션 창으로 표준화한 뒤 **학습
    도메인의 평균·표준편차로 되돌려** 놓는다 (head 가 학습 때 본 범위를
    유지해야 하므로).  adaBN 과의 차이는 이 되돌림뿐이다.
    """
    raise NotImplementedError          # bound in the analysis script, which
                                       # holds the train-side statistics


def t3a_prototypes(P0, Zt, probs, thresh=0.9, max_per_class=20):
    """확신도 높은 테스트 특징을 prototype 에 합친다 (T3A).

    P0    (3, d) 학습 prototype (support set 의 초기값)
    Zt    (n, d) 테스트 특징 (같은 공간)
    probs (n, 3) 현재 분류기의 확률
    라벨을 쓰지 않고, 확신도 문턱을 넘은 것만 클래스별로 최대 ``max_per_class``
    개까지 골라 평균에 넣는다.  문턱이 낮으면 틀린 것이 섞이고 높으면 아무것도
    안 들어와 P0 그대로가 된다.
    """
    P = P0.copy()
    conf, pred = probs.max(1), probs.argmax(1)
    for c in range(N_CLS):
        m = (pred == c) & (conf >= thresh)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(-conf[idx])][:max_per_class]
        P[c] = np.concatenate([P0[c][None], Zt[idx]]).mean(0)
    return P
