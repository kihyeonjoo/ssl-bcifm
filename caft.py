"""캘리브레이션 인지 파인튜닝 (Calibration-Aware Fine-Tuning, CAFT) — 2026-10-05 시험.

배포 때 새 사용자는 캘리브레이션 블록 (감정별 영상 한 편) 의 평균 특징을 빼고 (도메인 중심화) 분류한다.  CAFT 는 그 조건을
파인튜닝 안에서 흉내 낸다.

배치 = 같은 세션의 학습 피험자 K 명 × 같은 M 개 (클립, 시점) 위치.  위치는 감정별로 고르게 뽑는다 (캘리브레이션 블록처럼).
배치 순서는 피험자 우선: [피험자1의 M 창, 피험자2의 M 창, ...], 위치 순서는 모든 피험자가 같다.

① 배치 안 즉석 중심화: 각 피험자의 M 창 평균을 그 피험자 창에서 뺀 뒤 head 에 넣는다.  평균은 현재 모델로 바로 계산하므로
   9월 시도 (미리 계산한 도메인 평균표) 의 '낡음' 이 없다.  기울기는 평균을 통해서도 흐른다 (배치 정규화와 같은 방식).
② 사람 간 자극 시점 정렬 손실 (선택): 같은 (클립, 시점) 위치의 K 개 중심화 특징이 서로 같은 방향을 가리키도록, 위치마다
   평균 쌍별 (1 − 코사인) 을 줄인다.  K 개 단위벡터의 평균 m 에 대해 평균 쌍별 코사인 = (K·|m|² − 1)/(K − 1).

검증 · 시험 평가는 기존 domain_center + center_fresh_eval 경로 (점수 매기기 직전 현재 모델로 도메인 평균) 를 쓴다.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


class CAFTBatchSampler(Sampler):
    """같은 세션의 피험자 K 명이 같은 (클립, 시점) M 곳을 보는 창들로 배치를 만든다."""

    def __init__(self, ds, n_classes: int, subjects_per_batch: int = 4, windows_per_subject: int = 15,
                 seed: int = 0):
        self.K, self.M, self.n_cls = int(subjects_per_batch), int(windows_per_subject), int(n_classes)
        self.seed, self.epoch = int(seed), 0
        meta = ds._seg_meta                                         # [(subject, session, clip)] (창 순서 = 시간 순서)
        labels = [s[1] for s in ds._segments]
        self.idx: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for i, (s, a, c) in enumerate(meta):
            self.idx[(s, a, c)].append(i)
        lab_of: Dict[Tuple[int, int], List[int]] = defaultdict(list)   # (세션, 클립) → 피험자별 라벨
        subj_of: Dict[int, set] = defaultdict(set)                       # 세션 → 그 세션이 있는 피험자
        for (s, a, c), w in self.idx.items():
            lab_of[(a, c)].append(int(labels[w[0]]))
            subj_of[a].add(s)
        # 세션마다: 감정별 클립 목록 (라벨이 피험자마다 다르면 다수결 — SEED 계열은 모두 같다)
        self.sessions = sorted(a for a, ss in subj_of.items() if len(ss) >= self.K)
        self.subj_of = {a: sorted(subj_of[a]) for a in self.sessions}
        self.clips_by_cls: Dict[int, Dict[int, List[int]]] = {}
        for a in self.sessions:
            d = defaultdict(list)
            for (aa, c), ls in lab_of.items():
                if aa == a:
                    d[int(np.bincount(ls).argmax())].append(c)
            self.clips_by_cls[a] = {k: sorted(v) for k, v in d.items()}
        self.n_batches = max(1, len(meta) // (self.K * self.M))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 100003 + self.epoch)
        self.epoch += 1                                               # 다음 epoch 은 다른 배치
        for _ in range(self.n_batches):
            a = self.sessions[rng.integers(len(self.sessions))]
            subs = list(rng.choice(self.subj_of[a], size=self.K, replace=False))
            cls_avail = sorted(self.clips_by_cls[a])
            per = [self.M // len(cls_avail)] * len(cls_avail)
            for j in rng.choice(len(cls_avail), size=self.M - sum(per), replace=False):
                per[j] += 1                                          # M 이 감정 수로 안 나뉘면 남는 자리를 무작위로
            pos = []
            for k, n in zip(cls_avail, per):
                for c in rng.choice(self.clips_by_cls[a][k], size=n, replace=True):
                    L = min(len(self.idx.get((s, a, int(c)), [])) for s in subs)
                    if L == 0:
                        continue
                    pos.append((int(c), int(rng.integers(L))))
            if len(pos) < self.M:                                    # 드물게 빈 자리 — 있는 위치로 채운다
                pos += [pos[i % len(pos)] for i in range(self.M - len(pos))]
            yield [self.idx[(s, a, c)][t] for s in subs for (c, t) in pos]


def in_batch_center(z: torch.Tensor, K: int, M: int) -> torch.Tensor:
    """피험자 우선 배치 (K × M) 에서 피험자마다 자기 M 창 평균을 뺀다."""
    zz = z.view(K, M, -1)
    return (zz - zz.mean(dim=1, keepdim=True)).view(K * M, -1)


def stim_align_loss(zc: torch.Tensor, K: int, M: int) -> torch.Tensor:
    """같은 (클립, 시점) 위치의 K 개 중심화 특징의 평균 쌍별 (1 − 코사인)."""
    zn = F.normalize(zc.float().view(K, M, -1), dim=-1)
    m = zn.mean(dim=0)                                               # (M, d)
    mean_cos = (K * (m * m).sum(-1) - 1.0) / (K - 1)                 # 위치별 평균 쌍별 코사인
    return (1.0 - mean_cos).mean()
