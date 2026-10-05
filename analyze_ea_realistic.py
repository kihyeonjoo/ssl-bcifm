"""
분석 B: EA 까지 현실적 캘리브레이션으로 — 최종 방법의 현실적 성능

S8 분석 2 는 **중심화만** 현실적 제약에 넣고 EA 는 세션 전체로 추정한 채 뒀다.
그래서 "조건4 = 0.7293" 은 아직 반쯤 transductive 다.  여기서는 테스트 피험자의
**diag EA R̄ 와 특징 중심화 μ 를 같은 캘리브레이션 창에서만** 계산한다.

  EA 가 바뀌면 백본 입력이 바뀌므로 특징을 다시 뽑아야 한다 — 테스트 피험자의
  창만 GPU 로 다시 통과시킨다.  학습 피험자 쪽 EA·prototype·μ_train 은 그대로이므로
  `cache_ft/` 를 재사용한다 (EA 는 도메인별이라 테스트 피험자의 추정이 학습
  피험자의 특징에 영향을 주지 않는다).

프로토콜은 S8 분석 2 와 같다: 각 테스트 세션에서 감정별 1클립씩 떼고(SEED 3,
SEED-V 5), 남은 클립으로만 평가한다.  캘리브레이션 창은 떼어낸 클립에서만 나온다.

조건: 감정별 20초 / 40초 / 감정별 1클립 전체.  반복 10회 (S8 은 20회였고 같은 rng 시드의
앞 10개를 쓴다 — 반복은 클립 선택 잡음만 줄인다).
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

from analyze_centering import (N_CLS, FS, SEG, roles_for_fold, clip_index, l2,
                               load_head, bootstrap_ci, head_probs)
from analyze_calib_protocol import _train_side, _score, extra_metrics
from calib_common import (calib_indices, clips_by_domain_label, cond_key,
                          draw_holdout, parse_cond)
from analyze_evidence import decide_head, decide_proto
from analyze_evidence import choose as choose_tau

from aggregate_logits import mde_from
from dataset_config import get as _cfg_root
CFG = _cfg_root()
ROOT = _cfg_root().root
SEC_PER_WIN = SEG / FS
EA_KW = dict(ea=True, ea_mode="diag", ea_scope="session", ea_scale=0.2,
             ea_trim=0.05, ea_eps=1e-6)


@torch.no_grad()
def forward_cls(model, ds, device, batch_size=128):
    """Raw CLS for every segment of ``ds``, in dataset order (DataLoader path).

    Kept for the verification step, which must go through the dataset's own
    __getitem__ to prove the fast path below reproduces it."""
    from torch.utils.data import DataLoader
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                    pin_memory=(device.type == "cuda"))
    out = []
    for b in ld:
        eeg = b["eeg"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            allt = model.labram.forward_features(
                eeg, input_chans=model.input_chans, return_all_tokens=True)
        out.append(allt[:, 0].float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


class FastEncoder:
    """Re-encode one subject under many EA fits without touching the CPU.

    The DataLoader path costs ~8 s per pass when the machine is loaded, and this
    analysis needs hundreds of passes.  Two facts make it avoidable:

      * **diag EA is a per-channel scaling.**  ``W = scale * diag(d**-0.5)`` is
        diagonal, so ``W @ X`` is an elementwise multiply by a (62,) vector —
        verified against the dataset's own output to 0.0.
      * ``norm`` is skipped when ``ea=True``, so nothing else touches the signal.

    So the raw segments go to the GPU once and each EA fit is one broadcast
    multiply there.
    """

    def __init__(self, ds, device, batch_size=128):
        raw = np.stack([s[0] for s in ds._segments]).astype(np.float32)
        self.X = torch.from_numpy(raw).to(device)        # (n, 62, T)
        self.n, self.C, self.T = self.X.shape
        self.device = device
        self.batch_size = batch_size
        self.n_patches = self.T // ds.patch_size
        self.patch_size = ds.patch_size
        keys = sorted(set(ds._seg_group))
        self.key_of = {k: i for i, k in enumerate(keys)}
        self.keys = keys
        self.gidx = torch.as_tensor(
            [self.key_of[k] for k in ds._seg_group], device=device)

    def scale_from(self, transforms):
        """{group key: (62,62) W} -> (n_groups, 62) per-channel factors."""
        S = np.stack([np.diag(transforms[k]) for k in self.keys])
        return torch.from_numpy(S.astype(np.float32)).to(self.device)

    @torch.no_grad()
    def encode(self, model, transforms):
        S = self.scale_from(transforms)[self.gidx]       # (n, 62)
        out = []
        for i in range(0, self.n, self.batch_size):
            x = self.X[i:i + self.batch_size] * S[i:i + self.batch_size, :, None]
            x = x.reshape(-1, self.C, self.n_patches, self.patch_size)
            with torch.autocast(device_type=self.device.type,
                                dtype=torch.bfloat16,
                                enabled=(self.device.type == "cuda")):
                allt = model.labram.forward_features(
                    x, input_chans=model.input_chans, return_all_tokens=True)
            out.append(allt[:, 0].float().cpu())
        return torch.cat(out).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea_cache", default=CFG.cache_dir)
    ap.add_argument("--ckpt_dir", default="checkpoints_s0")
    ap.add_argument("--labram_repo", default="/home/kihyeonjoo/LaBraM")
    ap.add_argument("--labram_ckpt",
                    default="/home/kihyeonjoo/LaBraM/checkpoints/labram-base.pth")
    ap.add_argument("--seconds", type=float, nargs="+", default=[20, 40])
    ap.add_argument("--modes", nargs="+", default=["prefix", "spread"],
                    help="캘리브레이션 창을 클립 앞부분에서 뽑을지, 클립 전체에 "
                         "고르게 뽑을지.  'full' 조건은 둘이 같으므로 한 번만 돈다.")
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"results/{CFG.prefix('ea_realistic')}.npz")
    # Both reference runs must use the SAME number of repeats as this one, or a
    # difference in the clip draws is read as a difference between methods.
    ap.add_argument("--ref_ea", default=f"results/{CFG.prefix('calib_protocol_r10')}.npz",
                    help="EA transductive + realistic centring, same n_rep")
    ap.add_argument("--ref_noea", default=f"results/{CFG.prefix('calib_protocol_noea_r10')}.npz",
                    help="no EA + realistic centring, same n_rep")
    args = ap.parse_args()

    device = torch.device(args.device)
    from data.seed_raw_dataset import SEEDRawDataset
    from data.seedv_raw_dataset import SEEDVRawDataset
    from diagnose_prototypes import build_model
    # 로더는 설정에서.  SEEDRawDataset 을 박아 두면 SEED-V 의 .cnt 를 못 읽는다.
    DS = {'SEEDRawDataset': SEEDRawDataset,
          'SEEDVRawDataset': SEEDVRawDataset}[_cfg_root().loader]

    R = defaultdict(lambda: defaultdict(list))
    # one entry per distinct EA fit: (seconds, mode).  'full' appears once
    # because a whole clip is the same under either mode.
    specs = [(sec, m) for m in args.modes for sec in args.seconds]
    specs.append(("full", "prefix"))
    conds = [cond_key(sec, m) for sec, m in specs]
    print(f"[조건] {conds}  (EA 재적합 {len(specs)}회/반복)", flush=True)

    for ts in _cfg_root().subjects:
        # one load per test subject: EA is per (subject, session), so fitting on
        # this subject alone gives the same full-session transform the caching
        # run produced (verified below against cache_ft on the first subject).
        ds = DS(root=ROOT, subjects=[ts], sessions=_cfg_root().sessions,
                segment_length=_cfg_root().seg, step=_cfg_root().seg,
                patch_size=200, norm="scale100", **EA_KW)
        full_groups = {k: list(v) for k, v in
                       _by_group(ds._seg_group).items()}
        meta_local = np.array([[m[0], m[1], m[2]] for m in ds._seg_meta], int)
        lab_local = np.array([s[1] for s in ds._segments], int)
        cidx_local = clip_index(meta_local)

        by = clips_by_domain_label(cidx_local, lab_local, ts)
        te_doms = sorted({(k[0], k[1]) for k in cidx_local})
        enc = FastEncoder(ds, device)

        for sd in args.seeds:
            ck_path = os.path.join(args.ckpt_dir, f"{CFG.ckpt_prefix}S{ts}_seed{sd}.pt")
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            model = build_model(args.labram_repo, args.labram_ckpt, device,
                               ck["model"])
            head = load_head(ck_path)

            # train side from the cache (unchanged by the test subject's EA)
            z = np.load(os.path.join(args.ea_cache, f"S{ts}_seed{sd}.npz"))
            cls_c, meta_c, lab_c = (z["cls"], z["meta"].astype(int),
                                    z["lab"].astype(int))
            train, _, _ = roles_for_fold(ts)
            mu_tr_global, P = _train_side(cls_c, meta_c, lab_c, train)

            if ts == 1 and sd == args.seeds[0]:
                _verify_against_cache(model, ds, enc, device, cls_c,
                                      meta_c, lab_c, head, ts)

            # τ 는 이 fold 의 검증 피험자 2명으로 고른다.  선택은 캐시(EA
            # transductive)의 특징으로 하는데, τ 가 재는 것은 클립 안의 시간
            # 경과이고 그것은 EA 추정 방식에 거의 의존하지 않기 때문이다.
            hp_tw, _, _ = choose_tau(cls_c, l2(cls_c), meta_c, lab_c,
                                     clip_index(meta_c), head, ts, [20, 40],
                                     "proto")
            tau = hp_tw["tau"]
            rng = np.random.default_rng(1000 * ts + sd)   # same draws as S8
            for _ in range(args.n_rep):
                held, held_set = draw_holdout(by, te_doms, rng)
                eval_keys = [k for k in sorted(cidx_local)
                             if k not in held_set]
                y_c = np.array([lab_local[cidx_local[k][0]] for k in eval_keys])

                for (sec, mode), cond in zip(specs, conds):
                    calib = calib_indices(held, cidx_local, te_doms, sec, mode)
                    # refit EA from the calibration windows only, then re-encode
                    ds.refit_alignment_explicit(
                        {d: calib[d] for d in te_doms},
                        scale=EA_KW["ea_scale"], trim=EA_KW["ea_trim"],
                        eps=EA_KW["ea_eps"])
                    cls_new = enc.encode(model, ds._aligner.transforms)
                    Z = l2(cls_new)
                    mu_raw = {d: cls_new[calib[d]].mean(0) for d in te_doms}
                    mu_z = {d: Z[calib[d]].mean(0) for d in te_doms}
                    r = _score(cls_new, Z, head, cidx_local, eval_keys, y_c,
                               mu_raw, mu_z, P, mu_tr_global, centred=True)
                    for m, v in r.items():
                        R[f"eaReal_{cond}|{m}"][ts].append(v)
                    # + 결정 단계 시간 가중 (τ 는 검증 피험자로 고른 값).
                    # 중심화·EA 는 그대로이고 클립 집계만 바뀐다.
                    Pn = l2(P)
                    ph_p, pp_p = [], []
                    for i2, k2 in enumerate(eval_keys):
                        w2 = cidx_local[k2]
                        d2 = (k2[0], k2[1])
                        ph = head_probs(head, cls_new[w2] - mu_raw[d2]
                                        + mu_tr_global)
                        A2 = Z[w2] - mu_z[d2]
                        sim2 = l2(A2) @ Pn.T
                        ph_p.append(decide_head(ph, "time", tau, 0.3))
                        pp_p.append(decide_proto(A2, sim2, Pn, "time", tau, 0.3))
                    ph_p, pp_p = np.array(ph_p), np.array(pp_p)
                    R[f"eaRealTW_{cond}|head_clip"][ts].append(
                        float((ph_p == y_c).mean()))
                    R[f"eaRealTW_{cond}|proto_clip"][ts].append(
                        float((pp_p == y_c).mean()))
                    for k3, v3 in extra_metrics(ph_p, y_c, "head_clip").items():
                        R[f"eaRealTW_{cond}|{k3}"][ts].append(v3)
                    for k3, v3 in extra_metrics(pp_p, y_c, "proto_clip").items():
                        R[f"eaRealTW_{cond}|{k3}"][ts].append(v3)
                    # conditions 1/3: no centring, but still THIS EA fit.  The
                    # calibration recording buys the EA transform even when the
                    # feature mean is not used, so this is the honest "EA only"
                    # baseline and it needs no extra forward pass.
                    r0 = _score(cls_new, Z, head, cidx_local, eval_keys, y_c,
                                None, None, P, mu_tr_global, centred=False)
                    for m, v in r0.items():
                        R[f"eaRealNoCen_{cond}|{m}"][ts].append(v)
                print(f"  S{ts} seed{sd} tau={tau:g}  " + "  ".join(
                    f"{c} {np.mean(R[f'eaReal_{c}|proto_clip'][ts]):.3f}"
                    f"/{np.mean(R[f'eaRealTW_{c}|proto_clip'][ts]):.3f}"
                    for c in conds), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def arr(key):
        return np.array([np.mean(R[key][s]) for s in sorted(R[key])])

    np.savez(args.out, **{k.replace("|", "__"): arr(k) for k in R})
    print(f"\n[저장] {args.out}")
    _report(arr, conds, args)


def _by_group(seg_group):
    d = defaultdict(list)
    for i, k in enumerate(seg_group):
        d[k].append(i)
    return d


def _verify_against_cache(model, ds, enc, device, cls_c, meta_c, lab_c, head, ts):
    """The full-session EA fit on this subject alone must reproduce the cache.

    If it does not, the transform this script installs is not the one the
    checkpoint was trained under and every number below is meaningless.

    The check is on the QUANTITIES THIS SCRIPT USES, not on raw activations.
    Re-running the backbone in bf16 with a different batch layout perturbs a
    few large activations by ~1 (max over 500k elements), while the median
    element is bit-identical — judging that by ``max|Δ| / mean|x|`` fails a
    correct setup.  So compare the four conditions instead, which is what the
    numbers below are made of.
    """
    from analyze_centering import four_conditions
    got = forward_cls(model, ds, device)
    fast = enc.encode(model, ds._aligner.transforms)
    fd = float(np.abs(got - fast).max())
    print(f"  [검증] 빠른 경로 vs DataLoader 경로 max {fd:.6f}", flush=True)
    if fd > 1e-3:
        raise SystemExit(f"검증 실패: 빠른 경로가 {fd:.5f} 어긋난다")
    sel = meta_c[:, 0] == ts
    want = cls_c[sel]
    if got.shape != want.shape:
        raise SystemExit(f"검증 실패: 모양 {got.shape} vs 캐시 {want.shape}")

    d = np.abs(got - want)
    swapped = cls_c.copy()
    swapped[sel] = got
    a = four_conditions(cls_c, meta_c, lab_c, head, ts)
    b = four_conditions(swapped, meta_c, lab_c, head, ts)
    worst = max(max(abs(a[k][0] - b[k][0]), abs(a[k][1] - b[k][1]))
                for k in (1, 2, 3, 4))
    print(f"  [검증] 특징 차이 중앙값 {np.median(d):.2e} / 99% {np.percentile(d,99):.3f}"
          f" / max {d.max():.3f};  조건1~4 최대 차이 {worst:.5f}", flush=True)
    if worst > 0.005:
        raise SystemExit(f"검증 실패: 조건값이 {worst:.4f} 어긋난다 — EA 설정 불일치")


def _report(arr, conds, args):
    from analyze_centering import bootstrap_ci
    W = 100
    try:
        S8 = np.load(args.ref_ea)
    except FileNotFoundError:
        S8 = None
        print(f"  [경고] {args.ref_ea} 없음 — (3) 비교 불가")
    try:
        NOEA = np.load(args.ref_noea)
    except FileNotFoundError:
        NOEA = None
        print(f"  [경고] {args.ref_noea} 없음 — (2) 비교 불가")

    for metric, nm in (("proto_clip", "조건4 prototype  clip"),
                       ("proto_win", "조건4 prototype  window"),
                       ("head_clip", "조건2 head  clip"),
                       ("head_win", "조건2 head  window")):
        print(f"\n{'='*W}\n{nm}   n={CFG.n_subjects}, 시드 평균 x 반복 {args.n_rep}\n{'='*W}")
        print(f"  {'캘리브레이션':<14}{'EA+중심화':>10}{'EA만(중심화X)':>12}"
              f"{'EA전체+중심화':>12}{'EA현실화 손실':>14}{'EA없음+중심화':>10}")
        for c in conds:
            a = arr(f"eaReal_{c}|{metric}")
            nc = arr(f"eaRealNoCen_{c}|{metric}")
            s8 = S8[f"{c}__{metric}"].mean() if S8 is not None and \
                f"{c}__{metric}" in S8 else np.nan
            no = NOEA[f"{c}__{metric}"].mean() if NOEA is not None and \
                f"{c}__{metric}" in NOEA else np.nan
            print(f"  {c:<14}{a.mean():>10.4f}{nc.mean():>12.4f}"
                  f"{s8:>12.4f}{a.mean()-s8:>+14.4f}{no:>10.4f}")

    # SEED 는 A팔 확정값(0.0386), 다른 데이터셋은 그 비교의 짝지은 차이에서.
    TH = CFG.mde
    print(f"\n{'='*W}\n(2) 같은 조건에서 EA 있음 vs 없음  [재학습 비교 — 감지 한계 "
          + (f"{TH:+.4f} 병기]" if TH is not None
             else "는 비교마다 짝지은 차이에서 계산]") + f"\n{'='*W}")
    if NOEA is None:
        print(f"  {args.ref_noea} 가 없다 — 먼저 만들 것")
    else:
        for metric in ("proto_clip", "proto_win", "head_clip", "head_win"):
            for c in conds:
                k = f"{c}__{metric}"
                if k not in NOEA:
                    continue
                a, b = arr(f"eaReal_{c}|{metric}"), NOEA[k]
                d = a - b
                lo, hi = bootstrap_ci(d)
                p = stats.wilcoxon(a, b).pvalue
                _th = TH if TH is not None else mde_from(d)
                mark = (f"점추정 한계({_th:+.4f}) "
                        + ("넘음" if d.mean() > _th else "아래"))
                print(f"  {metric:<11}{c:<8} Δ{d.mean():+.4f} "
                      f"CI[{lo:+.4f},{hi:+.4f}]"
                      f"{'  0포함' if lo <= 0 <= hi else '       '} "
                      f"p={p:.4f} {int((d>0).sum())}/{len(d)}  [{mark}]")

    print(f"\n{'='*W}\n(3) EA 를 transductive -> 현실적으로 바꿀 때 잃는 양\n{'='*W}")
    if S8 is None:
        print(f"  {args.ref_ea} 가 없다")
    else:
        for metric in ("proto_clip", "proto_win", "head_clip", "head_win"):
            for c in conds:
                k = f"{c}__{metric}"
                if k not in S8:
                    continue
                a, b = arr(f"eaReal_{c}|{metric}"), S8[k]
                d = a - b
                lo, hi = bootstrap_ci(d)
                p = stats.wilcoxon(a, b).pvalue
                print(f"  {metric:<11}{c:<8} Δ{d.mean():+.4f} "
                      f"CI[{lo:+.4f},{hi:+.4f}]"
                      f"{'  0포함' if lo <= 0 <= hi else '       '} "
                      f"p={p:.4f} {int((d>0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
