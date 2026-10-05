"""
Cache the fine-tuned features once, RAW, so later analyses need no GPU.

``diagnose_prototypes.py`` L2-normalises as it extracts, which is right for
cosine prototypes and wrong for anything that feeds the classifier head — the
head was trained on the unnormalised CLS token and its LayerNorm expects that
scale.  This writes the raw vectors; a consumer normalises only where it needs
to.

One .npz per checkpoint so a died run resumes, and so a later analysis can load
just the folds it needs.

    cls  (n_seg, 200)  raw CLS token
    pat  (n_seg, 200)  raw mean over patch tokens
    meta (n_seg, 3)    subject, session, clip
    lab  (n_seg,)
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import torch

from data.seed_raw_dataset import SEEDRawDataset
from data.seedv_raw_dataset import SEEDVRawDataset
from data.seediv_raw_dataset import SEEDIVRawDataset
from data.deap_raw_dataset import DEAPRawDataset

from dataset_config import get as _cfg_root

# 데이터셋은 환경변수 DATASET 으로 고른다 (dataset_config).  로더를 여기서
# 박아 두면 SEED-V 체크포인트에 SEED 입력을 먹여도 조용히 돌아간다.
_LOADERS = {"SEEDRawDataset": SEEDRawDataset,
            "SEEDVRawDataset": SEEDVRawDataset,
            "SEEDIVRawDataset": SEEDIVRawDataset, "DEAPRawDataset": DEAPRawDataset}
CFG = _cfg_root()
ROOT = CFG.root


@torch.no_grad()
def extract_raw(model, ds, device, batch_size=128, amp=True):
    from torch.utils.data import DataLoader
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=(device.type == "cuda"))
    CLS, PAT, meta, lab = [], [], [], []
    for b in ld:
        eeg = b["eeg"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp and device.type == "cuda"):
            allt = model.labram.forward_features(
                eeg, input_chans=model.input_chans, return_all_tokens=True)
        allt = allt.float()
        CLS.append(allt[:, 0].cpu())
        PAT.append(allt[:, 1:].mean(1).cpu())
        meta.append(torch.stack([b["subject"], b["session"], b["clip"]], 1))
        lab.append(b["label"])
    return (torch.cat(CLS).numpy().astype(np.float32),
            torch.cat(PAT).numpy().astype(np.float32),
            torch.cat(meta).numpy().astype(np.int16),
            torch.cat(lab).numpy().astype(np.int8))


def main():
    ap = argparse.ArgumentParser()
    # 기본값은 --no_ea 에 따라 바뀐다 (아래).  None 으로 두고 뒤에서 채운다 —
    # 여기서 EA 팔 경로를 박아 두면 --no_ea 만 주고 경로를 깜빡했을 때
    # **EA 로 학습한 체크포인트에 EA 없는 입력**을 먹이게 되고, 그래도 돌아간다.
    ap.add_argument("--ckpt_glob", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--labram_repo", default="/home/kihyeonjoo/LaBraM")
    ap.add_argument("--labram_ckpt",
                    default="/home/kihyeonjoo/LaBraM/checkpoints/labram-base.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--no_ea", action="store_true",
                    help="extract WITHOUT Euclidean Alignment. The features must "
                         "be produced by the same input pipeline the checkpoint "
                         "was trained on, so a no-EA arm needs this flag.")
    ap.add_argument("--fp32", action="store_true",
                    help="extract in TRUE fp32: no bf16 autocast and TF32 off "
                         "(cuDNN convolutions default to TF32 on Ampere/Ada). "
                         "Writes to a separate *_fp32 directory only.")
    args = ap.parse_args()

    # 2026-10-04: SEED 모델의 CLS 특징이 bf16 정밀도에 민감하다는 것을 찾았다
    # (같은 입력에서 fp32 vs bf16 코사인 < 0.9 인 창 21%).  기존 캐시는 학습·평가
    # 때와 같은 bf16 이므로 그대로 두고, fp32 는 별도 폴더에만 쓴다.
    if args.fp32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    # 체크포인트와 입력 파이프라인은 반드시 같은 팔이어야 한다.
    pre = CFG.noea_ckpt_prefix if args.no_ea else CFG.ckpt_prefix
    if args.ckpt_glob is None:
        args.ckpt_glob = f"checkpoints_s0/{pre}S*_seed*.pt"
    if args.out_dir is None:
        args.out_dir = (CFG.cache_dir + ("_noea" if args.no_ea else "")
                        + ("_fp32" if args.fp32 else ""))
    if args.fp32 != args.out_dir.rstrip("/").endswith("_fp32"):
        raise SystemExit(
            f"--fp32={args.fp32} 인데 출력 폴더가 {args.out_dir!r} 다.  fp32 특징은 "
            f"*_fp32 폴더에만, bf16 특징은 그 밖에만 쓴다 — 두 정밀도가 한 폴더에 "
            f"섞이면 어느 분석이 어느 정밀도인지 알 수 없다.")
    if pre not in args.ckpt_glob:
        raise SystemExit(
            f"--no_ea={args.no_ea} 인데 체크포인트 glob 이 {args.ckpt_glob!r} 다. "
            f"이 팔은 접두사 {pre!r} 를 써야 한다 — 다른 팔의 체크포인트에 "
            f"이 입력 파이프라인을 먹이면 조용히 틀린 특징이 나온다.")

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # The input pipeline must match the one the checkpoint was trained on, or
    # the features describe a different input than the head expects.
    ea_kw = (dict(ea=False) if args.no_ea else
             dict(ea=True, ea_mode="diag", ea_scope="session", ea_scale=0.2))
    DS = _LOADERS[CFG.loader]
    ds = DS(root=ROOT, subjects=CFG.subjects, sessions=CFG.sessions,
            segment_length=CFG.seg, step=CFG.seg, patch_size=200,
            norm="scale100", **ea_kw)
    print(f"[data] {CFG.name}: {len(ds)} segments, {CFG.n_classes} classes  "
          f"({'EA 없음 (scale100)' if args.no_ea else 'diag EA alpha=0.2'})", flush=True)

    from diagnose_prototypes import build_model
    paths = sorted(glob.glob(args.ckpt_glob))
    print(f"[ckpt] {len(paths)}", flush=True)
    for i, p in enumerate(paths, 1):
        m = re.search(r"S(\d+)_seed(\d+)", os.path.basename(p))
        ts, sd = int(m.group(1)), int(m.group(2))
        out = os.path.join(args.out_dir, f"S{ts}_seed{sd}.npz")
        if os.path.exists(out):
            print(f"  [{i}/{len(paths)}] S{ts} seed{sd}  건너뜀 (이미 있음)", flush=True)
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = build_model(args.labram_repo, args.labram_ckpt, device,
                            ck["model"], ch_names=getattr(ds, "ch_names", None))
        cls, pat, meta, lab = extract_raw(model, ds, device, args.batch_size,
                                          amp=not args.fp32)
        np.savez(out, cls=cls, pat=pat, meta=meta, lab=lab,
                 epoch=ck.get("epoch", -1), seed=sd, fold=ts,
                 precision="fp32" if args.fp32 else "bf16-autocast")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{i}/{len(paths)}] S{ts} seed{sd} (epoch {ck.get('epoch','?')}) "
              f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
