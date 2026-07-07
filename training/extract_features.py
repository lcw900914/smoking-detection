"""離線特徵抽取:凍結 backbone 對全資料集抽特徵存 .npy(fp16)。

供階段一(train_head.py)使用——時序頭訓練完全不碰影像,
batch 可拉到數百,快速掃描時序頭超參數。

輸出(每 clip):
    {out}/{clip_id}.npy   fp16, shape (N, C, H', W')
    {out}/{clip_id}.json  {"clip_id", "label", "stage_ids": [...]}

用法:
    python -m training.extract_features --data datasets/processed/train \
        --out datasets/features/train
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.backbone import build_backbone
from utils import load_config, resolve_device, check_vram_budget
from data.dataset import SmokingClipDataset


@torch.no_grad()
def extract(data_root: str, out_root: str, model_cfg: dict,
            batch_size: int = 64, device: str = "auto") -> None:
    """對 data_root 下所有 clip 逐幀抽特徵。"""
    dev = resolve_device(device)
    backbone = build_backbone(model_cfg["backbone"]).to(dev).eval()

    # 粗估 VRAM:ResNet-18 前向 batch 64 @224 約 2GB,安全範圍內
    check_vram_budget(2.0, context="離線特徵抽取(batch 64)")

    ds = SmokingClipDataset(data_root, augment=None)  # 只借用 clip 掃描
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    for clip in tqdm(ds.clips, desc="抽取特徵"):
        n = clip["num_frames"]
        feats = []
        for s in range(0, n, batch_size):
            idxs = range(s, min(s + batch_size, n))
            imgs = torch.stack([
                torch.from_numpy(
                    ds._load_frame(clip, j, None).transpose(2, 0, 1))
                for j in idxs
            ]).float().to(dev)
            feats.append(backbone(imgs).half().cpu())
        feats = torch.cat(feats).numpy()  # (N, C, H', W') fp16

        np.save(out / f"{clip['clip_id']}.npy", feats)
        with open(out / f"{clip['clip_id']}.json", "w",
                  encoding="utf-8") as f:
            json.dump({
                "clip_id": clip["clip_id"],
                "label": clip["label"],
                "stage_ids": [fr["stage_id"] for fr in clip["frames"]],
            }, f, ensure_ascii=False)
    print(f"[extract] 完成,輸出:{out}")


def main():
    parser = argparse.ArgumentParser(description="離線特徵抽取")
    parser.add_argument("--data", required=True, help="processed clip 目錄")
    parser.add_argument("--out", required=True, help="特徵輸出目錄")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_cfg = load_config(args.model_config)
    extract(args.data, args.out, model_cfg, args.batch_size, args.device)


if __name__ == "__main__":
    main()
