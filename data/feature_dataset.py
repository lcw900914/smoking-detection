"""離線特徵 dataset(階段一訓練用)。

讀 extract_features.py 輸出:每 clip 一個
    {feature_root}/{clip_id}.npy   — fp16, shape (N, C, H', W')
    {feature_root}/{clip_id}.json  — 標籤(由 label.json 精簡而來)

回傳已疊合的 channel-buffer 張量(與推理端 RingBuffer.get() layout
完全一致——共用 models.ring_buffer.stack_time_to_channels)。

亦支援讀取 distill_precompute.py 輸出的 teacher soft labels / 特徵
(kd_root 不為 None 時),供知識蒸餾訓練。
"""
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from models.ring_buffer import stack_time_to_channels
from data.dataset import window_indices


class FeatureClipDataset(Dataset):
    """離線特徵 clip dataset。

    Args:
        feature_root: extract_features.py 輸出目錄
        short_T / long_T / long_stride: 視窗設定(需與 model.yaml 一致)
        train: 訓練模式(隨機 anchor);否則固定取最後一幀
        samples_per_clip: 訓練時每 clip 每 epoch 取樣次數
        kd_root: teacher 蒸餾資料目錄(None 表示不用 KD)
    """

    def __init__(self, feature_root: str, short_T: int = 16,
                 long_T: int = 16, long_stride: int = 8,
                 train: bool = True, samples_per_clip: int = 4,
                 kd_root: Optional[str] = None):
        self.root = Path(feature_root)
        self.short_T, self.long_T = short_T, long_T
        self.long_stride = long_stride
        self.train = train
        self.samples_per_clip = samples_per_clip if train else 1
        self.kd_root = Path(kd_root) if kd_root else None

        self.metas: List[dict] = []
        for meta_file in sorted(self.root.glob("*.json")):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            npy = meta_file.with_suffix(".npy")
            if not npy.exists():
                continue
            meta["npy"] = npy
            self.metas.append(meta)
        if not self.metas:
            raise RuntimeError(f"{feature_root} 下找不到特徵檔")

    def __len__(self) -> int:
        return len(self.metas) * self.samples_per_clip

    def __getitem__(self, i: int):
        meta = self.metas[i % len(self.metas)]
        # mmap 讀取,只載入需要的幀,節省記憶體
        feats = np.load(meta["npy"], mmap_mode="r")  # (N, C, H, W) fp16
        n = feats.shape[0]

        if self.train:
            anchor = random.randint(min(self.short_T - 1, n - 1), n - 1)
        else:
            anchor = n - 1

        short_idx = window_indices(anchor, self.short_T, 1)
        long_idx = window_indices(anchor, self.long_T, self.long_stride)

        short_feats = torch.from_numpy(
            np.ascontiguousarray(feats[short_idx])).float()  # (T, C, H, W)
        long_feats = torch.from_numpy(
            np.ascontiguousarray(feats[long_idx])).float()

        # 與推理端 RingBuffer 完全一致的疊合 layout
        short_stacked = stack_time_to_channels(short_feats)  # (T*C, H, W)
        long_stacked = stack_time_to_channels(long_feats)

        stage_ids = meta["stage_ids"]
        stage_seq = torch.tensor([stage_ids[j] for j in short_idx],
                                 dtype=torch.long)
        stage_label = torch.mode(stage_seq).values

        is_smoking = 1.0 if meta["label"] == "smoking" else 0.0
        sample = {
            "short_feats": short_stacked,
            "long_feats": long_stacked,
            "stage_seq": stage_seq,
            "stage_label": stage_label,
            "clip_label": torch.tensor(is_smoking),
            "cycle_label": torch.tensor(is_smoking),
            "clip_id": meta["clip_id"],
        }

        # ---------- KD:讀 teacher 預計算輸出 ----------
        if self.kd_root is not None:
            t_logit = self.kd_root / f"{meta['clip_id']}_logits.npy"
            t_feat = self.kd_root / f"{meta['clip_id']}_feat.npy"
            if t_logit.exists():
                sample["teacher_logits"] = torch.from_numpy(
                    np.load(t_logit)).float()
            if t_feat.exists():
                sample["teacher_feat"] = torch.from_numpy(
                    np.load(t_feat)).float()
        return sample
