"""Phase 0/1 資料集:回抽的節點序列 + clip 標籤 → 正規化特徵視窗。

增強(僅訓練):時間裁切/變速、節點高斯噪聲、隨機關節 dropout、
水平翻轉(左右關節索引對調)——全部作用在原始 (T,17,3) 上,
再過 normalize_sequence,確保與推理前處理一致。
"""
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from stage2.normalize import normalize_sequence
from stage2.model import CLASSES

# COCO 水平翻轉的左右關節對調索引
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class PoseClipDataset(Dataset):
    """讀 annotations/pose/*.npz + annotations/clip_labels.json。

    Args:
        window: 取樣幀數(不足以重複填充,超過隨機/置中裁切)
        train: 訓練模式開增強
        exclude: 排除的標籤(預設排除 unsure)
    """

    def __init__(self, pose_dir: str = "annotations/pose",
                 labels_path: str = "annotations/clip_labels.json",
                 window: int = 128, train: bool = True,
                 exclude: tuple = ("unsure",),
                 min_valid: float = 0.3,
                 items: Optional[List[dict]] = None):
        self.window = window
        self.train = train
        if items is not None:           # 交叉驗證切分用
            self.items = items
            return

        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)["labels"]
        self.items = []
        n_low = 0
        for npz_path in sorted(Path(pose_dir).glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            clip_key = str(data["clip"])
            lab = labels.get(clip_key)
            if lab is None or lab["label"] in exclude \
                    or lab["label"] not in CLASSES:
                continue
            if data["valid"].mean() < min_valid:
                n_low += 1      # 關聯率過低:節點序列不可信,棄用
                continue
            self.items.append({
                "kpts": data["kpts"], "valid": data["valid"],
                "label": CLASSES.index(lab["label"]),
                "clip": clip_key,
            })
        if n_low:
            print(f"[dataset] 排除 {n_low} 段關聯率 <{min_valid:.0%} 的樣本")
        if not self.items:
            raise RuntimeError("沒有可用樣本(檢查回抽輸出與標籤檔)")

    def __len__(self):
        return len(self.items)

    # ---------- 增強(原始節點空間) ----------

    def _augment(self, kpts: np.ndarray) -> np.ndarray:
        kpts = kpts.copy()
        if random.random() < 0.5:                    # 水平翻轉
            kpts = kpts[:, FLIP_IDX]
            kpts[:, :, 0] *= -1                      # 正規化前先鏡射 x
            kpts[:, :, 0] += 2 * kpts[:, :, 0].mean()
        if random.random() < 0.8:                    # 節點噪聲(像素級)
            kpts[:, :, :2] += np.random.normal(0, 2.0, kpts[:, :, :2].shape)
        if random.random() < 0.3:                    # 關節 dropout(模擬遮擋)
            joints = np.random.choice(17, size=random.randint(1, 3),
                                      replace=False)
            kpts[:, joints, 2] = 0.0
        if random.random() < 0.5:                    # 變速 0.8–1.2x
            T = kpts.shape[0]
            idx = np.clip(np.arange(T) * random.uniform(0.8, 1.2),
                          0, T - 1).astype(int)
            kpts = kpts[idx]
        return kpts

    def _window(self, feats: np.ndarray) -> np.ndarray:
        T = feats.shape[0]
        if T >= self.window:
            s = random.randint(0, T - self.window) if self.train \
                else (T - self.window) // 2
            return feats[s:s + self.window]
        reps = int(np.ceil(self.window / T))
        return np.tile(feats, (reps, 1))[:self.window]

    def __getitem__(self, i):
        it = self.items[i]
        kpts = it["kpts"]
        if self.train:
            kpts = self._augment(kpts)
        feats = normalize_sequence(kpts)             # (T, 85)
        feats = self._window(feats)                  # (window, 85)
        return (torch.from_numpy(feats.astype(np.float32)),
                torch.tensor(it["label"], dtype=torch.long))
