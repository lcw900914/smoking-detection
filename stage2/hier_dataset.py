"""兩層模型的資料集。

L1(PrimitiveDataset)—— 逐幀基元,標籤由規則產生,不需要人標。
    增強一律作用在**原始關鍵點**上,增強完才重算運動學與偽標籤。
    順序反過來(先算標籤再增強)會讓標籤跟畫面對不上:把序列放慢
    1.2 倍之後,「舉手」的幀位置就變了。

L2(CompositionDataset)—— 片段序列 + 片段級人工標籤。
    片段可以來自規則,也可以來自訓練好的 L1;兩者輸出同一種 token,
    所以 L2 的程式碼不必知道自己吃的是哪一種。
"""
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from stage2.composition import STAT_DIM, analyze, normalize_stats
from stage2.kinematics import graph_features, kinematic_features
from stage2.primitives import IGNORE, rule_primitives_both
from stage2.taxonomy import deep_index

# COCO 17 點的左右對調索引(增強用;原始關鍵點仍是 17 點)
FLIP17 = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def load_pose_items(pose_dir: str = "annotations/pose",
                    labels_path: Optional[str] = None,
                    min_valid: float = 0.0) -> List[dict]:
    """讀回抽出來的節點序列;有標籤檔時一併帶上片段級標籤。"""
    labels = {}
    if labels_path and Path(labels_path).exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)["labels"]
    items = []
    for npz in sorted(Path(pose_dir).glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        if float(d["valid"].mean()) < min_valid:
            continue
        clip = str(d["clip"])
        items.append({
            "kpts": d["kpts"].astype(np.float32),
            "fps": float(d["fps"]) or 10.0,
            "clip": clip,
            "stem": npz.stem,
            "label": (labels.get(clip) or {}).get("label"),
        })
    if not items:
        raise RuntimeError(f"{pose_dir} 沒有可用的節點序列 "
                           f"(先跑 python -m stage2.extract_pose)")
    return items


def augment_kpts(kpts: np.ndarray, rng: random.Random) -> np.ndarray:
    """原始關鍵點空間的增強(水平翻轉 / 噪聲 / 關節遮擋 / 變速)。"""
    k = kpts.copy()
    if rng.random() < 0.5:                       # 水平翻轉
        vis = k[:, :, 2] > 0.1
        if vis.any():
            cx = float(k[:, :, 0][vis].mean())
            k = k[:, FLIP17]
            k[:, :, 0] = 2 * cx - k[:, :, 0]
    if rng.random() < 0.8:                       # 關鍵點噪聲(像素級)
        k[:, :, :2] += np.random.normal(0, 2.0, k[:, :, :2].shape)
    if rng.random() < 0.3:                       # 隨機關節遮擋
        joints = np.random.choice(17, size=rng.randint(1, 3),
                                  replace=False)
        k[:, joints, 2] = 0.0
    if rng.random() < 0.5:                       # 變速 0.8–1.2×
        T = k.shape[0]
        idx = np.clip(np.arange(T) * rng.uniform(0.8, 1.2),
                      0, T - 1).astype(int)
        k = k[idx]
    return k.astype(np.float32)


class PrimitiveDataset(Dataset):
    """L1:骨架 → 逐幀基元(規則偽標籤)。

    回傳 (graph (5,T,13), kin (T,45), labels (T,2))。
    labels 的 −1 代表規則沒把握,訓練時以 ignore_index 略過。
    """

    def __init__(self, items: List[dict], window: int = 128,
                 train: bool = True, seed: int = 0):
        self.items = items
        self.window = window
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def _crop(self, *arrays):
        T = arrays[0].shape[0]
        if T <= self.window:
            pad = self.window - T
            return [np.concatenate(
                [a, np.repeat(a[-1:], pad, axis=0)], axis=0)
                if pad else a for a in arrays]
        s = (self.rng.randint(0, T - self.window) if self.train
             else (T - self.window) // 2)
        return [a[s:s + self.window] for a in arrays]

    def __getitem__(self, i):
        it = self.items[i]
        kpts = augment_kpts(it["kpts"], self.rng) if self.train \
            else it["kpts"]
        fps = it["fps"]
        g = graph_features(kpts)                       # (T,13,5)
        kin = kinematic_features(kpts, fps)            # (T,45)
        lab = rule_primitives_both(kin, fps)           # (T,2)
        g, kin, lab = self._crop(g, kin, lab)
        return (torch.from_numpy(g).permute(2, 0, 1).contiguous(),
                torch.from_numpy(kin),
                torch.from_numpy(lab.astype(np.int64)))


class CompositionDataset(Dataset):
    """L2:片段序列 → 具體動作。

    primitives_fn 給定時用它產生逐幀基元(通常是包好的 L1),
    否則走規則路徑。兩者的 token 版面完全相同。
    """

    def __init__(self, items: List[dict], primitives_fn=None,
                 embed_fn=None, embed_dim: int = 0,
                 require_label: bool = True):
        self.samples = []
        for it in items:
            y = deep_index(it["label"]) if it["label"] else None
            if require_label and y is None:
                continue
            kin = kinematic_features(it["kpts"], it["fps"])
            prim = primitives_fn(it) if primitives_fn else None
            emb = embed_fn(it) if embed_fn else None
            a = analyze(kin, it["fps"], prim=prim, frame_embed=emb,
                        embed_dim=embed_dim)
            self.samples.append({
                "tokens": a.tokens, "times": a.times,
                "stats": normalize_stats(a.stats),
                "label": y, "clip": it["clip"], "stem": it["stem"],
                "analysis": a,
            })
        if require_label and not self.samples:
            raise RuntimeError(
                "沒有任何片段對得上深層類別。用 scripts/label_tool.py "
                "重標(見 stage2/taxonomy.py 的 DEEP_MERGE)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return (torch.from_numpy(s["tokens"]),
                torch.from_numpy(s["times"]),
                torch.from_numpy(s["stats"]),
                torch.tensor(s["label"] if s["label"] is not None else -1,
                             dtype=torch.long))


def collate_tokens(batch):
    """變長片段序列的補齊:回傳 tokens, times, mask, stats, y。"""
    n = max(b[0].shape[0] for b in batch)
    d = batch[0][0].shape[1]
    B = len(batch)
    tokens = torch.zeros(B, n, d)
    times = torch.zeros(B, n)
    mask = torch.zeros(B, n, dtype=torch.bool)
    stats = torch.zeros(B, STAT_DIM)
    y = torch.zeros(B, dtype=torch.long)
    for i, (tk, tm, st, yy) in enumerate(batch):
        k = tk.shape[0]
        tokens[i, :k], times[i, :k], mask[i, :k] = tk, tm, True
        stats[i], y[i] = st, yy
    return tokens, times, mask, stats, y


def primitive_label_stats(items: List[dict]) -> dict:
    """規則偽標籤的覆蓋率與類別分佈(訓練前先看一眼健不健康)。"""
    from collections import Counter
    c = Counter()
    for it in items:
        kin = kinematic_features(it["kpts"], it["fps"])
        lab = rule_primitives_both(kin, it["fps"])
        for v in range(IGNORE, 5):
            c[v] += int((lab == v).sum())
    return dict(c)
