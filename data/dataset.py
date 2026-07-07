"""Clip dataset:讀 preprocess.py 輸出的 jpg 序列 + 幀級階段標籤。

每個樣本包含:
- short_images: (T_s, 3, H, W) 短尺度連續幀(stride=1,時間由舊到新)
- long_images:  (T_l, 3, H, W) 長尺度取樣幀(stride=8)
- stage_seq:    (T_s,) 短視窗每幀的階段標籤
- stage_label:  () 短視窗聚合階段標籤(眾數)
- clip_label:   () 二元標籤(smoking=1)
- cycle_label:  () 週期標籤(smoking clip=1,其餘 0)

視窗取樣:以 anchor(視窗最新幀)為準往回取;不足處以最舊幀重複填充,
與推理端 RingBuffer 的填充行為一致。

增強(訓練時,整個 clip 一致以模擬追蹤偏移,而非逐幀獨立):
框抖動(±8% 平移縮放)、水平翻轉、色彩抖動。
"""
import json
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import imread

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def window_indices(anchor: int, T: int, stride: int) -> List[int]:
    """由 anchor(最新幀)往回取 T 個索引(間隔 stride),
    越界處 clamp 到 0(等價於以最舊幀重複填充)。回傳由舊到新。"""
    return [max(0, anchor - stride * (T - 1 - i)) for i in range(T)]


class ClipAugment:
    """整 clip 一致的增強:框抖動、水平翻轉、色彩抖動。

    每個 clip 抽一次隨機參數,套用到該 clip 的所有幀。
    """

    def __init__(self, bbox_jitter: float = 0.08, hflip: float = 0.5,
                 color_jitter: float = 0.3):
        self.bbox_jitter = bbox_jitter
        self.hflip = hflip
        self.color_jitter = color_jitter

    def sample_params(self) -> dict:
        """為一個 clip 抽一組增強參數。"""
        j = self.bbox_jitter
        c = self.color_jitter
        return {
            "tx": random.uniform(-j, j),       # 平移(相對邊長)
            "ty": random.uniform(-j, j),
            "scale": 1.0 + random.uniform(-j, j),  # 縮放
            "flip": random.random() < self.hflip,
            "brightness": 1.0 + random.uniform(-c, c),
            "contrast": 1.0 + random.uniform(-c, c),
        }

    def apply(self, img: np.ndarray, p: dict) -> np.ndarray:
        """對單幀套用(參數 p 由 sample_params 對整個 clip 共用)。"""
        H, W = img.shape[:2]
        # 框抖動:以仿射變換模擬 ROI 平移縮放偏移
        M = np.array([
            [p["scale"], 0, p["tx"] * W],
            [0, p["scale"], p["ty"] * H],
        ], dtype=np.float32)
        # 以影像中心為縮放錨點
        M[0, 2] += (1 - p["scale"]) * W / 2
        M[1, 2] += (1 - p["scale"]) * H / 2
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        if p["flip"]:
            img = img[:, ::-1]
        # 色彩抖動(亮度/對比)
        img = img.astype(np.float32)
        img = np.clip((img - 128) * p["contrast"] + 128 * p["brightness"],
                      0, 255).astype(np.uint8)
        return img


class SmokingClipDataset(Dataset):
    """讀 jpg 序列的 clip dataset(端到端訓練 / 特徵抽取用)。

    Args:
        root: preprocess 輸出目錄(內含多個 clip 資料夾)
        short_T / long_T / long_stride: 視窗設定(需與 model.yaml 一致)
        augment: 增強設定 dict(None 表示驗證模式,不增強)
        samples_per_clip: 訓練時每個 clip 每 epoch 取幾個隨機視窗
    """

    def __init__(self, root: str, short_T: int = 16, long_T: int = 16,
                 long_stride: int = 8, augment: Optional[dict] = None,
                 samples_per_clip: int = 1, image_size: int = 224):
        self.root = Path(root)
        self.short_T, self.long_T = short_T, long_T
        self.long_stride = long_stride
        self.image_size = image_size
        self.samples_per_clip = samples_per_clip
        self.aug = ClipAugment(**augment) if augment else None

        self.clips: List[dict] = []
        for label_file in sorted(self.root.glob("*/label.json")):
            with open(label_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta["num_frames"] < 2:
                continue
            meta["dir"] = label_file.parent
            self.clips.append(meta)
        if not self.clips:
            raise RuntimeError(f"{root} 下找不到任何 clip(label.json)")

    def __len__(self) -> int:
        return len(self.clips) * self.samples_per_clip

    def _load_frame(self, clip: dict, idx: int,
                    aug_params: Optional[dict]) -> np.ndarray:
        path = clip["dir"] / clip["frames"][idx]["file"]
        img = imread(path)  # 支援非 ASCII 路徑(Windows cv2.imread 限制)
        if img is None:
            raise RuntimeError(f"讀圖失敗:{path}")
        if img.shape[0] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        if aug_params is not None:
            img = self.aug.apply(img, aug_params)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb - _MEAN) / _STD

    def __getitem__(self, i: int):
        clip = self.clips[i % len(self.clips)]
        n = clip["num_frames"]

        # anchor:訓練隨機取、驗證取最後一幀(涵蓋最多內容)
        if self.aug is not None:
            anchor = random.randint(min(self.short_T - 1, n - 1), n - 1)
        else:
            anchor = n - 1

        short_idx = window_indices(anchor, self.short_T, 1)
        long_idx = window_indices(anchor, self.long_T, self.long_stride)

        # 整 clip 一致的增強參數
        aug_params = self.aug.sample_params() if self.aug else None

        need = sorted(set(short_idx + long_idx))
        cache = {j: self._load_frame(clip, j, aug_params) for j in need}

        short_images = torch.from_numpy(
            np.stack([cache[j].transpose(2, 0, 1) for j in short_idx]))
        long_images = torch.from_numpy(
            np.stack([cache[j].transpose(2, 0, 1) for j in long_idx]))

        stage_seq = torch.tensor(
            [clip["frames"][j]["stage_id"] for j in short_idx],
            dtype=torch.long)
        # 幀級標籤聚合:短視窗眾數作為視窗階段標籤
        stage_label = torch.mode(stage_seq).values

        is_smoking = 1.0 if clip["label"] == "smoking" else 0.0
        return {
            "short_images": short_images.float(),  # (T_s, 3, H, W)
            "long_images": long_images.float(),    # (T_l, 3, H, W)
            "stage_seq": stage_seq,                # (T_s,)
            "stage_label": stage_label,            # ()
            "clip_label": torch.tensor(is_smoking),
            "cycle_label": torch.tensor(is_smoking),
            "clip_id": clip["clip_id"],
        }
