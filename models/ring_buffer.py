"""雙時間尺度環形緩衝(channel-as-temporal-buffer 的核心資料結構)。

通道排列約定(全專案唯一真理,訓練與推理路徑都必須經過本模組的
`stack_time_to_channels`,以保證 layout 完全一致):

    (T, C, H, W) → (C*T, H, W)
    通道軸順序為 interleave:[c0t0, c0t1, ..., c0t{T-1}, c1t0, c1t1, ...]

如此 `Conv2d(C*T → C*T, kernel=1, groups=C)` 的第 i 組恰好包含
第 i 個特徵通道的 T 個時刻,等價於逐通道的 depthwise temporal conv。
"""
from collections import deque
from typing import Optional

import torch


def stack_time_to_channels(feats: torch.Tensor) -> torch.Tensor:
    """將時間軸疊進通道軸。

    Args:
        feats: (T, C, H, W) 或 (B, T, C, H, W),時間由舊到新排列。

    Returns:
        (C*T, H, W) 或 (B, C*T, H, W),通道順序為
        [c0t0, c0t1, ..., c0t{T-1}, c1t0, ...]。
    """
    if feats.dim() == 4:
        T, C, H, W = feats.shape
        # permute 後通道視角為 (C, T),reshape 攤平即得 interleave 排列
        return feats.permute(1, 0, 2, 3).reshape(C * T, H, W)
    if feats.dim() == 5:
        B, T, C, H, W = feats.shape
        return feats.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
    raise ValueError(f"預期 4 或 5 維張量,收到 {feats.dim()} 維")


class RingBuffer:
    """固定大小 FIFO 特徵緩衝。

    - `push(feat)` 依 stride 決定是否收錄(每 stride 次 push 收錄一次)
    - `get()` 回傳沿通道 concat 的 (T*C,)×H×W 張量(interleave 排列)
    - 未滿 T 幀時,以最舊有效幀重複填充在最前面(時間最舊處)
    """

    def __init__(self, T: int, C: int, H: int, W: int, stride: int = 1):
        assert T > 0 and stride > 0
        self.T, self.C, self.H, self.W = T, C, H, W
        self.stride = stride
        self._frames: deque = deque(maxlen=T)
        self._push_count = 0

    def push(self, feat: torch.Tensor) -> bool:
        """推入一幀特徵 (C, H, W)。回傳是否實際被收錄(依 stride)。"""
        assert feat.shape == (self.C, self.H, self.W), (
            f"特徵形狀 {tuple(feat.shape)} 不符 ({self.C},{self.H},{self.W})")
        recorded = (self._push_count % self.stride == 0)
        self._push_count += 1
        if recorded:
            # detach 避免推理端誤留計算圖
            self._frames.append(feat.detach())
        return recorded

    @property
    def is_empty(self) -> bool:
        return len(self._frames) == 0

    @property
    def is_full(self) -> bool:
        return len(self._frames) == self.T

    def get(self) -> torch.Tensor:
        """回傳 (T*C, H, W),通道排列 [c0t0, ..., c0t{T-1}, c1t0, ...]。

        緩衝為空時回傳全零;未滿時以最舊有效幀重複填充。
        """
        if self.is_empty:
            return torch.zeros(self.T * self.C, self.H, self.W)
        frames = list(self._frames)  # 由舊到新
        pad = [frames[0]] * (self.T - len(frames))
        stacked = torch.stack(pad + frames, dim=0)  # (T, C, H, W)
        return stack_time_to_channels(stacked)

    def reset(self) -> None:
        """清空緩衝(track ID switch 時呼叫)。"""
        self._frames.clear()
        self._push_count = 0


class DualScaleBuffer:
    """雙時間尺度緩衝:短尺度(逐幀)+ 長尺度(每 long_stride 幀收一張)。

    推理端每個 track 持有一組;`push()` 同時餵兩個 buffer。
    """

    def __init__(self, C: int, H: int, W: int,
                 short_T: int = 16, short_stride: int = 1,
                 long_T: int = 16, long_stride: int = 8):
        self.short = RingBuffer(short_T, C, H, W, stride=short_stride)
        self.long = RingBuffer(long_T, C, H, W, stride=long_stride)

    def push(self, feat: torch.Tensor) -> None:
        self.short.push(feat)
        self.long.push(feat)

    def get_short(self) -> torch.Tensor:
        return self.short.get()

    def get_long(self) -> torch.Tensor:
        return self.long.get()

    @property
    def short_ready(self) -> bool:
        """短尺度是否已有足夠幀(至少一幀即可推理,滿幀最可靠)。"""
        return not self.short.is_empty

    def reset(self) -> None:
        """track ID switch 或回收時清空兩個 buffer。"""
        self.short.reset()
        self.long.reset()
