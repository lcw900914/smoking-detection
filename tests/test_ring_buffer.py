"""RingBuffer 測試——通道排列正確性是重中之重。

驗證:
1. stack_time_to_channels 的 interleave 排列 [c0t0, ..., c0t{T-1}, c1t0, ...]
2. 分組時間卷積(groups=C)每組恰好對應同一特徵通道的 T 個時刻
3. continual(RingBuffer 逐幀 push)與 offline(dataset 疊合)路徑
   tensor layout 完全一致
4. stride 取樣、未滿填充、reset 行為
"""
import torch
import torch.nn as nn
import pytest

from models.ring_buffer import (RingBuffer, DualScaleBuffer,
                                stack_time_to_channels)

T, C, H, W = 4, 3, 2, 2


def make_frame(t: int) -> torch.Tensor:
    """建立可辨識的特徵幀:第 c 通道全為 100*c + t。"""
    return torch.stack([torch.full((H, W), 100.0 * c + t) for c in range(C)])


class TestStackLayout:
    """通道排列正確性。"""

    def test_interleave_order(self):
        """疊合後通道 k = c*T + t 必須等於幀 t 的通道 c。"""
        feats = torch.stack([make_frame(t) for t in range(T)])  # (T,C,H,W)
        stacked = stack_time_to_channels(feats)                 # (C*T,H,W)
        assert stacked.shape == (C * T, H, W)
        for c in range(C):
            for t in range(T):
                expected = 100.0 * c + t
                assert torch.all(stacked[c * T + t] == expected), \
                    f"通道 {c * T + t} 應為 c{c}t{t}={expected}"

    def test_batch_version(self):
        """5 維(含 batch)版本排列一致。"""
        feats = torch.stack([make_frame(t) for t in range(T)])
        batched = feats.unsqueeze(0).repeat(2, 1, 1, 1, 1)  # (2,T,C,H,W)
        stacked = stack_time_to_channels(batched)
        assert stacked.shape == (2, C * T, H, W)
        single = stack_time_to_channels(feats)
        assert torch.equal(stacked[0], single)
        assert torch.equal(stacked[1], single)

    def test_grouped_conv_sees_correct_channels(self):
        """Conv2d(groups=C) 第 c 組的輸出只由通道 c 的 T 個時刻決定。

        將每組權重設為全 1(輸出=組內和),驗證輸出等於 Σ_t (100c+t)。
        """
        feats = torch.stack([make_frame(t) for t in range(T)])
        stacked = stack_time_to_channels(feats).unsqueeze(0)  # (1,C*T,H,W)

        conv = nn.Conv2d(C * T, C * T, kernel_size=1, groups=C, bias=False)
        with torch.no_grad():
            conv.weight.fill_(1.0)  # 每個輸出通道 = 該組 T 個輸入通道之和
        out = conv(stacked)

        for c in range(C):
            expected = sum(100.0 * c + t for t in range(T))
            group_out = out[0, c * T:(c + 1) * T]
            assert torch.allclose(group_out,
                                  torch.full_like(group_out, expected)), \
                f"組 {c} 輸出應為 {expected}(僅含通道 {c} 的時刻)"


class TestRingBuffer:
    def test_push_and_get_order(self):
        """push 超過 T 幀後,get 回傳最近 T 幀(由舊到新)。"""
        buf = RingBuffer(T, C, H, W, stride=1)
        for t in range(T + 2):  # push 0..5,留下 2..5
            buf.push(make_frame(t))
        stacked = buf.get()
        expected = stack_time_to_channels(
            torch.stack([make_frame(t) for t in range(2, T + 2)]))
        assert torch.equal(stacked, expected)

    def test_stride_sampling(self):
        """stride=2 時每 2 次 push 收錄一次(第 0, 2, 4, ... 次)。"""
        buf = RingBuffer(T, C, H, W, stride=2)
        recorded = [buf.push(make_frame(t)) for t in range(8)]
        assert recorded == [True, False] * 4
        stacked = buf.get()
        expected = stack_time_to_channels(
            torch.stack([make_frame(t) for t in (0, 2, 4, 6)]))
        assert torch.equal(stacked, expected)

    def test_padding_with_oldest(self):
        """未滿 T 幀時,以最舊有效幀重複填充在最前面。"""
        buf = RingBuffer(T, C, H, W)
        buf.push(make_frame(0))
        buf.push(make_frame(1))
        stacked = buf.get()
        expected = stack_time_to_channels(torch.stack(
            [make_frame(0), make_frame(0), make_frame(0), make_frame(1)]))
        assert torch.equal(stacked, expected)

    def test_empty_returns_zeros(self):
        buf = RingBuffer(T, C, H, W)
        assert torch.all(buf.get() == 0)

    def test_reset(self):
        """reset 後緩衝清空、stride 計數歸零(track ID switch 情境)。"""
        buf = RingBuffer(T, C, H, W, stride=2)
        buf.push(make_frame(0))
        buf.push(make_frame(1))
        buf.reset()
        assert buf.is_empty
        assert buf.push(make_frame(5)) is True  # 計數歸零,第一次必收錄

    def test_shape_check(self):
        buf = RingBuffer(T, C, H, W)
        with pytest.raises(AssertionError):
            buf.push(torch.zeros(C + 1, H, W))


class TestContinualOfflineConsistency:
    """continual(逐幀 push)與 offline(dataset 一次疊合)必須完全一致。"""

    def test_short_scale(self):
        frames = [torch.randn(C, H, W) for _ in range(10)]
        buf = RingBuffer(T, C, H, W, stride=1)
        for f in frames:
            buf.push(f)
        # offline 路徑:直接取最後 T 幀疊合
        offline = stack_time_to_channels(torch.stack(frames[-T:]))
        assert torch.equal(buf.get(), offline)

    def test_long_scale_with_stride(self):
        """stride=8 的長尺度:與 dataset 的 window_indices 取法一致。"""
        from data.dataset import window_indices
        stride, n = 8, 8 * (T - 1) + 1  # 恰好填滿:push 0..24
        frames = [torch.randn(C, H, W) for _ in range(n)]
        buf = RingBuffer(T, C, H, W, stride=stride)
        for f in frames:
            buf.push(f)
        idx = window_indices(n - 1, T, stride)  # [0, 8, 16, 24]
        offline = stack_time_to_channels(
            torch.stack([frames[j] for j in idx]))
        assert torch.equal(buf.get(), offline)


class TestDualScaleBuffer:
    def test_push_feeds_both(self):
        dsb = DualScaleBuffer(C, H, W, short_T=T, short_stride=1,
                              long_T=T, long_stride=2)
        for t in range(8):
            dsb.push(make_frame(t))
        short = dsb.get_short()
        long_ = dsb.get_long()
        assert torch.equal(short, stack_time_to_channels(
            torch.stack([make_frame(t) for t in (4, 5, 6, 7)])))
        assert torch.equal(long_, stack_time_to_channels(
            torch.stack([make_frame(t) for t in (0, 2, 4, 6)])))

    def test_reset(self):
        dsb = DualScaleBuffer(C, H, W, short_T=T, long_T=T)
        dsb.push(make_frame(0))
        dsb.reset()
        assert dsb.short.is_empty and dsb.long.is_empty
