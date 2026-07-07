"""TemporalHead / FusionHead 測試:輸出形狀與算子白名單(NPU 相容性)。"""
import torch
import torch.nn as nn
import pytest

from models.temporal_head import TemporalHead, FusionHead

C, T, HH, WW = 8, 4, 7, 7


def test_output_shapes():
    head = TemporalHead(C=C, T=T, num_classes=4, mid_channels=32)
    x = torch.randn(2, T * C, HH, WW)
    logits, emb = head(x)
    assert logits.shape == (2, 4)
    assert emb.shape == (2, 32)


def test_channel_mismatch_raises():
    head = TemporalHead(C=C, T=T, num_classes=4, mid_channels=32)
    with pytest.raises(AssertionError):
        head(torch.randn(2, T * C + 1, HH, WW))


def test_fusion_head():
    fusion = FusionHead(embed_dim=32, hidden_dim=16)
    out = fusion(torch.randn(3, 32), torch.randn(3, 32))
    assert out.shape == (3,)


def test_grouped_conv_config():
    """分組時間卷積必須 groups=C、1×1 kernel。"""
    head = TemporalHead(C=C, T=T, num_classes=4, mid_channels=32)
    conv = head.temporal_conv[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.groups == C
    assert conv.kernel_size == (1, 1)
    assert conv.in_channels == conv.out_channels == T * C


def test_operator_whitelist():
    """禁止 Conv3D / LSTM / GRU / attention——僅允許邊緣 NPU 友善算子。"""
    forbidden = (nn.Conv3d, nn.LSTM, nn.GRU, nn.RNN,
                 nn.MultiheadAttention, nn.TransformerEncoderLayer)
    allowed = (nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.ReLU,
               nn.Linear, nn.AdaptiveAvgPool2d, nn.MaxPool2d,
               nn.Sequential, nn.ModuleList,
               TemporalHead, FusionHead)
    head = TemporalHead(C=C, T=T, num_classes=4, mid_channels=32)
    fusion = FusionHead(embed_dim=32)
    for model in (head, fusion):
        for m in model.modules():
            assert not isinstance(m, forbidden), \
                f"發現禁止算子:{type(m).__name__}"
            assert isinstance(m, allowed), \
                f"發現白名單外算子:{type(m).__name__}"
