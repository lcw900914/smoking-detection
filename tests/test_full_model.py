"""完整模型測試:continual 與 offline 兩條前向路徑輸出必須一致。

使用縮小版設定(小 backbone 輸出、小 T)以加快測試;
一致性只依賴 layout 與共享權重,與尺寸無關。
"""
import pytest
import torch

torchvision = pytest.importorskip("torchvision")

from models.full_model import build_model
from models.ring_buffer import DualScaleBuffer, stack_time_to_channels

MODEL_CFG = {
    "backbone": {"name": "resnet18", "pretrained": False,
                 "out_channels": 16},
    "feature": {"C": 16, "H": 8, "W": 8},
    "buffer": {
        "short": {"T": 4, "stride": 1},
        "long": {"T": 4, "stride": 2},
    },
    "temporal_head": {"mid_channels": 32,
                      "num_stage_classes": 4, "num_cycle_classes": 2},
    "fusion": {"hidden_dim": 16},
}
IMG = 64  # 64/8 = 8 → 特徵 8×8


@pytest.fixture(scope="module")
def model():
    m = build_model(MODEL_CFG)
    m.eval()  # BN 用 running stats,兩路徑才可比
    return m


def test_backbone_output_shape(model):
    feat = model.extract_feature(torch.randn(2, 3, IMG, IMG))
    assert feat.shape == (2, 16, 8, 8)


def test_forward_buffers_shapes(model):
    out = model.forward_buffers(torch.randn(2, 4 * 16, 8, 8),
                                torch.randn(2, 4 * 16, 8, 8))
    assert out["stage_logits"].shape == (2, 4)
    assert out["cycle_logits"].shape == (2, 2)
    assert out["cycle_score"].shape == (2,)
    assert torch.all((out["cycle_score"] >= 0) & (out["cycle_score"] <= 1))


def test_continual_vs_offline_consistency(model):
    """逐幀 push(continual)與 clip 一次前向(offline)結果必須一致。

    push 7 幀:短 buffer 留 3..6;長 buffer(stride=2)收 0,2,4,6。
    offline 路徑用相同幀組 clip 餵 model.forward()。
    """
    n = 7
    frames = [torch.randn(3, IMG, IMG) for _ in range(n)]

    # ---- continual 路徑 ----
    buf = DualScaleBuffer(C=16, H=8, W=8, short_T=4, short_stride=1,
                          long_T=4, long_stride=2)
    with torch.no_grad():
        for f in frames:
            feat = model.extract_feature(f.unsqueeze(0))[0]
            buf.push(feat)
        out_c = model.forward_buffers(buf.get_short().unsqueeze(0),
                                      buf.get_long().unsqueeze(0))

    # ---- offline 路徑 ----
    short_imgs = torch.stack(frames[3:7]).unsqueeze(0)          # (1,4,3,H,W)
    long_imgs = torch.stack([frames[j] for j in (0, 2, 4, 6)]).unsqueeze(0)
    with torch.no_grad():
        out_o = model(short_imgs, long_imgs)

    assert torch.allclose(out_c["stage_logits"], out_o["stage_logits"],
                          atol=1e-4)
    assert torch.allclose(out_c["cycle_logits"], out_o["cycle_logits"],
                          atol=1e-4)
    assert torch.allclose(out_c["cycle_score"],
                          torch.sigmoid(out_o["fusion_logit"]), atol=1e-4)


def test_freeze_unfreeze(model):
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.backbone.parameters())
    head_params = list(model.head_parameters())
    assert all(p.requires_grad for p in head_params)
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())
