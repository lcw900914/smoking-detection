"""損失函數測試:形狀、數值合理性、KD。"""
import pytest
import torch

from training.losses import (stage_ce_loss, cycle_bce_loss, fusion_bce_loss,
                             logit_kd_loss, FeatureKD, TotalLoss)

B = 4


def _outputs():
    return {
        "stage_logits": torch.randn(B, 4, requires_grad=True),
        "cycle_logits": torch.randn(B, 2, requires_grad=True),
        "fusion_logit": torch.randn(B, requires_grad=True),
        "emb_long": torch.randn(B, 32, requires_grad=True),
    }


def _batch():
    return {
        "stage_label": torch.randint(0, 4, (B,)),
        "cycle_label": torch.randint(0, 2, (B,)).float(),
        "clip_label": torch.randint(0, 2, (B,)).float(),
    }


def test_individual_losses_scalar():
    out, batch = _outputs(), _batch()
    for loss in (stage_ce_loss(out["stage_logits"], batch["stage_label"]),
                 cycle_bce_loss(out["cycle_logits"], batch["cycle_label"]),
                 fusion_bce_loss(out["fusion_logit"], batch["clip_label"])):
        assert loss.dim() == 0 and torch.isfinite(loss)


def test_logit_kd_zero_when_identical():
    """student 與 teacher logits 相同時 KD 損失為 0。"""
    logits = torch.randn(B, 2)
    assert logit_kd_loss(logits, logits, temperature=4.0).item() == \
        pytest.approx(0.0, abs=1e-6)


def test_logit_kd_positive_when_different():
    s, t = torch.randn(B, 2), torch.randn(B, 2) + 3.0
    assert logit_kd_loss(s, t).item() > 0


def test_feature_kd_projection():
    kd = FeatureKD(student_dim=32, teacher_dim=64)
    loss = kd(torch.randn(B, 32), torch.randn(B, 64))
    assert loss.dim() == 0 and loss.item() >= 0


def test_total_loss_weighting_and_backward():
    weights = {"stage_ce": 1.0, "cycle_bce": 0.5, "fusion_bce": 0.5}
    criterion = TotalLoss(weights)
    out, batch = _outputs(), _batch()
    losses = criterion(out, batch)
    expected = (weights["stage_ce"] * losses["stage_ce"]
                + weights["cycle_bce"] * losses["cycle_bce"]
                + weights["fusion_bce"] * losses["fusion_bce"])
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()  # 反向可通


def test_total_loss_with_kd():
    weights = {"stage_ce": 1.0, "cycle_bce": 0.5, "fusion_bce": 0.5,
               "logit_kd": 1.0, "feature_kd": 0.5}
    kd = FeatureKD(student_dim=32, teacher_dim=16)
    criterion = TotalLoss(weights, kd_temperature=4.0, feature_kd=kd)
    out, batch = _outputs(), _batch()
    batch["teacher_logits"] = torch.randn(B, 2)
    batch["teacher_feat"] = torch.randn(B, 16)
    losses = criterion(out, batch)
    assert "logit_kd" in losses and "feature_kd" in losses
    assert torch.isfinite(losses["total"])
