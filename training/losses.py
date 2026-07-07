"""損失函數:CE、logit KD(KL,τ=4)、feature KD(projection + MSE)。

總損失權重全部由 configs/train.yaml 提供,不寫死。
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def stage_ce_loss(stage_logits: torch.Tensor,
                  stage_label: torch.Tensor) -> torch.Tensor:
    """階段分類 CE(標籤為幀級標籤聚合後的視窗標籤)。"""
    return F.cross_entropy(stage_logits, stage_label)


def cycle_bce_loss(cycle_logits: torch.Tensor, cycle_label: torch.Tensor,
                   pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """週期頭 BCE:2 類 logits 取正類(index 1)對二元標籤。

    pos_weight:正類權重(≈ 負樣本數/正樣本數),補償類別不平衡。
    """
    return F.binary_cross_entropy_with_logits(
        cycle_logits[:, 1] - cycle_logits[:, 0], cycle_label.float(),
        pos_weight=pos_weight)


def fusion_bce_loss(fusion_logit: torch.Tensor, clip_label: torch.Tensor,
                    pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """融合 cycle score BCE(fusion 輸出為 sigmoid 前 logit)。"""
    return F.binary_cross_entropy_with_logits(
        fusion_logit, clip_label.float(), pos_weight=pos_weight)


def logit_kd_loss(student_logits: torch.Tensor,
                  teacher_logits: torch.Tensor,
                  temperature: float = 4.0) -> torch.Tensor:
    """Logit 蒸餾:KL(softmax(t/τ) ‖ softmax(s/τ)) × τ²。

    teacher_logits 為 distill_precompute.py 預先存好的 soft labels。
    """
    T = temperature
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


class FeatureKD(nn.Module):
    """特徵蒸餾:student 特徵過 projection 對齊 teacher 維度後 MSE。

    student 端取時序頭 GAP 後 embedding(向量),projection 用 Linear;
    亦支援空間特徵圖(用 1×1 conv),依輸入維度自動選擇。
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj_vec = nn.Linear(student_dim, teacher_dim)
        self.proj_map = nn.Conv2d(student_dim, teacher_dim, kernel_size=1)

    def forward(self, student_feat: torch.Tensor,
                teacher_feat: torch.Tensor) -> torch.Tensor:
        if student_feat.dim() == 2:      # (B, D)
            proj = self.proj_vec(student_feat)
        else:                            # (B, C, H, W)
            proj = self.proj_map(student_feat)
            # teacher 特徵若空間尺寸不同,GAP 對齊成向量比較
            if proj.shape[-2:] != teacher_feat.shape[-2:]:
                proj = proj.mean(dim=(-2, -1))
                teacher_feat = teacher_feat.mean(dim=(-2, -1)) \
                    if teacher_feat.dim() == 4 else teacher_feat
        return F.mse_loss(proj, teacher_feat)


class TotalLoss(nn.Module):
    """組合損失:CE + BCE(+ KD)。權重由 config 提供。

    Args:
        weights: {"stage_ce", "cycle_bce", "fusion_bce",
                  可選 "logit_kd", "feature_kd"}
        kd_temperature: logit KD 溫度 τ
        feature_kd: FeatureKD 模組(啟用 feature KD 時傳入;
                    其 projection 參數需一併加入 optimizer)
    """

    def __init__(self, weights: Dict[str, float],
                 kd_temperature: float = 4.0,
                 feature_kd: Optional[FeatureKD] = None,
                 pos_weight: Optional[float] = None):
        super().__init__()
        self.w = weights
        self.tau = kd_temperature
        self.feature_kd = feature_kd
        # 正類(smoking)損失權重,補償類別不平衡
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight)) if pos_weight else None)

    def forward(self, outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """回傳 {"total": ..., 各分項...}(分項供 tensorboard 記錄)。"""
        losses = {
            "stage_ce": stage_ce_loss(outputs["stage_logits"],
                                      batch["stage_label"]),
            "cycle_bce": cycle_bce_loss(outputs["cycle_logits"],
                                        batch["cycle_label"],
                                        pos_weight=self.pos_weight),
            "fusion_bce": fusion_bce_loss(outputs["fusion_logit"],
                                          batch["clip_label"],
                                          pos_weight=self.pos_weight),
        }
        if "teacher_logits" in batch and self.w.get("logit_kd", 0) > 0:
            losses["logit_kd"] = logit_kd_loss(
                outputs["cycle_logits"], batch["teacher_logits"], self.tau)
        if ("teacher_feat" in batch and self.feature_kd is not None
                and self.w.get("feature_kd", 0) > 0):
            losses["feature_kd"] = self.feature_kd(
                outputs["emb_long"], batch["teacher_feat"])

        total = sum(self.w.get(k, 0.0) * v for k, v in losses.items())
        losses["total"] = total
        return losses
