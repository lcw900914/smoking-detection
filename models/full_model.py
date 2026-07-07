"""完整模型組裝:backbone + 雙尺度時序頭 + 融合層。

兩種前向路徑:
- 訓練(offline):dataset 已組好疊合張量,直接餵時序頭;
  或端到端時輸入影像 clip,backbone 共享權重逐幀抽特徵後疊合。
- 推理(continual):新幀只跑一次 `extract_feature`,推入外部
  DualScaleBuffer 後以 `forward_buffers` 僅重算時序頭。
"""
from typing import Dict

import torch
import torch.nn as nn

from .backbone import build_backbone
from .ring_buffer import stack_time_to_channels
from .temporal_head import TemporalHead, FusionHead


class SmokingActionModel(nn.Module):
    """抽菸行為偵測完整模型。

    Attributes:
        backbone: 單幀特徵抽取(可替換)
        short_head: 短尺度階段頭(4 類:S1/S2/S3/background)
        long_head: 長尺度週期頭(2 類:週期/非週期)
        fusion: 融合層,輸出 cycle score logit
    """

    def __init__(self, model_cfg: dict):
        super().__init__()
        feat = model_cfg["feature"]
        buf = model_cfg["buffer"]
        th = model_cfg["temporal_head"]

        self.C = feat["C"]
        self.short_T = buf["short"]["T"]
        self.long_T = buf["long"]["T"]

        self.backbone = build_backbone(model_cfg["backbone"])
        self.short_head = TemporalHead(
            C=self.C, T=self.short_T,
            num_classes=th["num_stage_classes"],
            mid_channels=th["mid_channels"])
        self.long_head = TemporalHead(
            C=self.C, T=self.long_T,
            num_classes=th["num_cycle_classes"],
            mid_channels=th["mid_channels"])
        self.fusion = FusionHead(
            embed_dim=th["mid_channels"],
            hidden_dim=model_cfg["fusion"]["hidden_dim"])

    # ---------- 推理端(continual inference) ----------

    @torch.no_grad()
    def extract_feature(self, images: torch.Tensor) -> torch.Tensor:
        """單幀特徵抽取:(B, 3, H, W) → (B, C, H', W')。

        推理主迴圈把所有 track 的 ROI 合成一個 batch 呼叫一次。
        """
        return self.backbone(images)

    @torch.no_grad()
    def forward_buffers(self, short_stacked: torch.Tensor,
                        long_stacked: torch.Tensor) -> Dict[str, torch.Tensor]:
        """僅重算時序頭(不碰 backbone)。

        Args:
            short_stacked: (B, short_T*C, H', W') — DualScaleBuffer.get_short()
            long_stacked:  (B, long_T*C, H', W') — DualScaleBuffer.get_long()
        """
        stage_logits, emb_s = self.short_head(short_stacked)
        cycle_logits, emb_l = self.long_head(long_stacked)
        fusion_logit = self.fusion(emb_s, emb_l)
        return {
            "stage_logits": stage_logits,      # (B, 4)
            "cycle_logits": cycle_logits,      # (B, 2)
            "cycle_score": torch.sigmoid(fusion_logit),  # (B,)
        }

    # ---------- 訓練端 ----------

    def forward_features(self, short_stacked: torch.Tensor,
                         long_stacked: torch.Tensor) -> Dict[str, torch.Tensor]:
        """階段一:輸入離線特徵疊合張量(帶梯度)。回傳 fusion 為 logit。"""
        stage_logits, emb_s = self.short_head(short_stacked)
        cycle_logits, emb_l = self.long_head(long_stacked)
        fusion_logit = self.fusion(emb_s, emb_l)
        return {
            "stage_logits": stage_logits,
            "cycle_logits": cycle_logits,
            "fusion_logit": fusion_logit,
            "emb_short": emb_s,
            "emb_long": emb_l,
        }

    def forward(self, short_images: torch.Tensor,
                long_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """階段二(端到端):輸入影像 clip,backbone 共享權重逐幀前向。

        Args:
            short_images: (B, T_s, 3, H, W) 短尺度幀(時間由舊到新)
            long_images:  (B, T_l, 3, H, W) 長尺度幀
        """
        B, Ts = short_images.shape[:2]
        Tl = long_images.shape[1]

        # 兩個尺度的幀合併成一個大 batch,只跑一次 backbone
        all_imgs = torch.cat([
            short_images.flatten(0, 1),   # (B*Ts, 3, H, W)
            long_images.flatten(0, 1),    # (B*Tl, 3, H, W)
        ], dim=0)
        feats = self.backbone(all_imgs)   # (B*(Ts+Tl), C, H', W')

        C, Hf, Wf = feats.shape[1:]
        short_feats = feats[:B * Ts].view(B, Ts, C, Hf, Wf)
        long_feats = feats[B * Ts:].view(B, Tl, C, Hf, Wf)

        # 與推理端 RingBuffer.get() 完全一致的 layout(共用同一函式)
        short_stacked = stack_time_to_channels(short_feats)
        long_stacked = stack_time_to_channels(long_feats)
        return self.forward_features(short_stacked, long_stacked)

    def head_parameters(self):
        """時序頭 + 融合層參數(階段一只訓練這些)。"""
        for m in (self.short_head, self.long_head, self.fusion):
            yield from m.parameters()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)


def build_model(model_cfg: dict) -> SmokingActionModel:
    """依 configs/model.yaml 建立完整模型。"""
    return SmokingActionModel(model_cfg)
