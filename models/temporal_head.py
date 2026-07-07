"""時序融合頭:以純 2D 算子完成跨時間混合。

輸入為 channel-buffer 疊合後的 (T·C)×H×W 特徵圖,通道排列必須是
interleave([c0t0, ..., c0t{T-1}, c1t0, ...],見 ring_buffer.py)。

結構(全部僅用 Conv2d / BN / activation / pooling / Linear,
禁止 Conv3D、LSTM、GRU、attention,以保持邊緣 NPU 相容性):
1. 分組時間卷積 Conv2d(T·C → T·C, k=1, groups=C):
   每組恰為同一特徵通道的 T 個時刻 → 等價 depthwise temporal conv
2. Conv2d(T·C → mid, k=1):跨特徵語意混合
3. 兩個 3×3 conv block(mid → mid)
4. GAP → 分類頭
"""
import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, k: int = 3) -> nn.Sequential:
    """標準 conv + BN + SiLU block。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class TemporalHead(nn.Module):
    """單一尺度的時序融合頭。

    Args:
        C: 特徵通道數(backbone 輸出)
        T: 緩衝幀數
        num_classes: 分類頭輸出類別數
        mid_channels: 1×1 混合後通道數

    forward 回傳 (logits, embedding):
        logits: (B, num_classes)
        embedding: (B, mid_channels)  — GAP 後、分類前的向量,供融合層使用
    """

    def __init__(self, C: int = 128, T: int = 16,
                 num_classes: int = 4, mid_channels: int = 256):
        super().__init__()
        self.C, self.T = C, T
        TC = T * C

        # 1. 分組時間卷積:groups=C,第 i 組僅含通道 i 的 T 個時刻,
        #    各組獨立做時間加權(參數量 C·T²)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(TC, TC, kernel_size=1, groups=C, bias=False),
            nn.BatchNorm2d(TC),
            nn.SiLU(inplace=True),
        )
        # 2. 1×1 跨特徵語意混合
        self.mix = _conv_block(TC, mid_channels, k=1)
        # 3. 兩個 3×3 conv block
        self.block1 = _conv_block(mid_channels, mid_channels, k=3)
        self.block2 = _conv_block(mid_channels, mid_channels, k=3)
        # 4. GAP + 分類頭
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(mid_channels, num_classes)

        self.embed_dim = mid_channels

    def forward(self, x: torch.Tensor):
        """(B, T*C, H, W) → (logits (B, num_classes), embedding (B, mid))"""
        assert x.shape[1] == self.T * self.C, (
            f"輸入通道 {x.shape[1]} 不等於 T*C={self.T * self.C}")
        x = self.temporal_conv(x)
        x = self.mix(x)
        x = self.block1(x)
        x = self.block2(x)
        emb = self.gap(x).flatten(1)          # (B, mid_channels)
        logits = self.classifier(emb)
        return logits, emb


class FusionHead(nn.Module):
    """融合層:短、長尺度 embedding concat 後過 MLP,輸出 cycle score。

    輸出為 sigmoid 前的 logit(訓練用 BCEWithLogits 較穩定);
    推理時呼叫端自行 `torch.sigmoid`。
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb_short: torch.Tensor,
                emb_long: torch.Tensor) -> torch.Tensor:
        """(B, E), (B, E) → cycle logit (B,)"""
        return self.mlp(torch.cat([emb_short, emb_long], dim=1)).squeeze(-1)
