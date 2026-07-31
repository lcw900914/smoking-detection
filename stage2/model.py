"""M2 骨架時序編碼器(TCN)+ 分類頭。

小資料優先:膨脹 1D 卷積、約 10 萬參數、強正規化。
之後換 Transformer 只需保持 forward 介面 (B, T, 85) → logits。
"""
import torch
import torch.nn as nn

from stage2.normalize import FEATURE_DIM
from stage2.taxonomy import TRAIN_CLASSES

# 訓練用的合併類別(細標籤與合併規則見 stage2/taxonomy.py)。
# smoking 固定為 index 0,評估時直接取這一維當抽菸機率。
CLASSES = TRAIN_CLASSES


class TCNBlock(nn.Module):
    """Conv1d + BN + ReLU + Dropout,殘差連接。"""

    def __init__(self, c_in: int, c_out: int, dilation: int,
                 k: int = 5, dropout: float = 0.3):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, padding=pad,
                              dilation=dilation)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.res = (nn.Conv1d(c_in, c_out, 1) if c_in != c_out
                    else nn.Identity())

    def forward(self, x):  # (B, C, T)
        return self.drop(self.act(self.bn(self.conv(x)))) + self.res(x)


class PoseTCN(nn.Module):
    """骨架序列分類器:(B, T, 85) → logits (B, num_classes)。

    forward 亦回傳序列嵌入 (B, 128) 供 Level B(M4)作事件嵌入使用。
    """

    def __init__(self, num_classes: int = len(CLASSES),
                 in_dim: int = FEATURE_DIM, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.Sequential(
            TCNBlock(in_dim, 64, dilation=1, k=5, dropout=dropout),
            TCNBlock(64, 128, dilation=2, k=3, dropout=dropout),
            TCNBlock(128, 128, dilation=4, k=3, dropout=dropout),
        )
        self.head = nn.Linear(128, num_classes)
        self.embed_dim = 128

    def forward(self, x: torch.Tensor):
        """x: (B, T, 85) → (logits (B, C), embedding (B, 128))"""
        h = self.blocks(x.transpose(1, 2))   # (B, 128, T)
        emb = h.mean(dim=2)                  # 時間 GAP
        return self.head(emb), emb
