"""時空圖卷積積木:空間走拓樸圖,時間走膨脹卷積。

一個 block 做兩件事:
  空間 —— 每個節點聚合「自己 / 向心鄰居 / 離心鄰居」三個分區的特徵,
          分區權重各自獨立(這就是 ST-GCN 的 spatial configuration)。
  時間 —— 對每個節點各自做 1D 膨脹卷積,堆三層就能覆蓋數秒的時窗。

另外掛一個可學的**邊重要性遮罩**:拓樸給的是「哪些邊存在」,遮罩學的是
「哪些邊重要」。抽菸這種任務期待它把腕-鼻那條功能邊的權重養大。
"""
import torch
import torch.nn as nn


class SpatialGraphConv(nn.Module):
    """分區圖卷積:(B, C_in, T, V) → (B, C_out, T, V)。"""

    def __init__(self, c_in: int, c_out: int, A: torch.Tensor,
                 edge_importance: bool = True):
        super().__init__()
        self.register_buffer("A", A.clone().float())      # (K, V, V)
        self.K = A.shape[0]
        self.conv = nn.Conv2d(c_in, c_out * self.K, kernel_size=1)
        self.edge_mask = (nn.Parameter(torch.ones_like(self.A))
                          if edge_importance else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, V = x.shape
        y = self.conv(x).view(B, self.K, -1, T, V)
        A = self.A if self.edge_mask is None else self.A * self.edge_mask
        return torch.einsum("bkctv,kvw->bctw", y, A).contiguous()


class STGCNBlock(nn.Module):
    """空間圖卷積 + 時間膨脹卷積 + 殘差。"""

    def __init__(self, c_in: int, c_out: int, A: torch.Tensor,
                 kt: int = 5, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gcn = SpatialGraphConv(c_in, c_out, A)
        self.gcn_bn = nn.BatchNorm2d(c_out)
        pad = (kt - 1) // 2 * dilation
        self.tcn = nn.Sequential(
            nn.Conv2d(c_out, c_out, (kt, 1), padding=(pad, 0),
                      dilation=(dilation, 1)),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
        )
        self.res = (nn.Identity() if c_in == c_out else
                    nn.Sequential(nn.Conv2d(c_in, c_out, 1),
                                  nn.BatchNorm2d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gcn_bn(self.gcn(x)))
        return self.act(self.tcn(h) + self.res(x))


class STGCNTrunk(nn.Module):
    """三層 ST-GCN 主幹:(B, C, T, V) → (B, C_out, T, V),時間長度不變。

    膨脹 1/2/4、時間核 5 → 感受野 29 幀(10fps ≈ 2.9 秒),
    正好涵蓋一次舉手-停留-放下,又不會長到把整段動作糊成一塊。
    """

    def __init__(self, c_in: int, A: torch.Tensor,
                 channels=(32, 48, 64), kt: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(c_in * A.shape[-1])
        blocks = []
        prev = c_in
        for i, c in enumerate(channels):
            blocks.append(STGCNBlock(prev, c, A, kt=kt,
                                     dilation=2 ** i, dropout=dropout))
            prev = c
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape
        h = x.permute(0, 1, 3, 2).reshape(B, C * V, T)
        h = self.data_bn(h).view(B, C, V, T).permute(0, 1, 3, 2)
        for blk in self.blocks:
            h = blk(h)
        return h
