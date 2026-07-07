"""單幀特徵抽取 backbone。

規格:ResNet-18(torchvision 預訓練)改造——移除 avgpool 與 fc,
將 layer3 的 downsample stride 改為 1 以維持 stride-8,輸出經 1×1 conv
壓縮到 C=128;輸入 224×224 時輸出 128×28×28。

介面設計為可替換:之後換 CSPPartialNet 時,實作同樣的
`feature_dim` / `feature_size` 屬性並在 `build_backbone` 註冊即可。
"""
import torch
import torch.nn as nn


class ResNet18Backbone(nn.Module):
    """ResNet-18 stride-8 特徵抽取器。

    Attributes:
        feature_dim: 輸出通道數 C(預設 128)
        feature_size: 輸入 224 時的空間尺寸 (28, 28)
    """

    def __init__(self, out_channels: int = 128, pretrained: bool = True,
                 input_size: int = 224):
        super().__init__()
        import torchvision  # 延遲載入,避免無 torchvision 環境無法跑其他模組

        weights = (torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                   if pretrained else None)
        resnet = torchvision.models.resnet18(weights=weights)

        # layer3 原本 stride 2(整體 stride 16)→ 改成 1 維持 stride-8,
        # 保留 layer3 的容量(256 通道)再壓縮
        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,  # stride 4
            resnet.layer1,   # stride 4
            resnet.layer2,   # stride 8
            resnet.layer3,   # stride 8(已改)
        )
        # 1×1 conv 壓縮到 C=out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        self.feature_dim = out_channels
        self.feature_size = (input_size // 8, input_size // 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, C, 28, 28)"""
        return self.proj(self.stem(x))


# backbone 註冊表:名稱 → 建構函式(替換 backbone 只需在此加一行)
_BACKBONES = {
    "resnet18": ResNet18Backbone,
}


def build_backbone(cfg: dict) -> nn.Module:
    """依 model.yaml 的 backbone 區段建立 backbone。

    cfg 範例:{"name": "resnet18", "pretrained": true, "out_channels": 128}
    """
    name = cfg.get("name", "resnet18")
    if name not in _BACKBONES:
        raise ValueError(f"未知 backbone: {name},可用:{list(_BACKBONES)}")
    return _BACKBONES[name](
        out_channels=cfg.get("out_channels", 128),
        pretrained=cfg.get("pretrained", True),
    )
