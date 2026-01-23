import torch
import torch.nn as nn
from torchvision import models


class ImageBackbone(nn.Module):
    def __init__(self, backbone: str = "resnet18", out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        backbone = backbone.lower()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            base.conv1 = self._adapt_conv(base.conv1, in_channels=1)
            base.fc = nn.Identity()
            self.backbone = base
            in_features = 512
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            base.features[0][0] = self._adapt_conv(base.features[0][0], in_channels=1)
            base.classifier = nn.Identity()
            self.backbone = base
            in_features = 1280
        else:
            raise ValueError(f"Backbone no soportado: {backbone}")

        self.proj = nn.Linear(in_features, out_dim)

    @staticmethod
    def _adapt_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if conv.weight.shape[1] != in_channels:
                new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            else:
                new_conv.weight.copy_(conv.weight)
        return new_conv

    def forward(self, x):  # x: (B,1,H,W)
        feats = self.backbone(x)
        return self.proj(feats)
