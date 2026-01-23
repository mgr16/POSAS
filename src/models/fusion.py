import torch
import torch.nn as nn
from .cnn import ImageBackbone
from .mlp import TabularMLP

class FusionModel(nn.Module):
    def __init__(self, num_features: int, emb_cardinalities: dict, dropout: float = 0.2, backbone: str = "resnet18"):
        super().__init__()
        self.cnn = ImageBackbone(backbone=backbone, out_dim=128)
        self.tab = TabularMLP(num_features, emb_cardinalities, emb_dim=8, hidden=128, dropout=dropout)
        self.img_proj = nn.Linear(128, 128)
        self.tab_proj = nn.Linear(self.tab.out_dim, 128)
        self.gate = nn.Linear(256, 128)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, img, x_num, x_cats_dict):
        f_img = self.cnn(img)
        f_tab = self.tab(x_num, x_cats_dict)
        f_img = self.img_proj(f_img)
        f_tab = self.tab_proj(f_tab)
        gate = torch.sigmoid(self.gate(torch.cat([f_img, f_tab], dim=1)))
        f = gate * f_img + (1 - gate) * f_tab
        logits = self.head(f)
        return logits.squeeze(1)
