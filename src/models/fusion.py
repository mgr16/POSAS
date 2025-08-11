import torch
import torch.nn as nn
from .cnn import SmallCNN
from .mlp import TabularMLP

class FusionModel(nn.Module):
    def __init__(self, num_features: int, emb_cardinalities: dict, dropout: float = 0.2):
        super().__init__()
        self.cnn = SmallCNN(out_dim=64)
        self.tab = TabularMLP(num_features, emb_cardinalities, emb_dim=8, hidden=128, dropout=dropout)
        fused = 64 + self.tab.out_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, img, x_num, x_cats_dict):
        f_img = self.cnn(img)
        f_tab = self.tab(x_num, x_cats_dict)
        f = torch.cat([f_img, f_tab], dim=1)
        logits = self.head(f)
        return logits.squeeze(1)
