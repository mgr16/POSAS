import torch
import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, emb_cardinalities: dict, emb_dim: int = 8, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.embs = nn.ModuleDict({k: nn.Embedding(v, emb_dim) for k, v in emb_cardinalities.items()})
        tab_in = in_dim + (emb_dim * len(emb_cardinalities))
        self.mlp = nn.Sequential(
            nn.Linear(tab_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out_dim = hidden//2

    def forward(self, x_num, x_cats_dict):
        if self.embs:
            embs = [self.embs[k](v) for k, v in x_cats_dict.items()]  # cada v: (B,)
            x = torch.cat([x_num] + embs, dim=1)
        else:
            x = x_num
        return self.mlp(x)
