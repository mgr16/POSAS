import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = torch.where(targets==1, p, 1-p)
        loss = (1-pt) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = torch.where(targets==1, self.alpha, 1-self.alpha)
            loss = alpha_t * loss
        return loss.mean() if self.reduction=='mean' else loss.sum()
