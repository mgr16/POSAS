import torch
import numpy as np
from typing import Dict, Any, Tuple
try:
    from torch.cuda.amp import autocast, GradScaler  # CUDA
    has_cuda_amp = True
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def autocast(enabled=True):  # no-op
        yield
    class GradScaler:            # no-op
        def __init__(self, enabled=False): self.enabled = False
        def scale(self, x): return x
        def unscale_(self, *args, **kwargs): pass
        def step(self, optim): optim.step()
        def update(self): pass
    has_cuda_amp = False

class Engine:
    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 device: str = "cpu",
                 amp: bool = True,
                 clip_grad_norm: float = 1.0,
                 mixup_alpha: float = 0.0):
        self.model = model.to(device)
        self.optim = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.amp = (amp and has_cuda_amp and device == "cuda")
        self.scaler = GradScaler(enabled=self.amp)
        self.clip = clip_grad_norm
        self.mixup_alpha = mixup_alpha

    # -------------------------
    # Helpers
    # -------------------------
    def _to_device_float(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device, non_blocking=True).float()

    def _to_device_long(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device, non_blocking=True).long()

    def _cat_names_from_model(self) -> list:
        try:
            embs = getattr(getattr(self.model, "tab", None), "embs", None)
            if embs is not None:
                return list(embs.keys())
        except Exception:
            pass
        return []

    def _build_cats_dict(self, cats: Any) -> Dict[str, torch.Tensor]:
        if cats is None:
            return {}
        if isinstance(cats, dict):
            out = {}
            for k, v in cats.items():
                if not torch.is_tensor(v): v = torch.as_tensor(v)
                if v.dim() == 0: v = v.unsqueeze(0)
                out[k] = self._to_device_long(v)
            return out
        if torch.is_tensor(cats):
            if cats.dim() == 1: cats = cats.unsqueeze(1)
            cats = cats.to(self.device, non_blocking=True)
            B, C = cats.shape
            names = self._cat_names_from_model()
            if len(names) < C:
                names = names + [f"cat_{i}" for i in range(len(names), C)]
            names = names[:C]
            out = {}
            for i, name in enumerate(names):
                v = cats[:, i]
                if v.dtype != torch.long: v = v.long()
                out[name] = v
            return out
        return {}

    # -------------------------
    # Core steps
    # -------------------------
    def _step(self, batch: Dict[str, Any], train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = batch["img"].to(self.device, non_blocking=True)
        tab = batch["tab"].to(self.device, non_blocking=True).float()
        cats_dict = self._build_cats_dict(batch.get("cats", None))
        y = batch["y"].to(self.device).float()

        if train and self.mixup_alpha and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            perm = torch.randperm(img.size(0), device=img.device)
            img = lam * img + (1 - lam) * img[perm]
            tab = lam * tab + (1 - lam) * tab[perm]
            y = lam * y + (1 - lam) * y[perm]
            # Categorías: mantiene las originales para evitar interpolaciones inválidas

        with autocast(enabled=self.amp):
            logits = self.model(img, tab, cats_dict)
            loss = self.loss_fn(logits, y)

        if train:
            self.optim.zero_grad(set_to_none=True)
            if self.amp:
                self.scaler.scale(loss).backward()
                if self.clip:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optim.step()

        probs = torch.sigmoid(logits.detach())
        return loss.detach(), probs, y.detach()

    def run_epoch(self, loader, train: bool = True, scheduler=None):
        self.model.train(mode=train)
        losses, probs_all, ys_all = [], [], []
        for batch in loader:
            loss, probs, ys = self._step(batch, train=train)
            if train and scheduler is not None:
                # OneCycleLR / Cosine / StepLR: step por batch
                try:
                    scheduler.step()
                except TypeError:
                    # ReduceLROnPlateau requiere .step(metric) al final de la época
                    pass
            losses.append(loss.item())
            probs_all.append(probs.cpu())
            ys_all.append(ys.cpu())

        probs_all = torch.cat(probs_all).numpy()
        ys_all = torch.cat(ys_all).numpy()
        mean_loss = float(sum(losses) / max(len(losses), 1))
        return mean_loss, probs_all, ys_all

    @torch.no_grad()
    def evaluate(self, loader, threshold: float = 0.5):
        self.model.eval()
        _, probs, ys = self.run_epoch(loader, train=False)
        try:
            from .metrics import all_classification_metrics
            return all_classification_metrics(ys, probs, threshold)
        except Exception:
            return {"loss": None, "probs": probs, "ys": ys}
