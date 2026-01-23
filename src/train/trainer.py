import importlib.util
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from .engine import Engine
from .callbacks import EarlyStopping


@dataclass
class TrainerConfig:
    epochs: int
    device: str = "cpu"
    amp: bool = False
    clip_grad_norm: float = 1.0
    mixup_alpha: float = 0.0
    early_stopping_patience: int = 10
    log_dir: Optional[str] = None
    use_wandb: bool = False
    use_mlflow: bool = False


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        cfg: TrainerConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        run_name: Optional[str] = None,
    ):
        self.cfg = cfg
        self.engine = Engine(
            model,
            optimizer,
            loss_fn,
            device=cfg.device,
            amp=cfg.amp,
            clip_grad_norm=cfg.clip_grad_norm,
            mixup_alpha=cfg.mixup_alpha,
        )
        self.scheduler = scheduler
        self.early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="min")
        self.writer = SummaryWriter(log_dir=cfg.log_dir) if cfg.log_dir else None
        self.run_name = run_name or "run"
        self.wandb = self._init_wandb()
        self.mlflow = self._init_mlflow()

    def _init_wandb(self):
        if not self.cfg.use_wandb:
            return None
        if importlib.util.find_spec("wandb") is None:
            return None
        import wandb
        return wandb

    def _init_mlflow(self):
        if not self.cfg.use_mlflow:
            return None
        if importlib.util.find_spec("mlflow") is None:
            return None
        import mlflow
        return mlflow

    def _log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, step)
        if self.wandb:
            self.wandb.log(metrics, step=step)
        if self.mlflow:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.mlflow.log_metric(k, v, step=step)

    def fit(self, train_loader, val_loader) -> Dict[str, Any]:
        best_state = None
        best_loss = float("inf")

        for epoch in range(self.cfg.epochs):
            train_loss, _, _ = self.engine.run_epoch(train_loader, train=True, scheduler=self.scheduler)
            val_loss, probs, ys = self.engine.run_epoch(val_loader, train=False)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            try:
                from .metrics import all_classification_metrics
                metrics = all_classification_metrics(ys, probs, threshold=0.5)
            except Exception:
                metrics = {}

            metrics.update({"train_loss": train_loss, "val_loss": val_loss})
            self._log_metrics(metrics, step=epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu() for k, v in self.engine.model.state_dict().items()}

            if not self.early_stopper.step(val_loss) and self.early_stopper.should_stop():
                break

        return {"best_state": best_state, "best_loss": best_loss}
