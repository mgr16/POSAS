#!/usr/bin/env python3
import argparse
import os
import json
import optuna
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold, GroupKFold
from torch.utils.data import DataLoader

from src.config import Config
from src.utils.seeding import set_seed
from src.data.tabular import TabularPreprocessor
from src.data.datasets import MultimodalDataset
from src.models.fusion import FusionModel
from src.train.engine import Engine
from src.train.losses import FocalLoss
from src.train.metrics import all_classification_metrics
from src.train.trainer import Trainer, TrainerConfig


def _resolve_device(device_str: str):
    d = (device_str or "cpu").lower()
    if d == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if d == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _seed_workers(worker_id):
    import random
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def objective(trial, cfg: Config):
    cfg.training.lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    cfg.training.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg.training.dropout = trial.suggest_float("dropout", 0.3, 0.6)

    device = _resolve_device(cfg.training.device)
    set_seed(cfg.seed)

    df = pd.read_csv(cfg.paths.csv)
    df = df.drop(columns=getattr(cfg.features, "drop", []), errors="ignore")
    target_col = "Target"
    heatmap_col = "heatmap_filename"

    groups = df['player_id'].values if getattr(cfg, "use_group_kfold", False) and 'player_id' in df.columns else None
    kf = GroupKFold(n_splits=cfg.n_splits) if groups is not None \
        else StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    fold = 0
    tr_idx, va_idx = next(iter(kf.split(df, df[target_col].values, groups if groups is not None else None)))
    df_tr, df_va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()

    tab = TabularPreprocessor(cfg.features.numeric, cfg.features.categorical)
    tab.fit(df_tr)
    df_tr_num, emb_cards = tab.transform(df_tr)
    df_va_num, _ = tab.transform(df_va)
    numeric_features = tab.numeric

    df_tr_all = df_tr.join(df_tr_num, rsuffix="_num")
    df_va_all = df_va.join(df_va_num, rsuffix="_num")

    cfg_image = cfg.image.__dict__.copy()
    cat_idx_cols = [c + "__idx" for c in cfg.features.categorical]
    ds_tr = MultimodalDataset(df_tr_all, cfg_image=cfg_image, heatmaps_dir=cfg.paths.heatmaps_dir,
                              features_num=numeric_features, cat_idx_cols=cat_idx_cols, target_col=target_col)
    ds_va = MultimodalDataset(df_va_all, cfg_image=cfg_image, heatmaps_dir=cfg.paths.heatmaps_dir,
                              features_num=numeric_features, cat_idx_cols=cat_idx_cols, target_col=target_col)

    is_mps = (device == "mps")
    g = torch.Generator()
    g.manual_seed(cfg.seed + fold)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.training.batch_size, shuffle=True,
                       num_workers=0 if is_mps else 2,
                       pin_memory=False if is_mps else True,
                       persistent_workers=False if is_mps else True,
                       worker_init_fn=_seed_workers, generator=g)
    dl_va = DataLoader(ds_va, batch_size=cfg.training.batch_size, shuffle=False,
                       num_workers=0 if is_mps else 2,
                       pin_memory=False if is_mps else True,
                       persistent_workers=False if is_mps else True)

    model = FusionModel(
        num_features=len(numeric_features),
        emb_cardinalities=emb_cards,
        dropout=cfg.training.dropout,
        backbone=getattr(cfg.training, "backbone", "resnet18"),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=cfg.training.lr, steps_per_epoch=len(dl_tr), epochs=cfg.training.epochs
    )

    loss_fn = FocalLoss(
        gamma=getattr(cfg.loss, "gamma", 2.0),
        alpha=getattr(cfg.loss, "alpha", 0.25),
        label_smoothing=getattr(cfg.training, "label_smoothing", 0.0),
    )

    trainer = Trainer(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        cfg=TrainerConfig(
            epochs=cfg.training.epochs,
            device=device,
            amp=cfg.training.amp and device == "cuda",
            clip_grad_norm=cfg.training.clip_grad_norm,
            mixup_alpha=cfg.image.augment.get("mixup_alpha", 0.0),
            early_stopping_patience=10,
            log_dir=None,
        ),
        scheduler=scheduler,
        run_name=f"trial_{trial.number}",
    )

    result = trainer.fit(dl_tr, dl_va)
    best_state = result.get("best_state")
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    engine = Engine(model, optim, loss_fn, device=device, amp=cfg.training.amp and device == "cuda")
    _, probs, ys = engine.run_epoch(dl_va, train=False)
    metrics = all_classification_metrics(ys, probs, threshold=cfg.threshold)
    return metrics["f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/config.yaml")
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.cfg)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, cfg), n_trials=args.trials)

    os.makedirs(cfg.paths.reports_dir, exist_ok=True)
    out_path = os.path.join(cfg.paths.reports_dir, "optuna_best.json")
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("Best params:", study.best_params)
    print("Best F1:", study.best_value)


if __name__ == "__main__":
    main()
