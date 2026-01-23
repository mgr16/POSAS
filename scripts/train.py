#!/usr/bin/env python3
# scripts/train.py

import argparse, os, json
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
from src.train.metrics import all_classification_metrics
from src.train.losses import FocalLoss
from src.train.trainer import Trainer, TrainerConfig


def _resolve_device(device_str: str):
    d = (device_str or "cpu").lower()
    if d == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if d == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _check_columns(df: pd.DataFrame, cols: list, ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {ctx}: {missing}")


def _seed_workers(worker_id):
    # Inicializa semilla por worker para reproducibilidad
    import random, numpy as np, torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def _load_heatmap_stats(artifacts_dir: str):
    path = os.path.join(artifacts_dir, "heatmaps_mean_std.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            j = json.load(f)
        return float(j.get("mean", None)), float(j.get("std", None))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='config/config.yaml')
    args = ap.parse_args()

    cfg = Config.from_yaml(args.cfg)
    device = _resolve_device(cfg.training.device)
    set_seed(cfg.seed)

    df = pd.read_csv(cfg.paths.csv)
    df = df.drop(columns=getattr(cfg.features, "drop", []), errors='ignore')

    target_col = "Target"
    heatmap_col = "heatmap_filename"
    base_needed = list(cfg.features.numeric) + list(cfg.features.categorical) + [target_col, heatmap_col]
    _check_columns(df, base_needed, "dataset base")

    # Stats globales de heatmaps (si existen)
    hm_mean, hm_std = _load_heatmap_stats(cfg.paths.artifacts_dir)

    groups = df['player_id'].values if getattr(cfg, "use_group_kfold", False) and 'player_id' in df.columns else None
    kf = GroupKFold(n_splits=cfg.n_splits) if groups is not None \
         else StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    fold_metrics = []
    oof_rows = []  # acumulador OOF

    for fold, (tr, va) in enumerate(kf.split(df, df[target_col].values, groups if groups is not None else None)):
        df_tr, df_va = df.iloc[tr].copy(), df.iloc[va].copy()

        # ---- Preprocesado por fold (evita fuga) ----
        tab = TabularPreprocessor(cfg.features.numeric, cfg.features.categorical)
        tab.fit(df_tr)
        df_tr_num, emb_cards = tab.transform(df_tr)
        df_va_num, _         = tab.transform(df_va)
        if not emb_cards:
            emb_cards = {k: len(v) for k, v in tab.cat_maps.items()}
        numeric_features = tab.numeric

        df_tr_all = df_tr.join(df_tr_num, rsuffix="_num")
        df_va_all = df_va.join(df_va_num, rsuffix="_num")

        # Guardar artefactos del fold
        fold_dir = os.path.join(cfg.paths.artifacts_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        tab.save(os.path.join(fold_dir, "scaler.pkl"),
                 os.path.join(fold_dir, "cat_maps.json"))
        with open(os.path.join(fold_dir, "features.json"), "w") as f:
            json.dump({"numeric": numeric_features,
                       "categorical": cfg.features.categorical,
                       "drop": getattr(cfg.features, "drop", []),
                       "target": target_col}, f, indent=2)

        # ---- Datasets y DataLoaders ----
        # Inyecta mean/std globales si existen
        cfg_image = cfg.image.__dict__.copy()
        if hm_mean is not None and hm_std is not None:
            cfg_image["mean"] = hm_mean
            cfg_image["std"] = hm_std

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

        # ---- Modelo, optimizador, pérdida ----
        model = FusionModel(num_features=len(numeric_features),
                            emb_cardinalities=emb_cards,
                            dropout=cfg.training.dropout,
                            backbone=getattr(cfg.training, "backbone", "resnet18"))
        if getattr(cfg.training, "compile", False) and hasattr(torch, "compile") and device == "cuda":
            model = torch.compile(model)
        model = model.to(device)

        optim = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

        # Scheduler (opcional)
        scheduler = None
        if getattr(cfg.training, "scheduler", "none") == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim, max_lr=cfg.training.lr,
                steps_per_epoch=len(dl_tr), epochs=cfg.training.epochs
            )
        elif getattr(cfg.training, "scheduler", "none") == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="max", factor=0.5, patience=3, verbose=True
            )

        # pos_weight por fold (neg/pos)
        pos = int((df_tr[target_col] == 1).sum())
        neg = int((df_tr[target_col] == 0).sum())
        pw  = neg / max(pos, 1)
        pos_weight = torch.tensor([pw], device=device)
        loss_type = getattr(cfg, "loss", None).type if getattr(cfg, "loss", None) else "bce"
        if loss_type == "focal_loss":
            loss_fn = FocalLoss(
                gamma=getattr(cfg.loss, "gamma", 2.0),
                alpha=getattr(cfg.loss, "alpha", 0.25),
                label_smoothing=getattr(cfg.training, "label_smoothing", 0.0),
            )
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        use_amp = (cfg.training.amp and device == "cuda")
        mixup_alpha = cfg.image.augment.get("mixup_alpha", 0.0)
        trainer = Trainer(
            model=model,
            optimizer=optim,
            loss_fn=loss_fn,
            cfg=TrainerConfig(
                epochs=cfg.training.epochs,
                device=device,
                amp=use_amp,
                clip_grad_norm=cfg.training.clip_grad_norm,
                mixup_alpha=mixup_alpha,
                early_stopping_patience=10,
                log_dir=os.path.join(fold_dir, "tensorboard"),
                use_wandb=bool(os.environ.get("WANDB_PROJECT")),
                use_mlflow=bool(os.environ.get("MLFLOW_TRACKING_URI")),
            ),
            scheduler=scheduler,
            run_name=f"fold_{fold}",
        )

        # ---- Entrenamiento con EarlyStopping ----
        result = trainer.fit(dl_tr, dl_va)
        best_state = result.get("best_state")
        if best_state is None:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(best_state, os.path.join(fold_dir, 'best_model.pth'))

        # Reconstruye probs de validación con mejor estado (para OOF)
        model.load_state_dict(best_state)
        model.eval()
        engine = Engine(model, optim, loss_fn, device=device, amp=use_amp, clip_grad_norm=cfg.training.clip_grad_norm)
        _, probs_va, ys_va = engine.run_epoch(dl_va, train=False)
        df_oof = pd.DataFrame({
            "index": df_va_all.index.values,
            "Target": df_va_all[target_col].values.astype(int),
            "oof_proba": probs_va,
            "fold": fold
        })
        oof_rows.append(df_oof)

        # Threshold óptimo por fold (máx F1)
        from sklearn.metrics import f1_score
        best_t, best_t_f1 = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(ys_va, (probs_va >= t).astype(int), zero_division=0)
            if f1 > best_t_f1:
                best_t_f1, best_t = f1, t
        with open(os.path.join(fold_dir, "threshold.txt"), "w") as f:
            f.write(str(best_t))

        metrics = all_classification_metrics(ys_va, probs_va, threshold=cfg.threshold)
        fold_metrics.append((fold, float(metrics["f1"])))

    # Guardar OOF al final
    if oof_rows:
        oof = pd.concat(oof_rows, axis=0).sort_values("index")
        os.makedirs(cfg.paths.reports_dir, exist_ok=True)
        oof_path = os.path.join(cfg.paths.reports_dir, "oof.csv")
        oof.to_csv(oof_path, index=False)
        print(f"[OK] OOF guardado en {oof_path}")

    print("\nResumen F1 por fold:", fold_metrics)

if __name__ == '__main__':
    main()
