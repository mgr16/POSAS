#!/usr/bin/env python3
# scripts/infer.py
# Ensemble por folds usando artefactos guardados (scaler.pkl, cat_maps.json, best_model.pth, features.json)

import argparse, os, json, warnings
import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data.datasets import MultimodalDataset
from src.models.fusion import FusionModel

def _resolve_device(device_str: str):
    device_str = (device_str or "cpu").lower()
    if device_str == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        warnings.warn("MPS no disponible; usando CPU.")
        return "cpu"
    if device_str == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        warnings.warn("CUDA no disponible; usando CPU.")
        return "cpu"
    return "cpu"

def _require(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {what}: {path}. ¿Guardaste los artefactos por fold durante el entrenamiento?")
    return path

def _load_features_for_fold(fold_dir: str, cfg_features: dict):
    fpath = os.path.join(fold_dir, "features.json")
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            return json.load(f)
    warnings.warn(f"features.json no existe en {fold_dir}; usando features de config.yaml (verifica consistencia).")
    return {
        "numeric": cfg_features.get("numeric", []),
        "categorical": cfg_features.get("categorical", []),
        "drop": cfg_features.get("drop", []),
        "target": "Target"
    }

def _load_cat_maps(fold_dir: str):
    p = _require(os.path.join(fold_dir, "cat_maps.json"), "cat_maps.json")
    with open(p, "r") as f:
        return json.load(f)

def _load_scaler(fold_dir: str):
    import pickle
    p = _require(os.path.join(fold_dir, "scaler.pkl"), "scaler.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)

def _check_columns(df: pd.DataFrame, needed: list, ctx: str):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {ctx}: {missing}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/config.yaml")
    ap.add_argument("--csv", required=True, help="CSV de entrada para inferencia")
    ap.add_argument("--out", default="preds.csv", help="Ruta de salida (CSV con probabilidades)")
    ap.add_argument("--use_threshold", action="store_true", help="Si está presente, genera columna 'pred' usando threshold (per-fold si existe o config.threshold).")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.cfg)
    device = _resolve_device(cfg.training.device)

    df = pd.read_csv(args.csv)
    needed_cols = list(cfg.features.numeric) + list(cfg.features.categorical) + ["heatmap_filename"]
    _check_columns(df, needed_cols, "CSV de inferencia")

    folds_root = cfg.paths.artifacts_dir
    if not os.path.isdir(folds_root):
        raise FileNotFoundError(f"No existe la carpeta de artefactos: {folds_root}")
    fold_dirs = sorted([os.path.join(folds_root, d) for d in os.listdir(folds_root)
                        if d.startswith("fold_") and os.path.isdir(os.path.join(folds_root, d))])
    if not fold_dirs:
        raise FileNotFoundError(f"No se encontraron directorios fold_* en {folds_root}. ¿Ejecutaste el entrenamiento KFold?")

    probs_folds, thresholds = [], []

    for fold_dir in fold_dirs:
        features = _load_features_for_fold(fold_dir, {
            "numeric": cfg.features.numeric,
            "categorical": cfg.features.categorical,
            "drop": cfg.features.drop
        })
        cat_maps = _load_cat_maps(fold_dir)
        scaler = _load_scaler(fold_dir)

        df_fold = df.copy()
        needed = list(features["numeric"]) + list(features["categorical"]) + ["heatmap_filename"]
        _check_columns(df_fold, needed, f"CSV para {fold_dir}")

        # Transform numéricas
        X_num = pd.DataFrame(scaler.transform(df_fold[features["numeric"]]),
                             columns=features["numeric"], index=df_fold.index)
        # Encoders categóricos
        for col in features["categorical"]:
            mapping = cat_maps.get(col, {})
            X_num[col + "__idx"] = df_fold[col].astype(str).map(mapping).fillna(0).astype(int)

        df_all = df_fold.join(X_num, rsuffix="_num")

        # Target dummy si no está (no se usa en inferencia)
        target_col = features.get("target", "Target")
        if target_col not in df_all.columns:
            df_all[target_col] = 0

        from torch.utils.data import DataLoader
        ds = MultimodalDataset(df_all,
                               cfg_image=cfg.image.__dict__,
                               heatmaps_dir=cfg.paths.heatmaps_dir,
                               features_num=features["numeric"],
                               cat_idx_cols=[c + "__idx" for c in features["categorical"]] if "categorical" in features else [c + "__idx" for c in features["categorical"]],
                               target_col=target_col)

        is_mps = (device == "mps")
        dl = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=0 if is_mps else 2,
                        pin_memory=False if is_mps else True,
                        persistent_workers=False if is_mps else True)

        emb_cards = {k: len(v) for k, v in cat_maps.items()}
        model = FusionModel(num_features=len(features["numeric"]),
                            emb_cardinalities=emb_cards,
                            dropout=0.0).to(device)
        weights_path = _require(os.path.join(fold_dir, "best_model.pth"), "best_model.pth")
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()

        thr_path = os.path.join(fold_dir, "threshold.txt")
        fold_thr = None
        if os.path.exists(thr_path):
            try:
                with open(thr_path, "r") as f:
                    fold_thr = float(f.read().strip())
            except Exception:
                fold_thr = None
        thresholds.append(fold_thr)

        all_probs = []
        with torch.no_grad():
            for batch in dl:
                img = batch["img"].to(device)
                tab = batch["tab"].to(device)
                cats = batch["cats"]
                if isinstance(cats, dict):
                    cats_dict = {k: v.to(device) for k, v in cats.items()}
                else:
                    if cats.dim() == 1:
                        cats = cats.unsqueeze(1)
                    cats = cats.to(device)
                    cat_names = features["categorical"]
                    cats_dict = {cat_names[i]: cats[:, i] for i in range(cats.shape[1])}
                logits = model(img, tab, cats_dict)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_probs.append(probs)

        probs_folds.append(np.concatenate(all_probs, axis=0))

    probs = np.mean(np.stack(probs_folds, axis=0), axis=0)

    out = df.copy()
    out["proba"] = probs

    if args.use_threshold:
        valid_thrs = [t for t in thresholds if isinstance(t, (float, int))]
        thr = float(np.mean(valid_thrs)) if valid_thrs else float(cfg.threshold)
        out["pred"] = (out["proba"] >= thr).astype(int)

    out_path = args.out
    out.to_csv(out_path, index=False)
    print(f"[OK] Inferencia escrita en: {out_path}")

if __name__ == "__main__":
    main()
