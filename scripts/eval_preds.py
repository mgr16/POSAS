#!/usr/bin/env python3
# scripts/eval_preds.py
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


def compute_metrics(y, proba, pred):
    roc = None
    pr = None
    if proba is not None and len(np.unique(y)) > 1:
        try:
            roc = float(roc_auc_score(y, proba))
        except Exception:
            roc = None
        try:
            pr = float(average_precision_score(y, proba))
        except Exception:
            pr = None
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc,
        "pr_auc": pr,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "n": int(len(y)),
    }


essage = "Buscar umbral global que maximiza F1 usando columna 'proba'"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Ruta a preds.csv")
    ap.add_argument("--threshold", type=float, default=0.5, help="Umbral si no existe columna 'pred'")
    ap.add_argument("--find_best_threshold", action="store_true", help=essage)
    ap.add_argument("--out_json", default="reports/metrics_global.json")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    if "Target" not in df.columns:
        raise ValueError("preds.csv no contiene columna 'Target'.")

    y = df["Target"].astype(int).values
    proba = df["proba"].values if "proba" in df.columns else None

    if args.find_best_threshold:
        if proba is None:
            raise ValueError("No hay columna 'proba' para buscar el mejor umbral.")
        best_t, best_f1 = None, -1.0
        for t in np.linspace(0.05, 0.95, 181):  # paso 0.005
            pred_t = (proba >= t).astype(int)
            f1 = f1_score(y, pred_t, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        pred = (proba >= best_t).astype(int)
        metrics = compute_metrics(y, proba, pred)
        metrics["threshold_used"] = best_t
        metrics["optimization"] = "global_F1_on_preds.csv"
    else:
        if "pred" in df.columns:
            pred = df["pred"].astype(int).values
            metrics = compute_metrics(y, proba, pred)
            metrics["threshold_used"] = None
            metrics["optimization"] = "pre-thresholded"
        else:
            if proba is None:
                raise ValueError("preds.csv no tiene 'pred' ni 'proba'.")
            pred = (proba >= args.threshold).astype(int)
            metrics = compute_metrics(y, proba, pred)
            metrics["threshold_used"] = float(args.threshold)
            metrics["optimization"] = "fixed_threshold_arg"

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
