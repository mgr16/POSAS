#!/usr/bin/env python3
# scripts/prepare_data.py
# Auditoría de columnas, distribución de Target, verificación de heatmaps y mean/std global de heatmaps.

import argparse, os, json, sys
import numpy as np
import pandas as pd

from src.config import Config


def welford_online_mean_std():
    n = 0
    mean = 0.0
    M2 = 0.0
    def update(x):
        nonlocal n, mean, M2
        for xi in x.reshape(-1):
            n += 1
            delta = xi - mean
            mean += delta / n
            delta2 = xi - mean
            M2 += delta * delta2
    def finalize():
        if n < 2:
            return float(mean), float(0.0)
        variance = M2 / (n - 1)
        return float(mean), float(np.sqrt(variance))
    return update, finalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="Si >0, limita el número de heatmaps a escanear")
    ap.add_argument("--save_json", action="store_true", help="Guardar mean/std en JSON dentro de artifacts_dir")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.cfg)
    csv_path = cfg.paths.csv
    hm_dir = cfg.paths.heatmaps_dir

    print(f"CSV: {csv_path}\nHeatmaps dir: {hm_dir}")

    if not os.path.exists(csv_path):
        print(f"[ERROR] No existe CSV: {csv_path}", file=sys.stderr); sys.exit(1)
    if not os.path.isdir(hm_dir):
        print(f"[ERROR] No existe directorio de heatmaps: {hm_dir}", file=sys.stderr); sys.exit(1)

    df = pd.read_csv(csv_path)

    needed_cols = list(cfg.features.numeric) + list(cfg.features.categorical) + ["heatmap_filename", "Target"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        print(f"[ADVERTENCIA] Faltan columnas: {missing}")

    if "Target" in df.columns:
        vc = df["Target"].value_counts(dropna=False).to_dict()
        total = int(df.shape[0])
        print("\nDistribución Target:")
        for k, v in vc.items():
            p = 100.0 * v / total if total else 0.0
            print(f"  {k}: {v} ({p:.1f}%)")
    else:
        print("\n[INFO] Columna Target no encontrada; se omitirá distribución.")

    update, finalize = welford_online_mean_std()
    missing_files = []
    checked = 0

    import cv2

    for i, row in df.iterrows():
        fname = str(row.get("heatmap_filename", ""))
        if not fname or fname.lower() == "nan":
            missing_files.append((i, fname)); continue

        paths = [os.path.join(hm_dir, fname.replace(".jpg", ".npy")),
                 os.path.join(hm_dir, fname)]
        arr = None
        if os.path.exists(paths[0]):
            try:
                arr = np.load(paths[0], mmap_mode="r")
                arr = np.array(arr, dtype=np.float32)
            except Exception as e:
                print(f"[WARN] Error leyendo NPY {paths[0]}: {e}")
        if arr is None and os.path.exists(paths[1]):
            img = cv2.imread(paths[1], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                arr = img.astype(np.float32)

        if arr is None:
            missing_files.append((i, fname))
            continue

        arr = cv2.resize(arr, (cfg.image.size, cfg.image.size), interpolation=cv2.INTER_AREA)
        update(arr)

        checked += 1
        if args.limit > 0 and checked >= args.limit:
            break

    print(f"\nHeatmaps verificados: {checked}")
    if missing_files:
        print(f"Faltan {len(missing_files)} heatmaps (primeros 10): {missing_files[:10]}")

    mean, std = finalize()
    if checked == 0:
        print("\n[ERROR] No se pudo calcular mean/std (0 heatmaps válidos).")
    else:
        print(f"\nMean global: {mean:.6f} | Std global: {std:.6f}")
        if args.save_json:
            os.makedirs(cfg.paths.artifacts_dir, exist_ok=True)
            out_json = os.path.join(cfg.paths.artifacts_dir, "heatmaps_mean_std.json")
            with open(out_json, "w") as f:
                json.dump({"mean": mean, "std": std, "size": cfg.image.size, "count": checked}, f, indent=2)
            print(f"[OK] Guardado: {out_json}")

if __name__ == "__main__":
    main()
