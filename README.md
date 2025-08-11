# Proyecto ML multimodal (heatmaps + tabular)

## Requisitos

- Python 3.10+
- macOS con MPS opcional (o CUDA si disponible)

Instalar dependencias en un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

1) Preparar datos (opcional: calcula mean/std global de heatmaps):

```bash
python scripts/prepare_data.py --cfg config/config.yaml --save_json
```

2) Entrenar (KFold, guarda artefactos por fold y OOF):

```bash
PYTHONPATH=. python scripts/train.py --cfg config/config.yaml
```

3) Inferir sobre un CSV y guardar predicciones:

```bash
PYTHONPATH=. python scripts/infer.py --cfg config/config.yaml \
  --csv data/processed/datos_para_cnn_etiquetas\ -\ datos_para_cnn.csv.csv \
  --out reports/preds.csv --use_threshold
```

4) Evaluación global con umbral fijo o búsqueda de umbral óptimo:

```bash
# métrica con el 'pred' del CSV (pre-umbralizado por ensemble)
PYTHONPATH=. python scripts/eval_preds.py --preds reports/preds.csv --out_json reports/metrics_global.json

# busca umbral global que maximiza F1 usando la columna 'proba'
PYTHONPATH=. python scripts/eval_preds.py --preds reports/preds.csv \
  --find_best_threshold --out_json reports/metrics_global_opt.json
```

5) Evaluación OOF (out-of-fold, sin sesgo):

```bash
python - << 'PY'
import pandas as pd
p = 'reports/oof.csv'
oof = pd.read_csv(p).rename(columns={'oof_proba':'proba'})
oof.to_csv('reports/oof_eval.csv', index=False)
print('oof_eval.csv listo')
PY

PYTHONPATH=. python scripts/eval_preds.py --preds reports/oof_eval.csv \
  --find_best_threshold --out_json reports/metrics_oof.json
```

## Notas

- Las categóricas usan UNK=0 y los embeddings incluyen el índice 0.
- Si tienes múltiples filas por entidad, considera `use_group_kfold: true` y define `player_id`.
- AMP solo en CUDA; en macOS/MPS déjalo desactivado.
- No versionar datos ni modelos; usa Git LFS si deseas subir pesos.

## Entrenamiento
```bash
python scripts/train.py --cfg config/config.yaml
```

## Ablaciones

```bash
python scripts/ablations.py --cfg config/config.yaml --mode tabular_only
python scripts/ablations.py --cfg config/config.yaml --mode heatmap_only
python scripts/ablations.py --cfg config/config.yaml --mode fusion
```

## Inferencia

```bash
python scripts/infer.py --cfg config/config.yaml --csv data/processed/nuevo.csv --out preds.csv
```
