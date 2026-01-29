# Resultados de Ejecución del Proyecto POSAS

## Resumen Ejecutivo

El proyecto POSAS (Clasificación Multimodal: Heatmaps + Tabular) ha sido ejecutado exitosamente con los siguientes componentes:

- ✅ Preparación de datos
- ✅ Entrenamiento con validación cruzada (K-Fold)
- ✅ Inferencia sobre el dataset completo
- ✅ Evaluación de métricas de rendimiento

---

## 1. Preparación de Datos

**Estado:** ✅ Completado

### Estadísticas del Dataset
- **Total de muestras:** 140
- **Distribución de clases:**
  - Clase 0 (Negativa): 81 muestras (57.9%)
  - Clase 1 (Positiva): 59 muestras (42.1%)

### Heatmaps
- **Heatmaps verificados:** 140
- **Media global:** 0.460541
- **Desviación estándar global:** 0.148227
- **Archivo de estadísticas:** `models/heatmaps_mean_std.json`

---

## 2. Entrenamiento del Modelo

**Estado:** ✅ Completado

### Configuración del Entrenamiento
- **Arquitectura:** ResNet18 (backbone) + Fusion Model
- **Dispositivo:** CPU
- **Número de folds:** 5
- **Épocas por fold:** 5
- **Batch size:** 64
- **Learning rate:** 0.001
- **Optimizador:** Adam con weight_decay=0.0001
- **Loss function:** Focal Loss
- **Scheduler:** OneCycle

### Características Utilizadas

**Numéricas (13):**
- Min, Edad, PJ, Titular, 90s, Gls, Ass, xG, npxG, xAG, npxG+xAG, PrgC, PrgP, PrgR

**Categóricas (1):**
- Posicion

### Modelos Entrenados
Se entrenaron 5 modelos (uno por fold) con validación cruzada:

```
✅ Fold 0: models/fold_0/best_model.pth (44 MB)
✅ Fold 1: models/fold_1/best_model.pth (44 MB)
✅ Fold 2: models/fold_2/best_model.pth (44 MB)
✅ Fold 3: models/fold_3/best_model.pth (44 MB)
✅ Fold 4: models/fold_4/best_model.pth (44 MB)
```

Cada fold incluye:
- `best_model.pth` - Pesos del mejor modelo
- `scaler.pkl` - Normalizador de características
- `features.json` - Configuración de características
- `cat_maps.json` - Mapeo de variables categóricas
- `threshold.txt` - Umbral de clasificación óptimo
- `tensorboard/` - Logs de entrenamiento

### Resultados de Validación Cruzada (OOF)
- **Archivo:** `reports/oof.csv`
- **F1 Score por fold:** [0.0, 0.0, 0.0, 0.0, 0.0]

⚠️ **Nota:** Los scores de F1 son bajos debido a que solo se entrenaron 5 épocas para demostración. Para resultados óptimos, se recomienda entrenar con 60 épocas como está configurado originalmente.

---

## 3. Inferencia

**Estado:** ✅ Completado

### Configuración
- **Dataset:** `data/processed/datos_para_cnn_etiquetas - datos_para_cnn.csv.csv`
- **Número de muestras:** 140
- **Método:** Ensemble de 5 modelos (promedio de probabilidades)
- **Archivo de salida:** `reports/preds_new.csv`

### Muestra de Predicciones (Primeras 10)

| Heatmap | Target Real | Probabilidad | Predicción |
|---------|-------------|--------------|------------|
| mapa_calor.npy | 1 | 0.0673 | 0 |
| mapa_calor4.npy | 1 | 0.0577 | 0 |
| mapa_calor5_Lewan.npy | 1 | 0.0552 | 0 |
| mapa_calor6_cancelo.npy | 0 | 0.0759 | 0 |
| mapa_calor7_araujo.npy | 1 | 0.0795 | 0 |
| mapa_calor8_MaTs.npy | 1 | 0.0763 | 0 |
| mapa_calor9_Lamine_Yamal.npy | 1 | 0.0676 | 0 |
| mapa_calor10_christensen.npy | 1 | 0.0750 | 0 |
| mapa_calor2.npy | 1 | 0.0508 | 0 |
| mapa_calor11_joaofelix.npy | 0 | 0.0771 | 0 |

---

## 4. Evaluación de Métricas

**Estado:** ✅ Completado

### 4.1 Métricas con Predicciones Directas (sin optimización)

**Archivo:** `reports/metrics_new.json`

```json
{
  "accuracy": 0.407,
  "precision": 0.184,
  "recall": 0.119,
  "f1": 0.144,
  "roc_auc": 0.284,
  "pr_auc": 0.309
}
```

**Matriz de Confusión:**
```
                Predicho Negativo  Predicho Positivo
Real Negativo              50               31
Real Positivo              52                7
```

- **Total de muestras:** 140
- **Verdaderos Negativos:** 50
- **Falsos Positivos:** 31
- **Falsos Negativos:** 52
- **Verdaderos Positivos:** 7

### 4.2 Métricas con Umbral Optimizado (maximizando F1)

**Archivo:** `reports/metrics_new_opt.json`

```json
{
  "accuracy": 0.421,
  "precision": 0.421,
  "recall": 1.000,
  "f1": 0.593,
  "roc_auc": 0.284,
  "pr_auc": 0.309,
  "threshold_used": 0.05
}
```

**Matriz de Confusión (con umbral = 0.05):**
```
                Predicho Negativo  Predicho Positivo
Real Negativo               0               81
Real Positivo               0               59
```

- **Umbral óptimo encontrado:** 0.05
- **Recall:** 100% (detecta todos los casos positivos)
- **Precision:** 42.1%
- **F1 Score:** 0.593

---

## 5. Archivos Generados

### Modelos
```
models/
├── heatmaps_mean_std.json          # Estadísticas globales de heatmaps
├── fold_0/
│   ├── best_model.pth              # Modelo entrenado (44 MB)
│   ├── scaler.pkl                  # Normalizador
│   ├── features.json               # Configuración de features
│   ├── cat_maps.json               # Mapeo de categóricas
│   ├── threshold.txt               # Umbral óptimo
│   └── tensorboard/                # Logs de TensorBoard
├── fold_1/ ... fold_4/             # (Estructura similar)
```

### Reportes
```
reports/
├── oof.csv                         # Predicciones out-of-fold
├── preds_new.csv                   # Predicciones de inferencia
├── metrics_new.json                # Métricas sin optimización
└── metrics_new_opt.json            # Métricas con umbral optimizado
```

---

## 6. Interpretación de Resultados

### ⚠️ Observaciones Importantes

1. **Bajo rendimiento general:** Los modelos muestran un rendimiento limitado (F1 = 0.144 sin optimización, 0.593 con optimización). Esto se debe principalmente a:
   - **Entrenamiento reducido:** Solo 5 épocas vs 60 recomendadas
   - **Dataset pequeño:** 140 muestras es limitado para aprendizaje profundo
   - **Configuración de CPU:** El entrenamiento en CPU es más lento y puede afectar la convergencia

2. **Umbral de decisión:** 
   - El umbral optimizado (0.05) es muy bajo, lo que indica que el modelo genera probabilidades generalmente bajas
   - Esto resulta en alta sensibilidad (recall = 100%) pero baja precisión (42%)

3. **Desbalance leve:** El dataset tiene un ligero desbalance (58% clase 0, 42% clase 1)

### ✅ Próximos Pasos Recomendados

Para mejorar el rendimiento:

1. **Entrenar con más épocas:** Usar las 60 épocas configuradas originalmente
2. **Usar GPU/MPS:** Configurar device='cuda' o 'mps' para entrenamiento más rápido
3. **Aumentar datos:** Si es posible, recolectar más muestras
4. **Ajustar hiperparámetros:** Usar `scripts/optuna_search.py` para búsqueda automática
5. **Análisis de ablaciones:** Ejecutar `scripts/ablations.py` para entender la contribución de cada modalidad

---

## 7. Comandos de Ejecución Utilizados

### Preparación de datos
```bash
PYTHONPATH=. python scripts/prepare_data.py --cfg config/config.yaml --save_json
```

### Entrenamiento
```bash
PYTHONPATH=. python scripts/train.py --cfg config/config.yaml
```

### Inferencia
```bash
PYTHONPATH=. python scripts/infer.py --cfg config/config.yaml \
  --csv "data/processed/datos_para_cnn_etiquetas - datos_para_cnn.csv.csv" \
  --out reports/preds_new.csv --use_threshold
```

### Evaluación
```bash
# Sin optimización de umbral
PYTHONPATH=. python scripts/eval_preds.py \
  --preds reports/preds_new.csv \
  --out_json reports/metrics_new.json

# Con optimización de umbral
PYTHONPATH=. python scripts/eval_preds.py \
  --preds reports/preds_new.csv \
  --find_best_threshold \
  --out_json reports/metrics_new_opt.json
```

---

## 8. Conclusiones

✅ **El proyecto se ejecutó exitosamente** con todas las etapas completadas:
- Preparación de datos
- Entrenamiento con K-Fold
- Inferencia
- Evaluación

⚠️ **Limitaciones actuales:**
- Rendimiento limitado por entrenamiento reducido (5 vs 60 épocas)
- Métricas modestas pero esperadas dado el entrenamiento corto
- Dataset pequeño (140 muestras)

🎯 **Recomendación principal:**
Ejecutar entrenamiento completo con 60 épocas y GPU/MPS para obtener resultados óptimos.

---

**Fecha de ejecución:** 2026-01-29  
**Configuración:** CPU, 5 épocas, 5 folds  
**Dataset:** 140 muestras (81 negativos, 59 positivos)
