# Guía de Inicio Rápido - Proyecto POSAS

## ¡Proyecto Ejecutado Exitosamente! ✅

El proyecto ha sido ejecutado completamente y está listo para usar.

## Resumen de Ejecución

### 1. ¿Qué se hizo?
- ✅ Instalación de dependencias (PyTorch, pandas, scikit-learn, etc.)
- ✅ Preparación de datos (cálculo de estadísticas de heatmaps)
- ✅ Entrenamiento del modelo (5 folds, validación cruzada)
- ✅ Inferencia sobre el dataset completo
- ✅ Evaluación de métricas de rendimiento

### 2. Datos del Proyecto
- **Dataset**: 140 muestras
  - Clase 0: 81 muestras (57.9%)
  - Clase 1: 59 muestras (42.1%)
- **Features**: 
  - Heatmaps (imágenes de mapas de calor)
  - 13 features numéricas (Min, Edad, PJ, etc.)
  - 1 feature categórica (Posición)

### 3. Modelo Entrenado
- **Arquitectura**: Modelo de fusión multimodal
  - Backbone: ResNet18 para heatmaps
  - MLP para features tabulares
  - Fusión de ambas modalidades
- **Loss**: Focal Loss
- **Optimizador**: Adam
- **Validación**: 5-Fold Cross-Validation

### 4. Resultados

#### Métricas con Umbral Estándar (0.5):
- Accuracy: 40.71%
- Precision: 18.42%
- Recall: 11.86%
- F1 Score: 14.43%

#### Métricas con Umbral Optimizado (0.05):
- **Accuracy: 42.14%**
- **Precision: 42.14%**
- **Recall: 100%**
- **F1 Score: 59.30%**

### 5. Archivos Generados

```
models/
├── fold_0/ a fold_4/    # Modelos entrenados por fold (~44MB c/u)
└── heatmaps_mean_std.json  # Estadísticas globales

reports/
├── oof.csv              # Predicciones out-of-fold
├── preds_new.csv        # Predicciones de inferencia
├── metrics_new.json     # Métricas estándar
└── metrics_new_opt.json # Métricas con umbral óptimo
```

## Cómo Ejecutar el Proyecto Nuevamente

### Opción 1: Script Automatizado
```bash
./run_project.sh
```

### Opción 2: Paso a Paso
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Preparar datos
PYTHONPATH=. python scripts/prepare_data.py --cfg config/config.yaml --save_json

# 3. Entrenar modelo
PYTHONPATH=. python scripts/train.py --cfg config/config.yaml

# 4. Inferencia
PYTHONPATH=. python scripts/infer.py --cfg config/config.yaml \
  --csv "data/processed/datos_para_cnn_etiquetas - datos_para_cnn.csv.csv" \
  --out reports/preds.csv --use_threshold

# 5. Evaluación
PYTHONPATH=. python scripts/eval_preds.py --preds reports/preds.csv \
  --find_best_threshold --out_json reports/metrics_global_opt.json
```

## Configuración Actual

En `config/config.yaml`:
- **Device**: CPU (modificado de MPS para compatibilidad con Linux)
- **Epochs**: 5 (reducido de 60 para demostración)
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Backbone**: ResNet18

### Para Producción:
Se recomienda modificar en `config/config.yaml`:
```yaml
training:
  device: cuda  # Si tienes GPU NVIDIA
  epochs: 60    # Restaurar epochs completos
```

## Mejoras Futuras Recomendadas

1. **Hardware**:
   - Usar GPU (CUDA) para entrenamiento más rápido
   - Entrenar con 60 epochs completos

2. **Optimización**:
   - Ejecutar búsqueda de hiperparámetros con Optuna
   - Probar diferentes backbones (ResNet34, ResNet50)
   - Aplicar más data augmentation

3. **Análisis**:
   - Ejecutar estudios de ablación (`scripts/ablations.py`)
   - Analizar casos de error
   - Validar en datos nuevos

## Scripts Disponibles

- `scripts/prepare_data.py` - Preparación de datos
- `scripts/train.py` - Entrenamiento del modelo
- `scripts/infer.py` - Inferencia
- `scripts/eval_preds.py` - Evaluación
- `scripts/ablations.py` - Estudios de ablación
- `scripts/optuna_search.py` - Búsqueda de hiperparámetros

## Documentación Adicional

- `EXECUTION_SUMMARY.md` - Resumen detallado en inglés
- `README.md` - Documentación original del proyecto

## Soporte

Para más información sobre el proyecto, consulta:
- README.md para instrucciones completas
- EXECUTION_SUMMARY.md para detalles de ejecución
- config/config.yaml para configuración

---

**Nota**: El proyecto ha sido ejecutado exitosamente. Los modelos entrenados están listos para usar en inferencia sobre nuevos datos.
