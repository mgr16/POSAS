#!/bin/bash
# run_project.sh - Script to execute the POSAS project end-to-end

set -e  # Exit on error

echo "================================================"
echo "POSAS Project - Multimodal ML Execution Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Prepare data
echo ""
echo "Step 1: Preparing data and calculating heatmap statistics..."
PYTHONPATH=. python scripts/prepare_data.py --cfg config/config.yaml --save_json

# Train model
echo ""
echo "Step 2: Training model with K-Fold cross-validation..."
PYTHONPATH=. python scripts/train.py --cfg config/config.yaml

# Run inference
echo ""
echo "Step 3: Running inference on dataset..."
PYTHONPATH=. python scripts/infer.py --cfg config/config.yaml \
  --csv "data/processed/datos_para_cnn_etiquetas - datos_para_cnn.csv.csv" \
  --out reports/preds.csv --use_threshold

# Evaluate predictions
echo ""
echo "Step 4: Evaluating predictions..."
PYTHONPATH=. python scripts/eval_preds.py --preds reports/preds.csv \
  --out_json reports/metrics_global.json

echo ""
echo "Step 5: Finding optimal threshold..."
PYTHONPATH=. python scripts/eval_preds.py --preds reports/preds.csv \
  --find_best_threshold --out_json reports/metrics_global_opt.json

# Display results
echo ""
echo "================================================"
echo "Execution Complete!"
echo "================================================"
echo ""
echo "Results saved to:"
echo "  - Models: models/fold_*/"
echo "  - OOF predictions: reports/oof.csv"
echo "  - Inference: reports/preds.csv"
echo "  - Metrics: reports/metrics_global.json"
echo "  - Optimized metrics: reports/metrics_global_opt.json"
echo ""
echo "View metrics:"
cat reports/metrics_global_opt.json
