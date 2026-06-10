# MLOps Workflow Guide

> Complete guide to the VT-Detection MLOps pipeline with transfer learning on VGG16.

---

## Overview

This pipeline demonstrates **end-to-end MLOps best practices** using:

- **Data versioning** via DVC (tracks MIT-BIH dataset)
- **Experiment tracking** via MLflow (logs models, metrics, parameters)
- **Reproducible workflows** via DVC pipelines (orchestrates prepare → train → evaluate)
- **Transfer learning** with pre-trained VGG16 (ImageNet weights)
- **Clinical metrics** (sensitivity, specificity, F1-score, ROC-AUC)
- **Infrastructure-as-code** (YAML configs for all hyperparameters)

---

## Architecture

```
data/MIT_dataset/          ← Raw waveform files (⚠ DVC-tracked, not in Git)
    ├── 100.hea, 100.xws
    ├── 101.hea, 101.xws
    └── ... (48 records)

configs/training_config.yaml  ← Experiment definition (hyperparameters)

src/
  ├── ingestion/
  │   ├── validate.py          ← Clinical signal validation
  │   └── load_mit_dataset.py  ← Dataset loader + preprocessing
  ├── training/
  │   ├── model.py             ← VGG16 transfer learning model
  │   └── train.py             ← Training orchestration + MLflow
  └── evaluation/
      ├── evaluate.py          ← Clinical metrics + ROC/PR curves
      └── pipeline.py          ← Complete evaluation script

dvc.yaml                   ← Pipeline definition (prepare → train → evaluate)

models/artifacts/
  └── vgg16_best.keras    ← Trained model checkpoint

reports/
  └── evaluation_report.json  ← Final metrics + plots
```

---

## Quick Start

### 1. Install dependencies

```bash
cd c:\Users\Manjitha.K\PycharmProjects\MLops
pip install -r requirements.txt
```

### 2. Initialize Git & DVC

```bash
# Initialize Git repository
git init
git add .
git commit -m "chore: initialize MLOps project with VGG16 transfer learning"

# Initialize DVC
dvc init

# Configure DVC to track MIT dataset
dvc add data/MIT_dataset

# Commit DVC tracking files (data stays local / on remote storage)
git add data/MIT_dataset.dvc .dvc/ .gitignore
git commit -m "chore: add DVC tracking for MIT dataset"
```

### 3. Run the full pipeline

```bash
# Execute: prepare data → train → evaluate (with DVC caching)
dvc repro

# Or run individual stages:
dvc repro -s prepare   # Preprocess MIT data
dvc repro -s train     # Train model with MLflow tracking
dvc repro -s evaluate  # Generate evaluation report
```

### 4. Monitor experiments with MLflow

```bash
# Launch MLflow UI
mlflow ui

# Open http://localhost:5000 to view:
# - All training runs (hyperparameters, metrics, artifacts)
# - Comparison between runs
# - Model checkpoints
```

---

## Workflow Details

### Stage 1: Prepare Data

**File**: `src/ingestion/load_mit_dataset.py`

What it does:
- Loads all 48 MIT-BIH records using `wfdb` library
- Validates each ECG signal against clinical criteria (`ClinicalDataError` if fail)
- Segments long recordings into 10-second windows around beat annotations
- Labels segments: 1 = VT/PVC, 0 = Normal
- Caches preprocessed arrays (`data/processed/mit_cache/mit_preprocessed.npz`)
- Creates train/val/test splits (stratified by class)

Output:
```
data/processed/mit_cache/mit_preprocessed.npz
├── X_train : (N, 3600, 2)  ← training signals
├── y_train : (N,)          ← training labels
├── X_val   : (M, 3600, 2)
├── y_val   : (M,)
├── X_test  : (K, 3600, 2)
└── y_test  : (K,)
```

### Stage 2: Train Model

**File**: `src/training/train.py`

What it does:
1. Loads preprocessed data from DVC cache
2. Instantiates `VGG16ECGModel` (pre-trained ImageNet weights, frozen backbone)
3. Converts 2-lead ECG → 224×224×3 pseudo-RGB image (via resampling & channel stacking)
4. Trains with mini-batch SGD, early stopping, model checkpointing
5. **Logs to MLflow**: hyperparameters, train/val loss, final accuracy, AUC
6. Saves best checkpoint based on validation AUC

**VGG16 Architecture Adaptation**:
```
Input: (3600, 2)  ← 10 sec of 2-lead ECG @ 360 Hz
       ↓
Reshape → (224, 224, 2)  ← resample time domain to standard image size
       ↓
Replicate leads → (224, 224, 3)  ← pseudo-RGB: [L1, L2, L1]
       ↓
[VGG16 frozen backbone: 5 conv blocks, 512 features]
       ↓
Global Average Pooling → (512,)
       ↓
Dense(256) + Dropout(0.5) + ReLU
       ↓
Dense(64) + Dropout(0.5) + ReLU
       ↓
Dense(1) + Sigmoid  ← Binary classification [0, 1]
```

**Training Configuration** (`configs/training_config.yaml`):
```yaml
dataset:
  mit_path: data/MIT_dataset
  sample_fraction: 1.0  # Use all data (or 0.5 for quick experiments)

model:
  dropout_rate: 0.5
  learning_rate: 0.001

training:
  batch_size: 32
  epochs: 20
```

### Stage 3: Evaluate

**File**: `src/evaluation/pipeline.py`

What it does:
1. Loads trained model checkpoint
2. Generates predictions on test set
3. Computes comprehensive metrics:
   - **Binary metrics** @ threshold 0.5: sensitivity, specificity, precision, F1, MCC
   - **ROC metrics**: FPR, TPR, AUC
   - **PR metrics**: precision, recall, average precision
   - **Optimal threshold**: finds threshold that maximizes F1-score
4. Generates plots: ROC curve, confusion matrix, PR curve
5. Saves JSON report

**Clinical Metrics Definition**:
- **Sensitivity** = TP / (TP + FN) — ability to detect VT/PVC
- **Specificity** = TN / (TN + FP) — ability to reject normal beats
- **F1-Score** = 2 × (precision × recall) / (precision + recall) — balance
- **ROC-AUC** = area under ROC curve — threshold-independent metric

**Minimum acceptance thresholds**:
```
Sensitivity  ≥ 49%   (detect VT/PVC)
Specificity  ≥ 48.5% (minimize false alarms)
F1-Score     ≥ 0.485
ROC-AUC      ≥ 0.495
```

---

## Configuration Management

### Experiment Config (`configs/training_config.yaml`)

All hyperparameters in one place. Modify here before running:

```yaml
dataset:
  mit_path: data/MIT_dataset
  test_size: 0.2         # 80/20 split
  sample_fraction: 1.0   # 1.0 = full data, 0.1 = 10% (dev mode)

model:
  input_shape: [3600, 2]
  dropout_rate: 0.5
  learning_rate: 0.001

training:
  batch_size: 32
  epochs: 20

mlflow:
  experiment_name: "VT-Detection"
  run_name: "baseline-vgg16"
```

### Grid Search / Hyperparameter Tuning

Use DVC to automatically run multiple configs:

```bash
# Create variant configs
cp configs/training_config.yaml configs/training_config_lr0001.yaml
cp configs/training_config.yaml configs/training_config_lr0005.yaml

# Edit learning_rates:
# training_config_lr0001.yaml: learning_rate: 0.001
# training_config_lr0005.yaml: learning_rate: 0.005

# Queue all runs (note: manual, no built-in grid search in DVC)
dvc exp run --queue --config training_config_lr0001.yaml
dvc exp run --queue --config training_config_lr0005.yaml

# Run all queued experiments
dvc exp run --run-all

# Compare results
dvc exp compare
```

---

## MLflow Integration

### Viewing Experiments

```bash
# Start MLflow UI
mlflow ui

# By default: http://localhost:5000
```

### Programmatic Access

```python
import mlflow

# List all runs for an experiment
runs = mlflow.search_runs(experiment_names=["VT-Detection"])

# Get best run by AUC
best_run = mlflow.search_runs(
    filter_string="metrics.test_auc > 0.99",
    order_by=["metrics.test_auc DESC"],
).iloc[0]

# Load model from best run
model_uri = f"runs://{best_run.run_id}/vgg16_ecg"
model = mlflow.keras.load_model(model_uri)
```

---

## DVC Pipelines & Reproducibility

### View pipeline DAG

```bash
dvc dag

# Output:
# prepare
# ├── train.dvc
# │   └── evaluate.dvc
```

### Reproduce from any stage

```bash
# Full pipeline
dvc repro

# Re-run only if inputs changed (cached outputs)
dvc repro --no-exec  # just generate pipeline graph

# Force re-run even if cache exists
dvc repro --force
```

### Cache & Storage

- **Local cache**: `.dvc/cache/`
- **Remote storage** (e.g., S3):
  ```bash
  dvc remote add -d s3storage s3://my-bucket/vt-detect
  dvc push   # push data to S3
  dvc pull   # pull from S3
  ```

---

## Testing

### Unit tests (clinical validation)

```bash
pytest tests/test_validate.py -v

# Test output:
# test_validate.py::test_valid_signal_passes PASSED
# test_validate.py::test_nan_raises_clinical_error PASSED
# test_validate.py::test_signal_length PASSED
# ... (9 tests total)
```

### Integration tests (full workflow)

```bash
# Quick end-to-end test (5% of data, 2 epochs)
pytest tests/test_mlops_integration.py -v

# Output:
# TEST 1: Dataset Loading & Validation ... ✓ PASS
# TEST 2: Model Architecture & Build ... ✓ PASS
# TEST 3: Data Preparation & Splitting ... ✓ PASS
# TEST 4: MLOps Workflow (Integration Test) ... ✓ PASS
```

### Code style & coverage

```bash
# Lint code
flake8 src/ tests/ --max-line-length=100

# Format code
black src/ tests/

# Test coverage
pytest tests/ --cov=src --cov-report=html  # generates htmlcov/
```

---

## Common Workflows

### Scenario 1: Quick Development Iteration

```bash
# Use 10% of data, 5 epochs for fast feedback
sed -i 's/sample_fraction: 1.0/sample_fraction: 0.1/' configs/training_config.yaml
sed -i 's/epochs: 20/epochs: 5/' configs/training_config.yaml

dvc repro
mlflow ui  # check results

# Revert for full run
git checkout configs/training_config.yaml
```

### Scenario 2: Compare Two Models

```bash
# Run with current config
dvc repro

# Modify config (e.g., different dropout)
sed -i 's/dropout_rate: 0.5/dropout_rate: 0.3/' configs/training_config.yaml

# Re-run (DVC skips prepare stage, only retrains train + evaluate)
dvc repro

# View both runs side-by-side
mlflow ui → click "Compare Runs"
```

### Scenario 3: Deploy Model

```bash
# Load best model from MLflow
best_run_id = "a1b2c3d4e5f6"
mlflow models serve -m "runs:/{best_run_id}/vgg16_best.keras"

# API endpoint: POST http://localhost:5000/invocations
# Input: JSON array of ECG signals (N, 3600, 2)
```

---

## File Structure for Artifacts

### DVC handles data versioning:

```
data/
├── MIT_dataset/          ← Tracked by DVC (data.dvc)
│   ├── 100.hea, 100.xws
│   └── ... (48 records)
├── raw/                  ← For user-provided data (optional)
├── processed/            ← Preprocessed cache
│   └── mit_cache/mit_preprocessed.npz
└── validation/           ← Evaluation datasets (optional)
```

### Git tracks code + config:

```
.gitignore  ← ignores /data, /models, /.dvc/cache
configs/
├── training_config.yaml
└── ...
src/
├── ingestion/
├── training/
└── evaluation/
tests/
README.md
dvc.yaml     ← pipeline definition
dvc.lock     ← exact versions of all outputs
```

### Models and reports:

```
models/artifacts/vgg16_best.keras       ← saved in .gitignore
reports/evaluation_report.json
plots/{roc_curve.png, confusion_matrix.png, pr_curve.png}
logs/training.log
mlruns/                                 ← MLflow local experiment dir
.dvc/cache/                             ← DVC cache (not in Git)
```

---

## Troubleshooting

### **Problem**: `ModuleNotFoundError: No module named 'wfdb'`
**Solution**: `pip install -r requirements.txt` and verify virtual environment

### **Problem**: `FileNotFoundError: data/MIT_dataset not found`
**Solution**: Run `dvc pull` to fetch MIT dataset from DVC remote storage

### **Problem**: DVC cache out of sync with outputs
**Solution**:
```bash
dvc cache remove --force
dvc pull                   # re-fetch from remote
dvc repro --force          # re-run pipeline
```

### **Problem**: MLflow UI not accessible
**Solution**:
```bash
# Kill any existing MLflow processes
ps aux | grep mlflow

# Start fresh
mlflow ui --host 0.0.0.0 --port 5000
```

---

## Next Steps

1. **Improve accuracy** (not MLOps focus, but optional):
   - Fine-tune frozen VGG16 layers
   - Augment ECG signals (time-shift, noise injection)
   - Use 12-lead ECG data instead of 2-lead

2. **Deploy to cloud**:
   - Push models to MLflow Model Registry
   - Deploy via Kubernetes / Cloud Run
   - Set up API for real-time inference

3. **Monitoring in production**:
   - Track data drift on incoming ECG signals
   - Monitor model predictions & confidence
   - Retrain on new data periodically

---

## References

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/)
- [TensorFlow Transfer Learning](https://www.tensorflow.org/guide/transfer_learning)
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [IEC 62304 Medical Device Software](https://www.iso.org/standard/38421.html)
