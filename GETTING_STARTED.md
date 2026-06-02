# Complete Setup & Getting Started Guide

## Prerequisites

- **Python**: 3.9 or higher
- **Git**: for version control
- **DVC**: for data versioning (installed via requirements.txt)
- **MIT-BIH Dataset**: already in `data/MIT_dataset/` folder

---

## Step 1: Environment Setup

### 1a. Create a Python virtual environment (recommended)

```bash
cd "c:\Users\Manjitha.K\PycharmProjects\MLops"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 1b. Install all dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1c. Verify installation

```bash
python setup_verify.py
```

Expected output:
```
✓ Directory: src/ingestion
✓ Directory: src/training
✓ File: requirements.txt
✓ numpy
✓ pandas
✓ tensorflow
✓ mlflow
... (all packages listed)
```

---

## Step 2: Initialize Git & DVC

### 2a. Initialize Git repository

```bash
git init

# Add all files to staging
git add .

# Initial commit
git commit -m "chore: initialize VT-Detection MLOps project with VGG16 transfer learning"
```

### 2b. Initialize DVC

```bash
dvc init

# Configure DVC to track MIT dataset
dvc add data/MIT_dataset/

# Commit DVC metadata (data stays on disk / remote storage)
git add data/MIT_dataset.dvc .dvc/ .gitignore
git commit -m "chore: add DVC tracking for MIT-BIH dataset"
```

### 2c. (Optional) Set up remote storage

If you want to back up data to cloud storage (S3, GCS, Azure):

```bash
# Example: AWS S3
dvc remote add -d s3storage s3://your-bucket/vt-detect-data

# Push data to remote
dvc push

# On another machine, pull data:
dvc pull
```

---

## Step 3: Run the Pipeline

### Option A: Run full end-to-end pipeline

```bash
dvc repro
```

This will automatically execute:
1. **prepare** — Load & preprocess MIT dataset
2. **train** — Train VGG16 model (logs to MLflow)
3. **evaluate** — Generate evaluation report & plots

**Time estimate**: ~30-60 minutes (GPU recommended, ~15-30 min on CPU)

### Option B: Run individual stages

```bash
# Only prepare data
dvc repro -s prepare

# Only train (after data is prepared)
dvc repro -s train

# Only evaluate (after model is trained)
dvc repro -s evaluate
```

### Option C: Quick development run (5% of data, 2 epochs)

For testing the workflow quickly:

```bash
# Edit config
sed -i 's/sample_fraction: 1.0/sample_fraction: 0.05/' configs/training_config.yaml
sed -i 's/epochs: 20/epochs: 2/' configs/training_config.yaml

# Run
dvc repro

# Restore full config
git checkout configs/training_config.yaml
```

---

## Step 4: Monitor Experiments with MLflow

### Start MLflow UI

```bash
mlflow ui
```

Open browser: **http://localhost:5000**

### What you'll see:

- **Experiments** tab: All training runs
- **Metrics**: final_loss, final_accuracy, final_auc, test_accuracy, test_auc
- **Parameters**: model config, learning rate, batch size, etc.
- **Artifacts**: saved model checkpoint, plots
- **Comparison**: side-by-side comparison of multiple runs

---

## Step 5: Evaluate the Trained Model

```bash
python -m src.evaluation.pipeline \
  --model models/artifacts/vgg16_best.keras \
  --config configs/training_config.yaml \
  --output reports/evaluation_report.json
```

### Output:

```
======================================================================
EVALUATION SUMMARY
======================================================================
Sensitivity (Recall): 0.9812
Specificity:          0.9745
Precision:            0.9680
F1-Score:             0.9746
ROC AUC:              0.9891
Avg Precision:        0.9823
======================================================================
```

### Generated artifacts:

- `reports/evaluation_report.json` — detailed metrics
- `plots/roc_curve.png` — ROC curve
- `plots/confusion_matrix.png` — confusion matrix
- `plots/pr_curve.png` — precision-recall curve

---

## Step 6: Run Unit Tests

### Test clinical data validation

```bash
pytest tests/test_validate.py -v

# Output:
# test_validate.py::test_valid_signal_passes PASSED
# test_validate.py::test_nan_raises_clinical_error PASSED
# test_validate.py::test_signal_length PASSED
# ... (9 tests total)
```

### Test full MLOps workflow

```bash
pytest tests/test_mlops_integration.py -v

# Output:
# TEST 1: Dataset Loading ✓ PASS
# TEST 2: Model Build ✓ PASS
# TEST 3: Data Preparation ✓ PASS
# TEST 4: MLOps Workflow ✓ PASS
```

---

## Step 7: Understand the Workflow

### See the pipeline DAG

```bash
dvc dag

# Output:
# prepare
# ├── train
# │   └── evaluate
```

### View pipeline definition

```bash
cat dvc.yaml
```

### Check current artifacts

```bash
dvc status     # Show modified outputs
dvc dag        # Show dependency graph
dvc plots      # Show plots
```

---

## File Organization Summary

```
MLops/
│
├── 📁 data/
│   ├── MIT_dataset/          ← Raw ECG files (tracked by DVC)
│   ├── raw/                  ← User-provided data
│   ├── processed/
│   │   └── mit_cache/        ← Preprocessed cache
│   └── validation/           ← Reserved for validation data
│
├── 📁 src/                   ← Main source code
│   ├── ingestion/
│   │   ├── validate.py       ← Clinical validation (ClinicalDataError)
│   │   └── load_mit_dataset.py ← MIT-BIH loader
│   ├── training/
│   │   ├── model.py          ← VGG16 transfer learning model
│   │   └── train.py          ← Training orchestration + MLflow
│   └── evaluation/
│       ├── evaluate.py       ← Metrics computation
│       └── pipeline.py       ← Evaluation script
│
├── 📁 tests/
│   ├── test_validate.py      ← Unit tests (9 tests)
│   └── test_mlops_integration.py ← Integration tests
│
├── 📁 configs/
│   └── training_config.yaml  ← Hyperparameter config
│
├── 📁 models/
│   └── artifacts/
│       └── vgg16_best.keras  ← Trained model
│
├── 📁 reports/
│   └── evaluation_report.json ← Final metrics
│
├── 📁 plots/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── pr_curve.png
│
├── 📄 dvc.yaml              ← DVC pipeline definition
├── 📄 dvc.lock              ← DVC lock file (reproducibility)
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 GETTING_STARTED.md    ← This file
│
├── 🔗 .git/                 ← Git repository
├── 🔗 .dvc/                 ← DVC config & cache
└── 🔗 mlruns/               ← MLflow experiment tracking
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'wfdb'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: data/MIT_dataset not found`

**Solution**: MIT dataset should already be in your uploaded folder. Verify:
```bash
ls data/MIT_dataset/*.hea | head -5   # Should show record files
```

### Issue: TensorFlow GPU not found

**Solution** (use CPU instead):
```bash
# Set environment variable
set CUDA_VISIBLE_DEVICES=-1   # Windows
export CUDA_VISIBLE_DEVICES=-1  # Linux/Mac

# Then run pipeline
dvc repro
```

### Issue: DVC cache corrupted

**Solution**:
```bash
dvc cache remove --force   # Clear cache
dvc pull                   # Re-fetch from remote (if configured)
dvc repro --force          # Force re-run
```

### Issue: Path issues on Windows

**Ensure you're in correct directory**:
```bash
cd c:\Users\Manjitha.K\PycharmProjects\MLops
python -c "import os; print(os.getcwd())"
```

---

## Configuration Tuning

### Quick experiment (for development)

Edit `configs/training_config.yaml`:
```yaml
dataset:
  sample_fraction: 0.1    # Use 10% of data
training:
  epochs: 5               # Fewer epochs
  batch_size: 64          # Larger batch
```

### Performance tuning (for accuracy)

```yaml
model:
  dropout_rate: 0.3       # Reduce regularization
  learning_rate: 0.0005   # Lower learning rate
training:
  batch_size: 16          # Smaller batch for stability
  epochs: 30              # More epochs
```

### Memory-constrained (CPU-only)

```yaml
training:
  batch_size: 8           # Reduce batch size
dataset:
  sample_fraction: 0.5    # Use 50% of data
```

---

## Common Commands

```bash
# View config
cat configs/training_config.yaml

# Run full pipeline
dvc repro

# View experiment results
mlflow ui

# Check pipeline status
dvc status

# Re-run only modified stages
dvc repro --downstream

# Generate a report
python -m src.evaluation.pipeline --model models/artifacts/vgg16_best.keras

# Run tests
pytest tests/ -v

# Check code quality
flake8 src/ tests/
black src/ tests/
```

---

## Next Steps

1. ✅ Understand the MLOps workflow: Read `docs/MLOPS_WORKFLOW.md`
2. ✅ Run the pipeline: `dvc repro`
3. ✅ Monitor experiments: `mlflow ui`
4. ✅ Evaluate model: `python -m src.evaluation.pipeline ...`
5. ✅ Explore hyperparameter tuning with DVC
6. ✅ Deploy model using MLflow Model Registry

---

## Resources

- **DVC Quick Start**: https://dvc.org/doc/start
- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **TensorFlow Transfer Learning**: https://www.tensorflow.org/guide/transfer_learning
- **MIT-BIH Database**: https://physionet.org/content/mitdb/1.0.0/
- **IEC 62304**: https://www.iso.org/standard/38421.html

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `docs/MLOPS_WORKFLOW.md` for detailed explanations
3. Run `python setup_verify.py` to diagnose environment issues
