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
- **Balanced accuracy** = (sensitivity + specificity) / 2 — accuracy that isn't
  fooled by class imbalance (a model that never flags VT/PVC scores 0.5)
- **F1-Score** = 2 × (precision × recall) / (precision + recall) — balance
- **ROC-AUC** = area under ROC curve — threshold-independent metric

**Minimum acceptance threshold** (the promotion gate, `gate.thresholds`):
```
Balanced accuracy  ≥ 0.55
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
```

### Model Registry

Each training run logs the trained model and registers it as a new version of
`vt-detector` (configurable via `mlflow.registered_model_name`). Every version is
tagged with where it came from:

| Tag | Meaning |
|---|---|
| `git_commit` | Commit the training code was run from |
| `git_dirty` | `true` if there were uncommitted changes (not reproducible from git alone) |
| `data_md5` | DVC hash of the dataset, from `data/MIT_dataset.dvc` |

New versions get the **`challenger`** alias. The **`champion`** alias is what
downstream consumers should load; moving it is a separate, metrics-gated step,
so training alone never changes the production model.

```python
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow_tracking.db")
client = MlflowClient()

# Which version is the current challenger, and what was it trained from?
mv = client.get_model_version_by_alias("vt-detector", "challenger")
print(mv.version, mv.tags)

# Load a model by alias. Each version is a self-contained pyfunc (network +
# preprocessing + input validation): raw ECG windows in, probabilities out.
model = mlflow.pyfunc.load_model("models:/vt-detector@challenger")
probabilities = model.predict(signals)  # signals: float32 (N, 3600, 2), mV
```

In the MLflow UI, open **Models → vt-detector** to see versions, tags and aliases.
Set `mlflow.register_model: false` to skip logging the model (e.g. on low-disk machines).

### Promotion Gate

The `gate` DVC stage (`src/evaluation/gate.py`) runs after `evaluate`:

1. Compares `reports/evaluation_report.json` against `gate.thresholds` in
   `configs/training_config.yaml` (sensitivity, specificity, precision, f1_score,
   roc_auc, average_precision).
2. Records the result on the `challenger` version: a `gate=passed|failed` tag, plus
   `eval_*` metrics on its training run.
3. If every threshold is met, moves the **`champion`** alias to that version.
   If not, exits with code 1 (failing `dvc repro` and the CI job) and leaves
   `champion` where it was.

The result is written to `reports/gate.json` (a DVC metric, so `dvc metrics show`
includes it). In GitHub Actions the gate table
also appears on the workflow run's summary page. Use `--no-promote` to check
thresholds without moving the alias.

**In CI** (`.github/workflows/train.yml`): push-triggered runs train a smoke-sized
model (1% of data, 1 epoch) that isn't expected to meet the thresholds, so the gate
is *report-only* there (`--no-promote`, `continue-on-error`). Manual runs
(**Actions → Train & Evaluate → Run workflow**, where you choose the data fraction
and epochs) run the real `gate` stage and fail the job on a miss.

### Serving

Each registered version is an MLflow **pyfunc** (`src/serving/pyfunc_model.py`) that
bundles the network, the ECG→image preprocessing, the clinical input validation and
its own copy of `src/`, plus a signature (`float32 (-1, 3600, 2)`) and an input
example. It runs without the repo, so it can be served directly:

```powershell
# From the repo root with the venv activated (.venv\Scripts\activate), so that the
# `uvicorn` that MLflow launches is the venv's.
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_tracking.db"
mlflow models serve -m "models:/vt-detector@champion" -p 5001 --env-manager local

# In a second terminal: score a real 10-second MIT-BIH window
python scripts/predict_request.py --record 119 --start 60
```

- The alias is resolved when the server starts: restart it after a new `champion`
  is promoted. `models:/vt-detector@challenger` or `models:/vt-detector/3` also work.
- `--env-manager local` uses the current environment. Without it, MLflow builds an
  isolated environment from the model's pinned requirements (slower; needs `virtualenv`).

REST API:

| Request | Response |
|---|---|
| `GET /ping` | `200` once the model is loaded |
| `POST /invocations` with `{"inputs": [window, ...]}`, each window 3600 rows of `[lead1_mV, lead2_mV]` | `{"predictions": [p, ...]}`, P(VT/PVC) per window |
| Wrong shape, or a window failing clinical validation (NaN/Inf, outside ±5 mV) | `400 INVALID_PARAMETER_VALUE` with the reason, e.g. `[VOLTAGE_RANGE] ... (lead: window 0, lead 1)` |

### Docker

Build an image from a registered version. The model, its pinned requirements and
MLflow's model server are baked in, so the container needs neither the repo nor the
registry at runtime. Docker Desktop must be running (`docker desktop start`).

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_tracking.db"

# Tag the image with the version it contains (check the alias in the MLflow UI)
mlflow models build-docker -m "models:/vt-detector@champion" -n vt-detector:v3

docker run --rm -p 5002:8080 vt-detector:v3
python scripts/predict_request.py --url http://127.0.0.1:5002/invocations --record 119 --start 60
```

- Same REST API as `mlflow models serve` (`/ping`, `/invocations`), on port 8080
  inside the container.
- The alias is resolved at **build** time: the image holds that exact version.
  Rebuild after a new `champion` is promoted.
- To see what goes into the image without building it:
  `mlflow models generate-dockerfile -m "models:/vt-detector@champion" -d build/docker`.
  It writes a Dockerfile (Python slim base matching the training Python, nginx,
  MLflow, the model under `/opt/ml/model`, then its pinned requirements) and the
  model directory.
- The image is about 6 GB (TensorFlow plus MLflow's full dependency set) and the
  first build takes around 10 minutes.

**From CI: published to GitHub Container Registry.** When a manual Train & Evaluate
run (**Actions → Train & Evaluate → Run workflow**) passes the gate and promotes a new
champion, its `publish-image` job builds that model into an image the same way
(`mlflow models generate-dockerfile` + `docker build`), starts it and sends it the
model's stored input example, and only if that works pushes it:

```powershell
docker pull ghcr.io/manjithadulana98/vt-detector:champion   # latest promoted model
docker run --rm -p 5002:8080 ghcr.io/manjithadulana98/vt-detector:champion
```

- Tags: `champion` moves to each newly promoted model, like the registry alias;
  `run-<n>` stays fixed to the image from Train & Evaluate run *n*. The image labels
  record the commit and MLflow training run.
- Push-triggered runs don't publish: their gate is report-only, so nothing is promoted.
- A 1%, 1-epoch model is close to random and may miss the gate; use a larger
  *sample_fraction* or more *epochs* if the gate keeps failing.
- The first push creates the package as **private**. To pull without logging in, make
  it public under **GitHub profile → Packages → vt-detector → Package settings**;
  otherwise log in first with a personal access token that has `read:packages`
  (`docker login ghcr.io`).
- The workflow pushes with the job's built-in `GITHUB_TOKEN` (`packages: write` on
  that job only), so no secrets need setting up.

> The tracking DB is a local SQLite file, so the registry lives on the machine that
> trained the model. In CI it is discarded after each run; a shared tracking server
> is needed for a registry that persists across machines.

### Monitoring & Drift Detection

Once a model is serving, the question becomes: does the traffic it receives still
look like the data it was trained and tested on? Live traffic has no labels, so
accuracy can't be measured directly. Instead the inputs and the model's outputs are
compared with a **reference profile** built from the held-out test split, using
[Evidently](https://docs.evidentlyai.com/).

```
served model ──(VT_PREDICTION_LOG_DIR)──► monitoring/predictions/*.jsonl ─┐
                                                                          ├─► src.monitoring.drift ─► reports/drift/ + MLflow
dvc stage `reference` ──► data/monitoring/reference.csv ──────────────────┘
```

**What is compared.** Each 10-second window is reduced to a few features
(`src/monitoring/features.py`), computed the same way on both sides:

| Feature (per lead) | What moves it |
|---|---|
| `mean_mv` | baseline offset, electrode/DC problems |
| `std_mv`, `ptp_mv` | amplifier gain, electrode placement |
| `noise_mv` (std of sample-to-sample change) | muscle noise, mains hum |
| `heart_rate_bpm` (lead 1) | a different patient population; also noise, which breaks beat detection |
| `prediction` | the model's output: prediction drift |

Each column gets a two-sample Kolmogorov–Smirnov test (drifted if p < 0.05); the
traffic counts as drifted when at least 30% of the columns (3 of 10) drift
(`--drift-share`). Evidently's default is 50%, but about 0.5 columns drift by chance
at p < 0.05, and a single-cause problem only moves the columns it touches: simulated
electrode noise moved 4 columns, which a 50% rule misses.

With the 1% smoke-test data the reference is only ~120 windows, so it is itself a
noisy sample (one column can look drifted on clean traffic); a full-data reference
(~11k windows) makes the check much steadier.

**1. Build the reference profile** (a DVC stage, after `train`; also run in CI):

```powershell
dvc repro -s reference     # -> data/monitoring/reference.csv
```

**2. Serve with prediction logging on.** The served model appends one JSON line per
scored window (features + prediction, no raw signal) and one per rejected request,
stamped with the training run that produced the model:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_tracking.db"
$env:VT_PREDICTION_LOG_DIR = "monitoring/predictions"
mlflow models serve -m "models:/vt-detector@champion" -p 5001 --env-manager local

# Docker: pass the variable and mount a directory for the logs
docker run --rm -p 5002:8080 -e VT_PREDICTION_LOG_DIR=/logs -v ${PWD}/monitoring/predictions:/logs vt-detector:v4
```

Only versions registered after this feature was added log anything (the logging code
is bundled with each version). Wrong-shape requests are refused by MLflow's signature
check before the model runs, so they appear in the server log, not the prediction log.

**3. Generate traffic.** `scripts/simulate_traffic.py` sends beat-centred MIT-BIH
windows, optionally with a simulated problem (`--drift gain|noise|baseline`):

```powershell
python scripts/simulate_traffic.py --windows 150                 # looks like the reference
python scripts/simulate_traffic.py --windows 150 --drift noise   # noisy electrodes
```

**4. Check for drift:**

```powershell
python -m src.monitoring.drift --log-dir monitoring/predictions --since 30m
```

- Writes `reports/drift/drift_report.html` (Evidently's interactive report: per-column
  distributions and test results) and `reports/drift/drift.json`.
- Records a run in the MLflow experiment **vt-detector-monitoring** (drift scores,
  rejection rate, the report as an artifact), so drift checks can be compared over time
  in the MLflow UI.
- `--since` takes an ISO time or `30m` / `2h` / `1d`. Fewer than 30 scored windows
  (`--min-windows`) is reported as "not enough traffic" instead of tested.
- Requests the model refused (clinical input validation) never reach the drift test,
  so their share is checked separately: more than 5% rejected windows
  (`--max-rejection-rate`) raises a rejection-rate alert. A gain problem big enough
  to push signals out of the valid voltage range shows up here, not as drift.
- Exits 0; add `--fail-on-drift` to exit 1 on drift or a rejection-rate alert, e.g.
  in a scheduled job that should alert. Add `--fail-on-insufficient-data` where
  traffic is expected: then "not enough traffic" also exits 1, since an empty log
  means serving or logging is broken.

**5. In CI** (`.github/workflows/drift.yml`). The same check runs on GitHub Actions:

```
Train & Evaluate ──► artifact "model-bundle" ──► Drift Check workflow
  (registered model + reference.csv            serve it, send MIT-BIH windows,
   + bundle.json: version, run, commit)        drift check --fail-on-drift
```

- **When:** after every successful Train & Evaluate run on `main` (checks the model
  that run just trained), every Monday at 06:00 UTC, and on demand from the
  Actions tab (**Drift Check → Run workflow**). Scheduled and manual runs check the
  model from the latest successful Train & Evaluate run.
- **Alarm:** drift or a rejection-rate alert fails the run, and GitHub sends a
  notification email for the failed run. The run summary shows the model, the
  traffic and the per-column table; the Evidently HTML report, the prediction log
  and the server log are uploaded as the `drift-report` artifact.
- **Seeing it fire:** run it manually with *Simulated problem* = `noise`, `gain` or
  `baseline` (and optionally a *strength*). `gain` with strength `2.5` pushes most
  windows out of the valid voltage range, so it trips the rejection-rate alert
  instead.
- **Where the traffic comes from:** there is no long-running deployment, so the
  job starts the server itself and replays MIT-BIH windows through it. In a real
  setup the servers would write their prediction logs to shared storage (a bucket),
  and the scheduled job would download recent logs instead of generating traffic.
- **False alarms:** each run (and each re-run) sends a different random sample.
  Against the small 1% reference, roughly 1 clean run in 20 crosses the 3-column
  threshold by chance. Re-running draws a fresh sample, which is a quick way to
  tell a fluke from real drift. Production setups usually alert only when drift
  shows up in consecutive checks.
- **Limits:** the bundle artifact is kept for 90 days, so a scheduled run more than
  90 days after the last training run fails with a message asking for a new training
  run. GitHub also pauses scheduled workflows in public repositories after 60 days
  without repository activity.

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

2. **Deploy to cloud** (champion images are already published to GHCR, see Docker):
   - Run the `champion` image on Kubernetes / Cloud Run
   - A shared MLflow tracking server, so the registry outlives CI jobs

3. **Monitoring in production** (input/prediction drift and the scheduled CI check
   are covered above):
   - Ship prediction logs from real servers to shared storage for the scheduled check
   - Collect labels for a sample of live windows to track real accuracy
   - Retrain on new data when drift persists

---

## References

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/)
- [TensorFlow Transfer Learning](https://www.tensorflow.org/guide/transfer_learning)
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [IEC 62304 Medical Device Software](https://www.iso.org/standard/38421.html)
