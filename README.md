# VT-Detect MLOps Pipeline

> **An IEC 62304-compliant, end-to-end MLOps pipeline for automated Ventricular Tachycardia (VT) detection from 12-lead ECG signals.**

---

## Project Goal

The primary objective of this project is to design, train, validate, and deploy a deep-learning model that detects **Ventricular Tachycardia** from ECG waveform data.

### Gold Standard Validation Dataset — MIT-BIH Arrhythmia Database

The **[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)** (PhysioNet) serves as the **Gold Standard** validation set for all model performance evaluations. It consists of:

| Property | Detail |
|---|---|
| Records | 48 half-hour two-lead ECG recordings |
| Subjects | 47 patients (mixed inpatient / outpatient) |
| Sampling rate | 360 Hz |
| Resolution | 11-bit over ±5 mV |
| Annotations | Beat-by-beat labels by two independent cardiologists |

All model checkpoints must achieve the following minimum thresholds on the MIT-BIH hold-out set before being eligible for deployment:

| Metric | Minimum Threshold |
|---|---|
| Sensitivity (Recall) | ≥ 98 % |
| Specificity | ≥ 97 % |
| F1-Score | ≥ 0.97 |
| AUC-ROC | ≥ 0.99 |

These thresholds are derived from **ANSI/AAMI EC57** performance standards for arrhythmia detectors used in medical devices.

---

## Directory Structure

```
MLops/
├── .github/
│   └── workflows/
│       └── ci.yml                  # Automated CI: lint + test
├── configs/                        # YAML experiment configurations
├── data/
│   ├── raw/                        # Immutable source data (DVC-tracked)
│   ├── processed/                  # Feature-engineered datasets
│   └── validation/                 # MIT-BIH Gold Standard validation set
├── docs/
│   └── IEC62304/                   # Software lifecycle documentation
├── models/
│   └── artifacts/                  # Serialized model checkpoints
├── src/
│   ├── ingestion/
│   │   └── validate.py             # Clinical data validation (SU-INGEST-01)
│   ├── training/                   # Model architecture & training loop
│   └── evaluation/                 # Metrics, confusion matrix, ROC curves
├── tests/
│   └── test_validate.py            # Unit tests (TC-001 – TC-009)
├── requirements.txt
└── README.md
```

---

## IEC 62304 Compliance Overview

This project follows **IEC 62304:2006+AMD1:2015** — the international standard for medical device software lifecycle processes.

| IEC 62304 Activity | Implementation |
|---|---|
| Software Development Planning | `docs/IEC62304/` |
| Software Requirements Analysis | Inline `SRS-*` requirement tags in source files |
| Software Architecture Design | `src/` module decomposition |
| Software Unit Implementation | Each `src/**/*.py` module maps to a Software Unit (SU) |
| Software Unit Verification | `tests/` — pytest with ≥ 90 % coverage gate |
| Software Integration | GitHub Actions CI (`ci.yml`) |
| Software System Testing | MIT-BIH Gold Standard validation set |

**Software Safety Classification: Class B** (non-life-sustaining, but potential for serious injury if incorrect VT detection leads to missed treatment).

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd MLops
pip install -r requirements.txt
```

### 2. Run unit tests

```bash
pytest tests/ -v --cov=src
```

### 3. Pull versioned data via DVC

```bash
dvc pull
```

---

## Data Versioning with DVC

All datasets under `data/` are version-controlled with **[DVC](https://dvc.org/)** and excluded from Git. This ensures:

- Reproducibility of every model training run.
- Compliance with audit trail requirements (IEC 62304 §8).
- Safe handling of patient-adjacent data (HIPAA / GDPR considerations).

See [DVC Remote Storage Setup](#dvc-remote-storage-setup) below for configuration details.

---

## DVC Remote Storage Setup

After initialising the repository (see commands below), configure a remote storage backend:

```bash
# Example: local filesystem remote (replace with S3/GCS for production)
dvc remote add -d myremote /path/to/your/dvc-storage
dvc remote modify myremote endpointurl https://s3.amazonaws.com   # S3 example

# Push data to remote
dvc push
```

---

## References

- [MIT-BIH Arrhythmia Database — PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- [ANSI/AAMI EC57 — Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms](https://www.aami.org/)
- [IEC 62304:2006+AMD1:2015 — Medical device software lifecycle processes](https://www.iso.org/standard/38421.html)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

*Maintained by the Biomedical Engineering Research Team.*
