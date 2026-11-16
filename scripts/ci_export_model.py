"""Export the model version this run registered, with its reference profile.

Writes model-bundle/ for the drift workflow (.github/workflows/drift.yml):

    model-bundle/model/          the registered pyfunc (servable with `mlflow models serve`)
    model-bundle/reference.csv   the reference profile built from that model
    model-bundle/bundle.json     which version, training run and commit it came from

The registry only lives for the CI job, so this is how a later job gets the model.
"""
import json
import os
import shutil
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

CONFIG = "configs/training_config.yaml"
REFERENCE = Path("data/monitoring/reference.csv")
BUNDLE = Path("model-bundle")
ALIAS = "challenger"  # set on every version train.py registers

with open(CONFIG, encoding="utf-8") as f:
    mlflow_config = yaml.safe_load(f).get("mlflow", {})
mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "sqlite:///mlflow_tracking.db"))
name = mlflow_config.get("registered_model_name", "vt-detector")

version = MlflowClient().get_model_version_by_alias(name, ALIAS)
shutil.rmtree(BUNDLE, ignore_errors=True)
mlflow.artifacts.download_artifacts(f"models:/{name}@{ALIAS}", dst_path=str(BUNDLE / "model"))
shutil.copy(REFERENCE, BUNDLE / "reference.csv")

info = {
    "registered_model": name,
    "version": version.version,
    "training_run_id": version.run_id,
    "git_commit": os.environ.get("GITHUB_SHA", version.tags.get("git_commit")),
    "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
}
(BUNDLE / "bundle.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(f"Exported {name} version {version.version} (run {version.run_id}) to {BUNDLE}/")
