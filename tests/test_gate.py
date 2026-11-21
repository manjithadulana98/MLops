"""Tests for the model promotion gate (src/evaluation/gate.py)."""

from __future__ import annotations

import json

import mlflow
import pytest
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.evaluation import gate
from src.registry import CHALLENGER_ALIAS, CHAMPION_ALIAS

MODEL = "vt-detector-test"
THRESHOLDS = {"sensitivity": 0.8, "roc_auc": 0.9}


def make_report(sensitivity: float = 0.85, roc_auc: float = 0.95) -> dict:
    return {
        "binary_metrics": {
            "sensitivity": sensitivity,
            "specificity": 0.9,
            "balanced_accuracy": (sensitivity + 0.9) / 2,
            "precision": 0.5,
            "f1_score": 0.6,
        },
        "roc_metrics": {"roc_auc": roc_auc},
        "pr_metrics": {"average_precision": 0.7},
    }


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    """Restore the global MLflow URI and keep test output out of a real CI job summary."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    # Otherwise it would override the test configs' throwaway SQLite registries.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def registry(tmp_path):
    """Isolated SQLite registry with one model version aliased as challenger."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    exp_id = client.create_experiment(
        "gate-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    run = client.create_run(exp_id)
    client.create_registered_model(MODEL)
    mv = client.create_model_version(
        MODEL, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
    )
    client.set_registered_model_alias(MODEL, CHALLENGER_ALIAS, mv.version)
    return {"uri": uri, "client": client, "run_id": run.info.run_id, "version": str(mv.version)}


def run_gate(tmp_path, tracking_uri: str, report: dict, *extra_args: str):
    config = {
        "gate": {"thresholds": THRESHOLDS},
        "mlflow": {"tracking_uri": tracking_uri, "registered_model_name": MODEL},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "gate.json"

    code = gate.main(
        ["--config", str(config_path), "--report", str(report_path), "--output", str(output),
         *extra_args]
    )
    return code, json.loads(output.read_text(encoding="utf-8"))


# ── Threshold checks ─────────────────────────────────────────────────────────


def test_check_thresholds_reports_each_metric():
    checks = {c["metric"]: c for c in gate.check_thresholds(make_report(0.7), THRESHOLDS)}
    assert checks["sensitivity"] == {
        "metric": "sensitivity", "value": 0.7, "minimum": 0.8, "passed": False
    }
    assert checks["roc_auc"]["passed"]


def test_threshold_is_inclusive():
    assert gate.check_thresholds(make_report(0.8), {"sensitivity": 0.8})[0]["passed"]


def test_balanced_accuracy_can_be_gated():
    # (0.3 + 0.9) / 2 = 0.6
    [check] = gate.check_thresholds(make_report(sensitivity=0.3), {"balanced_accuracy": 0.55})
    assert check["passed"]
    assert check["value"] == pytest.approx(0.6)
    [check] = gate.check_thresholds(make_report(sensitivity=0.1), {"balanced_accuracy": 0.55})
    assert not check["passed"]


def test_unknown_metric_or_no_thresholds_is_an_error():
    with pytest.raises(ValueError, match="Unknown gate metric"):
        gate.check_thresholds(make_report(), {"accuracy": 0.5})
    with pytest.raises(ValueError, match="No gate thresholds"):
        gate.check_thresholds(make_report(), {})


# ── Promotion ────────────────────────────────────────────────────────────────


def test_passing_gate_promotes_challenger_to_champion(tmp_path, registry):
    code, result = run_gate(tmp_path, registry["uri"], make_report())
    client = registry["client"]

    assert code == 0 and result["passed"]
    assert result["model"] == {"name": MODEL, "version": registry["version"], "promoted": True}
    champion = client.get_model_version_by_alias(MODEL, CHAMPION_ALIAS)
    assert str(champion.version) == registry["version"]
    assert champion.tags["gate"] == "passed"
    metrics = client.get_run(registry["run_id"]).data.metrics
    assert metrics["eval_roc_auc"] == pytest.approx(0.95)


def test_failing_gate_exits_1_and_leaves_champion_unchanged(tmp_path, registry):
    code, result = run_gate(tmp_path, registry["uri"], make_report(roc_auc=0.6))
    client = registry["client"]

    assert code == 1 and not result["passed"]
    assert result["model"]["promoted"] is False
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(MODEL, CHAMPION_ALIAS)
    assert client.get_model_version(MODEL, registry["version"]).tags["gate"] == "failed"


def test_no_promote_records_result_without_moving_champion(tmp_path, registry):
    code, result = run_gate(tmp_path, registry["uri"], make_report(), "--no-promote")

    assert code == 0 and result["model"]["promoted"] is False
    with pytest.raises(MlflowException):
        registry["client"].get_model_version_by_alias(MODEL, CHAMPION_ALIAS)


def test_missing_challenger_still_checks_thresholds(tmp_path):
    empty_registry = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    code, result = run_gate(tmp_path, empty_registry, make_report())

    assert code == 0 and result["passed"]
    assert result["model"] == {"name": MODEL, "version": None, "promoted": False}


def test_writes_github_step_summary(tmp_path, registry, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    run_gate(tmp_path, registry["uri"], make_report(roc_auc=0.6))

    text = summary.read_text(encoding="utf-8")
    assert "Promotion gate: FAILED" in text
    assert "| roc_auc | 0.6000 | 0.9000 | **FAIL** |" in text
