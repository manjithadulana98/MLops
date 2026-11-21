"""
Monitoring tests: features, prediction log and drift check
==========================================================

Uses synthetic ECG-like windows (spikes at a known heart rate), so these run
without the dataset or a trained model.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from src.monitoring import drift
from src.monitoring.features import FEATURE_COLUMNS, compute_features, estimate_heart_rate
from src.monitoring.prediction_log import (
    LOG_DIR_ENV,
    PredictionLogger,
    read_prediction_logs,
)

FS = 360


def synthetic_windows(n: int, seed: int, gain: float = 1.0, noise: float = 0.02,
                      bpm_range=(60, 100)) -> np.ndarray:
    """(n, 3600, 2) windows: narrow Gaussian 'QRS' spikes at a random rate, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(3600) / FS
    windows = np.empty((n, 3600, 2), dtype=np.float32)
    for i in range(n):
        period = 60.0 / rng.uniform(*bpm_range)
        beats = np.arange(rng.uniform(0, period), 10, period)
        spikes = np.exp(-((t[:, None] - beats[None, :]) ** 2) / (2 * 0.012 ** 2)).sum(axis=1)
        for lead, amplitude in enumerate((1.2, 0.6)):
            windows[i, :, lead] = gain * amplitude * spikes + rng.normal(0, noise, 3600)
    return windows


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv(LOG_DIR_ENV, raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)  # keep test runs out of a real server
    previous_uri = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous_uri)


# ── Features ─────────────────────────────────────────────────────────────────


def test_features_have_one_value_per_window_in_column_order():
    features = compute_features(synthetic_windows(5, seed=0))
    assert list(features) == FEATURE_COLUMNS
    assert all(values.shape == (5,) for values in features.values())


def test_heart_rate_matches_spike_rate():
    t = np.arange(3600) / FS
    beats = np.arange(0.3, 10, 0.8)  # one beat every 0.8 s = 75 bpm
    lead = np.exp(-((t[:, None] - beats[None, :]) ** 2) / (2 * 0.012 ** 2)).sum(axis=1)
    assert estimate_heart_rate(lead) == pytest.approx(75, abs=1)
    assert estimate_heart_rate(np.zeros(3600)) == 0.0


def test_amplitude_and_noise_features_respond_to_distortion():
    clean = compute_features(synthetic_windows(20, seed=1))
    louder = compute_features(synthetic_windows(20, seed=1, gain=2.0))
    noisy = compute_features(synthetic_windows(20, seed=1, noise=0.2))
    assert np.median(louder["lead1_ptp_mv"] / clean["lead1_ptp_mv"]) == pytest.approx(2, rel=0.1)
    assert np.all(noisy["lead1_noise_mv"] > 3 * clean["lead1_noise_mv"])
    np.testing.assert_allclose(louder["heart_rate_bpm"], clean["heart_rate_bpm"], atol=1)


# ── Prediction log ───────────────────────────────────────────────────────────


def test_prediction_logger_is_off_unless_configured(monkeypatch, tmp_path):
    assert PredictionLogger.from_env() is None
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
    assert PredictionLogger.from_env("run-1").log_dir == tmp_path


def test_prediction_log_round_trip(tmp_path):
    log = PredictionLogger(tmp_path / "logs", model_run_id="run-1")
    log.log_scored(synthetic_windows(3, seed=0), np.array([0.1, 0.5, 0.9]))
    log.log_rejected(4, "VOLTAGE_RANGE")

    scored, rejected = read_prediction_logs(tmp_path / "logs")
    assert scored["prediction"].tolist() == [0.1, 0.5, 0.9]
    assert set(FEATURE_COLUMNS) <= set(scored.columns)
    assert set(scored["model_run_id"]) == {"run-1"}
    assert rejected[["reason", "n_windows"]].values.tolist() == [["VOLTAGE_RANGE", 4]]

    future = datetime.now(timezone.utc) + timedelta(minutes=1)
    scored, rejected = read_prediction_logs(tmp_path / "logs", since=future)
    assert scored.empty
    assert rejected.empty


def test_prediction_log_failures_do_not_raise(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    log = PredictionLogger(blocker)  # can't create a directory here
    log.log_scored(synthetic_windows(1, seed=0), np.array([0.5]))
    log.log_scored(np.zeros((1, 10)), np.array([0.5]))  # features can't be computed
    log.log_rejected(1, "SHAPE")


def test_reading_an_empty_log_dir(tmp_path):
    scored, rejected = read_prediction_logs(tmp_path)
    assert scored.empty
    assert rejected.empty


# ── Drift check ──────────────────────────────────────────────────────────────


def reference_frame(n: int = 300) -> pd.DataFrame:
    frame = pd.DataFrame(compute_features(synthetic_windows(n, seed=100)))
    frame["prediction"] = np.random.default_rng(100).beta(2, 5, n)
    return frame


def write_traffic(log_dir, n: int, seed: int, **distortion) -> None:
    predictions = np.random.default_rng(seed).beta(2, 5, n)
    PredictionLogger(log_dir).log_scored(synthetic_windows(n, seed=seed, **distortion),
                                         predictions)


def run_drift(tmp_path, *extra) -> tuple[int, dict]:
    reference = tmp_path / "reference.csv"
    if not reference.exists():
        reference_frame().to_csv(reference, index=False)
    code = drift.main([
        "--reference", str(reference),
        "--log-dir", str(tmp_path / "logs"),
        "--output-dir", str(tmp_path / "out"),
        *extra,
    ])
    return code, json.loads((tmp_path / "out" / drift.SUMMARY_JSON).read_text())


def test_in_distribution_traffic_shows_no_drift(tmp_path):
    write_traffic(tmp_path / "logs", 80, seed=1)
    code, summary = run_drift(tmp_path, "--no-mlflow", "--fail-on-drift")
    assert code == 0
    assert summary["status"] == "no_drift"
    assert not summary["rejection_alert"]
    assert summary["n_current"] == 80
    assert summary["columns"]["prediction"]["method"] == drift.DRIFT_METHOD
    assert (tmp_path / "out" / drift.REPORT_HTML).exists()


def test_gain_change_is_flagged_as_drift(tmp_path):
    write_traffic(tmp_path / "logs", 80, seed=2, gain=1.5)
    code, summary = run_drift(tmp_path, "--no-mlflow", "--fail-on-drift")
    assert code == 1
    assert summary["status"] == "drift"
    columns = summary["columns"]
    for name in ("lead1_std_mv", "lead1_ptp_mv", "lead2_std_mv", "lead2_ptp_mv"):
        assert columns[name]["drifted"]
        assert columns[name]["current_median"] > columns[name]["reference_median"]
    assert not columns["heart_rate_bpm"]["drifted"]  # same rhythm, just louder
    # Without --fail-on-drift the check only reports.
    assert run_drift(tmp_path, "--no-mlflow")[0] == 0


def test_too_little_traffic_skips_the_test(tmp_path):
    write_traffic(tmp_path / "logs", 20, seed=3)
    code, summary = run_drift(tmp_path, "--no-mlflow", "--fail-on-drift")
    assert code == 0
    assert summary["status"] == "insufficient_data"
    assert not summary["rejection_alert"]
    assert not (tmp_path / "out" / drift.REPORT_HTML).exists()
    # Where traffic is expected (CI sends it), an empty-ish log is an alarm.
    assert run_drift(tmp_path, "--no-mlflow", "--fail-on-insufficient-data")[0] == 1


def test_high_rejection_rate_raises_an_alert(tmp_path):
    write_traffic(tmp_path / "logs", 40, seed=3)
    PredictionLogger(tmp_path / "logs").log_rejected(8, "VOLTAGE_RANGE")
    PredictionLogger(tmp_path / "logs").log_rejected(2, "NON_FINITE_VALUES")
    code, summary = run_drift(tmp_path, "--no-mlflow", "--fail-on-drift")
    assert code == 1
    assert summary["status"] == "no_drift"  # the inputs it did score look normal
    assert summary["rejection_alert"]
    assert summary["rejection_rate"] == pytest.approx(10 / 50)
    assert summary["rejection_reasons"] == {"VOLTAGE_RANGE": 8, "NON_FINITE_VALUES": 2}
    assert summary["model_run_ids"] == {"unknown": 50}  # counted in windows
    # A higher limit tolerates it.
    assert run_drift(tmp_path, "--no-mlflow", "--fail-on-drift",
                     "--max-rejection-rate", "0.25")[0] == 0


def test_drift_check_is_recorded_in_mlflow(tmp_path):
    config = tmp_path / "config.yaml"
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    config.write_text(yaml.safe_dump({"mlflow": {"tracking_uri": tracking_uri}}))
    # Created up front so run artifacts land in tmp_path, not ./mlruns.
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        drift.MLFLOW_EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    write_traffic(tmp_path / "logs", 60, seed=4, gain=1.5)

    _, summary = run_drift(tmp_path, "--config", str(config))

    [run] = mlflow.search_runs([experiment_id], output_format="list")
    assert run.data.tags["status"] == summary["status"] == "drift"
    assert run.data.metrics["n_current"] == 60
    assert "drift_score_lead1_noise_mv" in run.data.metrics
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id, "drift")}
    assert artifacts == {f"drift/{drift.REPORT_HTML}", f"drift/{drift.SUMMARY_JSON}"}


def test_parse_since():
    now = datetime.now(timezone.utc)
    assert abs(drift.parse_since("30m") - (now - timedelta(minutes=30))) < timedelta(seconds=5)
    assert drift.parse_since("2026-01-02T03:04:05") == datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc
    )
