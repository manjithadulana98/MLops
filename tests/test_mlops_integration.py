"""
MLOps pipeline integration tests
================================

Covers the pieces the DVC pipeline wires together:

1. MIT-BIH loading, splitting and cache behaviour (needs data/MIT_dataset)
2. Model build / train / evaluate / predict / save / load on synthetic signals
3. MLflow Model Registry: registration, challenger alias and lineage tags

Tests that need the real dataset are *skipped* (not silently passed) when it
isn't present, e.g. in CI. Fetch it with `dvc pull`. They use a single record
so they run in seconds; nothing is written outside pytest's tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.ingestion import load_mit_dataset
from src.ingestion.load_mit_dataset import MITDatasetLoader
from src.ingestion.validate import validate_ecg_signal
from src.training.model import VGG16ECGModel

DATASET = Path("data/MIT_dataset")
# Record 119 has ~400 ventricular beats, so stratified splits have both classes.
FAST_RECORD = "119"

needs_dataset = pytest.mark.skipif(
    not (DATASET / f"{FAST_RECORD}.hea").exists(),
    reason="MIT-BIH dataset not present (run `dvc pull`)",
)


def synthetic_ecg(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Small batch of ECG-shaped signals (n, 3600, 2) with alternating labels."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, 3600, dtype=np.float32)
    base = np.sin(2 * np.pi * 1.2 * t)[None, :, None]
    X = base + 0.1 * rng.standard_normal((n, 3600, 2))
    y = np.arange(n) % 2
    return X.astype(np.float32), y.astype(np.uint8)


@pytest.fixture(scope="module")
def built_model() -> VGG16ECGModel:
    model = VGG16ECGModel(input_shape=(3600, 2), dropout_rate=0.5, learning_rate=1e-3)
    model.build()
    return model


# ── 1. Data ──────────────────────────────────────────────────────────────────


@needs_dataset
def test_load_record_and_validate(tmp_path):
    loader = MITDatasetLoader(DATASET, cache_dir=tmp_path)
    signals, indices, types = loader.load_record(FAST_RECORD)

    assert signals.ndim == 2 and signals.shape[1] == 2
    assert len(indices) == len(types) > 0
    validate_ecg_signal(signals[:3600, 0])  # raises if the signal is non-compliant


@needs_dataset
def test_split_proportions_and_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(load_mit_dataset, "MIT_RECORDS", [FAST_RECORD])
    loader = MITDatasetLoader(DATASET, cache_dir=tmp_path)

    full = loader.load_and_preprocess_all_records(
        test_size=0.1, val_size=0.2, sample_fraction=1.0, force_recompute=True
    )
    X_tr, y_tr, X_va, y_va, X_te, y_te = loader.load_and_preprocess_all_records(
        test_size=0.1, val_size=0.2, sample_fraction=0.5, force_recompute=True
    )

    n_full = sum(len(a) for a in full[::2])
    n = len(X_tr) + len(X_va) + len(X_te)
    # sample_fraction shrinks every split, not just train
    assert n == pytest.approx(0.5 * n_full, rel=0.1)
    assert len(X_te) / n == pytest.approx(0.1, abs=0.02)
    assert len(X_va) / n == pytest.approx(0.2, abs=0.02)
    assert X_tr.shape[1:] == (3600, 2)
    for labels in (y_tr, y_va, y_te):
        assert set(np.unique(labels)) == {0, 1}


@needs_dataset
def test_cache_reused_only_for_same_split_params(tmp_path, monkeypatch):
    monkeypatch.setattr(load_mit_dataset, "MIT_RECORDS", [FAST_RECORD])
    loader = MITDatasetLoader(DATASET, cache_dir=tmp_path)
    params = dict(test_size=0.1, val_size=0.2, sample_fraction=0.5)

    first = loader.load_and_preprocess_all_records(**params)
    cached = loader.load_and_preprocess_all_records(**params)
    np.testing.assert_array_equal(first[4], cached[4])

    # A different split must not be served from the stale cache.
    resplit = loader.load_and_preprocess_all_records(**{**params, "test_size": 0.3})
    assert len(resplit[4]) > 2 * len(first[4])


# ── 2. Model ─────────────────────────────────────────────────────────────────


def test_model_architecture(built_model):
    keras_model = built_model.model
    assert keras_model.input_shape == (None, 224, 224, 3)
    assert keras_model.output_shape == (None, 1)
    assert not keras_model.layers[0].trainable  # VGG16 base stays frozen


def test_train_evaluate_save_load(built_model, tmp_path):
    X, y = synthetic_ecg(8)

    history = built_model.train(
        X[:4], y[:4], X[4:6], y[4:6],
        batch_size=2,
        epochs=1,
        checkpoint_path=str(tmp_path / "ckpt.keras"),
        save_best_checkpoint=False,
    )
    assert len(history["loss"]) == 1 and np.isfinite(history["loss"][0])

    metrics = built_model.evaluate(X[6:], y[6:], batch_size=2)
    assert {"loss", "accuracy", "auc"} <= metrics.keys()

    preds = built_model.predict(X[6:])
    assert preds.shape == (2,)
    assert np.all((preds >= 0) & (preds <= 1))

    path = tmp_path / "model.keras"
    built_model.save(str(path))
    reloaded = VGG16ECGModel()
    reloaded.load(str(path))
    np.testing.assert_allclose(reloaded.predict(X[6:]), preds, rtol=1e-4, atol=1e-6)


# ── 3. Model Registry ────────────────────────────────────────────────────────


def test_register_model_sets_challenger_alias_and_lineage(built_model, tmp_path):
    from src.training.train import CHALLENGER_ALIAS, TrainingPipeline

    name = "vt-detector-test"
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    try:
        exp_id = mlflow.create_experiment(
            "registry-test", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        pipeline = TrainingPipeline(
            {"dataset": {"mit_path": str(DATASET)}, "mlflow": {"registered_model_name": name}}
        )

        lineage = pipeline.lineage_tags()
        assert len(lineage["git_commit"]) == 40
        assert lineage["git_dirty"] in {"true", "false"}
        assert lineage["data_md5"].endswith(".dir")  # read from data/MIT_dataset.dvc

        keras_path = tmp_path / "model.keras"
        built_model.save(str(keras_path))
        X, _ = synthetic_ecg(3)
        for _ in range(2):
            with mlflow.start_run(experiment_id=exp_id):
                pipeline.register_model(built_model, lineage, str(keras_path), X[:1])

        client = MlflowClient()
        challenger = client.get_model_version_by_alias(name, CHALLENGER_ALIAS)
        # The alias follows the newest version; older versions stay in the registry.
        assert str(challenger.version) == "2"
        assert client.get_model_version(name, "1").tags == lineage
        assert challenger.tags == lineage

        # The registered version is a self-contained pyfunc: raw ECG in, probability out.
        served = mlflow.pyfunc.load_model(f"models:/{name}@{CHALLENGER_ALIAS}")
        assert served.metadata.get_input_schema().inputs[0].shape == (-1, 3600, 2)
        np.testing.assert_allclose(
            served.predict(X), built_model.predict(X), rtol=1e-4, atol=1e-6
        )
        # Bad input surfaces as MlflowException, which the scoring server maps to HTTP 400.
        with pytest.raises(MlflowException, match="VOLTAGE_RANGE"):
            served.predict(X * 100)  # ~130 mV: outside the +/-5 mV physiological range
        with pytest.raises(MlflowException, match="shape"):
            served.predict(np.zeros((1, 100, 2), dtype=np.float32))
    finally:
        mlflow.set_tracking_uri(previous_uri)
