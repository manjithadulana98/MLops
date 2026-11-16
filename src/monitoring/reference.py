"""
Reference profile for drift monitoring
--------------------------------------
Computes the monitoring features (see features.py) and the model's predicted
probability for every window of the held-out test split, and writes them as a
CSV. Live traffic logged by the served model is later compared against this
file by ``src.monitoring.drift``.

The test split is used because it comes from the same distribution as the
training data but was never trained on, so the model's predictions on it are
representative of what it should output on in-distribution traffic.

Usage:
    python -m src.monitoring.reference --config configs/training_config.yaml \
        --model models/artifacts/vgg16_best.keras --output data/monitoring/reference.csv

IEC 62304 Traceability
    Software Unit  : SU-MON-03
    Requirement Ref: SRS-MON-001 (Post-deployment input monitoring)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.ingestion.load_mit_dataset import MITDatasetLoader
from src.monitoring.features import FEATURE_COLUMNS, compute_features
from src.training.model import VGG16ECGModel

logger = logging.getLogger(__name__)


def build_reference(signals, probabilities) -> pd.DataFrame:
    """One row per window: monitoring features plus the model's prediction."""
    frame = pd.DataFrame(compute_features(signals), columns=FEATURE_COLUMNS)
    frame["prediction"] = probabilities
    return frame


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Build the drift-monitoring reference profile")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--model", default="models/artifacts/vgg16_best.keras")
    parser.add_argument("--output", default="data/monitoring/reference.csv")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f).get("dataset", {})

    # Same split parameters as training, so the cached test split is reused.
    loader = MITDatasetLoader(dataset_config.get("mit_path"), dataset_config.get("cache_dir"))
    *_, X_test, _ = loader.load_and_preprocess_all_records(
        test_size=dataset_config.get("test_size", 0.2),
        val_size=dataset_config.get("val_size", 0.1),
        sample_fraction=dataset_config.get("sample_fraction", 1.0),
    )

    model = VGG16ECGModel()
    model.load(args.model)
    logger.info(f"Scoring {len(X_test)} test windows for the reference profile...")
    reference = build_reference(X_test, model.predict(X_test))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output, index=False)
    logger.info(f"Reference profile ({len(reference)} windows) saved to {output}")


if __name__ == "__main__":
    main()
