"""Prepare and cache MIT-BIH data splits for the DVC `prepare` stage.

Usage:
    python -m src.ingestion.prepare --config configs/training_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.ingestion.load_mit_dataset import MITDatasetLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH cached dataset splits")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds = config.get("dataset", {})
    loader = MITDatasetLoader(
        ds.get("mit_path", "data/MIT_dataset"),
        ds.get("cache_dir", "data/processed/mit_cache"),
    )

    loader.load_and_preprocess_all_records(
        test_size=ds.get("test_size", 0.2),
        val_size=ds.get("val_size", 0.1),
        sample_fraction=ds.get("sample_fraction", 1.0),
        force_recompute=ds.get("force_recompute", False),
    )


if __name__ == "__main__":
    main()
