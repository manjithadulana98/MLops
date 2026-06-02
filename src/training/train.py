"""
VT Detection Model Training Pipeline
-------------------------------------
End-to-end training orchestration with MLflow experiment tracking.

IEC 62304 Traceability
    Software Unit  : SU-TRAIN-02
    Requirement Ref: SRS-TRAIN-001 (Training workflow & reproducibility)

Usage:
    python -m src.training.train --config configs/training_config.yaml
"""

import argparse
import json
import logging.config
import os
from pathlib import Path

import numpy as np
import mlflow
import yaml

from src.ingestion.validate import validate_ecg_signal, ClinicalDataError
from src.ingestion.load_mit_dataset import MITDatasetLoader
from src.training.model import VGG16ECGModel

# ── Logging configuration ────────────────────────────────────────────────────

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": "training.log",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {"handlers": ["default", "file"], "level": "DEBUG"},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates data loading, model training, and MLflow logging."""

    def __init__(self, config: dict) -> None:
        """Initialize pipeline with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary (loaded from YAML).
        """
        self.config = config
        self.loader = None
        self.model = None

        logger.info(f"TrainingPipeline initialized with config keys: {config.keys()}")

    def load_dataset(self) -> tuple:
        """Load MIT-BIH dataset with validation.

        Returns
        -------
        X_train, y_train, X_val, y_val, X_test, y_test : tuple of np.ndarray
        """
        dataset_config = self.config.get("dataset", {})
        dataset_path = dataset_config.get("mit_path", "data/MIT_dataset")
        cache_dir = dataset_config.get("cache_dir", "data/processed/mit_cache")

        logger.info(f"Loading dataset from {dataset_path}")

        self.loader = MITDatasetLoader(dataset_path, cache_dir)
        X_train, y_train, X_val, y_val, X_test, y_test = (
            self.loader.load_and_preprocess_all_records(
                test_size=dataset_config.get("test_size", 0.2),
                val_size=dataset_config.get("val_size", 0.1),
                force_recompute=dataset_config.get("force_recompute", False),
            )
        )

        # Sampling for quick development iteration (optional)
        sample_fraction = dataset_config.get("sample_fraction", 1.0)
        if sample_fraction < 1.0:
            n_train = int(X_train.shape[0] * sample_fraction)
            X_train, y_train = X_train[:n_train], y_train[:n_train]
            logger.info(f"Downsampled training to {n_train} samples")

        logger.info(
            f"Dataset loaded: train {X_train.shape}, val {X_val.shape}, "
            f"test {X_test.shape}"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_model(self) -> VGG16ECGModel:
        """Instantiate and build VGG16 transfer learning model.

        Returns
        -------
        model : VGG16ECGModel
        """
        model_config = self.config.get("model", {})

        self.model = VGG16ECGModel(
            input_shape=tuple(model_config.get("input_shape", [3600, 2])),
            dropout_rate=model_config.get("dropout_rate", 0.5),
            learning_rate=model_config.get("learning_rate", 1e-3),
        )

        self.model.build()
        self.model.summary()
        logger.info("Model built successfully")

        return self.model

    def train_model(
        self,
        model: VGG16ECGModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Train model with MLflow tracking.

        Parameters
        ----------
        model : VGG16ECGModel
            Compiled model instance.
        X_train, y_train, X_val, y_val : np.ndarray
            Training and validation data.

        Returns
        -------
        history : dict
            Training history (loss, acc, etc. per epoch).
        """
        train_config = self.config.get("training", {})
        checkpoint_path = self.config.get("paths", {}).get(
            "checkpoint", "models/artifacts/vgg16_best.keras"
        )

        logger.info("Starting model training...")

        history = model.train(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=train_config.get("batch_size", 32),
            epochs=train_config.get("epochs", 20),
            checkpoint_path=checkpoint_path,
        )

        return history

    def evaluate_model(
        self,
        model: VGG16ECGModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Evaluate model on test set.

        Parameters
        ----------
        model : VGG16ECGModel
            Trained model.
        X_test, y_test : np.ndarray
            Test data.

        Returns
        -------
        metrics : dict
            Test set metrics.
        """
        logger.info("Evaluating model on test set...")
        metrics = model.evaluate(X_test, y_test)
        logger.info(f"Test metrics: {metrics}")

        return metrics

    def run(self) -> None:
        """Execute the full training pipeline with MLflow logging."""
        mlflow_config = self.config.get("mlflow", {})
        experiment_name = mlflow_config.get("experiment_name", "VT-Detection-Default")
        run_name = mlflow_config.get("run_name", "baseline-vgg16")

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            try:
                # Log configuration
                mlflow.log_params(self.flatten_dict(self.config))

                # Load data
                X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset()

                # Log dataset sizes
                mlflow.log_metric("train_samples", X_train.shape[0])
                mlflow.log_metric("val_samples", X_val.shape[0])
                mlflow.log_metric("test_samples", X_test.shape[0])
                mlflow.log_metric("vt_class_ratio_train", np.mean(y_train))

                # Build model
                model = self.build_model()

                # Train
                history = self.train_model(model, X_train, y_train, X_val, y_val)

                # Log training history (last epoch)
                for metric_name, metric_values in history.items():
                    if isinstance(metric_values, list) and len(metric_values) > 0:
                        mlflow.log_metric(
                            f"final_{metric_name}", metric_values[-1]
                        )

                # Evaluate
                test_metrics = self.evaluate_model(model, X_test, y_test)

                # Log test metrics
                for metric_name, metric_value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", metric_value)

                # Save model
                checkpoint_path = self.config.get("paths", {}).get(
                    "checkpoint", "models/artifacts/vgg16_best.keras"
                )
                model.save(checkpoint_path)
                mlflow.log_artifact(checkpoint_path)

                logger.info("✓ Training pipeline completed successfully")
                logger.info(f"MLflow run: {mlflow.active_run().info.run_id}")

            except Exception as e:
                logger.error(f"✗ Training failed: {e}", exc_info=True)
                raise

    @staticmethod
    def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
        """Flatten nested dict for MLflow parameter logging.

        Parameters
        ----------
        d : dict
            Nested dictionary.
        parent_key : str
            Prefix for flattened keys.
        sep : str
            Separator for nested keys.

        Returns
        -------
        flat : dict
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    TrainingPipeline.flatten_dict(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, str(v)))
        return dict(items)


def main():
    """Entry point for training script."""
    parser = argparse.ArgumentParser(description="Train VT detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration YAML",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")

    # Run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
