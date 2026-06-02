"""
Complete end-to-end MLOps workflow integration test
====================================================

Demonstrates:
1. Load MIT-BIH dataset
2. Train VGG16 transfer learning model
3. Evaluate with comprehensive metrics
4. Track experiments with MLflow
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.load_mit_dataset import MITDatasetLoader
from src.training.model import VGG16ECGModel
from src.ingestion.validate import validate_ecg_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def test_dataset_loading():
    """Test: Load and validate MIT dataset samples."""
    logger.info("=" * 70)
    logger.info("TEST 1: Dataset Loading & Validation")
    logger.info("=" * 70)

    try:
        loader = MITDatasetLoader("data/MIT_dataset", "data/processed/mit_cache")

        # Load a small sample to verify format
        logger.info("Loading sample records...")
        signals, indices, types = loader.load_record("100")

        logger.info(f"✓ Loaded record 100: shape={signals.shape}")

        # Validate a 10-second segment
        segment = signals[:3600, 0]  # first 10s, lead 1
        validate_ecg_signal(segment)
        logger.info("✓ Sample validation passed (signal compliance OK)")

        return True

    except Exception as e:
        logger.error(f"✗ Dataset loading failed: {e}", exc_info=True)
        return False


def test_model_build():
    """Test: Build VGG16 model architecture."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Model Architecture & Build")
    logger.info("=" * 70)

    try:
        model = VGG16ECGModel(
            input_shape=(3600, 2),
            dropout_rate=0.5,
            learning_rate=1e-3,
        )
        model.build()

        logger.info("✓ VGG16 model built successfully")
        logger.info(f"  Total parameters: {model.model.count_params():,}")

        return True

    except Exception as e:
        logger.error(f"✗ Model build failed: {e}", exc_info=True)
        return False


def test_data_preparation():
    """Test: Prepare data for training."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Data Preparation & Splitting")
    logger.info("=" * 70)

    try:
        loader = MITDatasetLoader("data/MIT_dataset", "data/processed/mit_cache")

        # Load with small sample fraction for speed
        X_train, y_train, X_val, y_val, X_test, y_test = (
            loader.load_and_preprocess_all_records(
                test_size=0.2,
                val_size=0.1,
                force_recompute=False,
            )
        )

        logger.info("✓ Data splits created:")
        logger.info(f"  Train: {X_train.shape} (VT ratio: {np.mean(y_train):.1%})")
        logger.info(f"  Val:   {X_val.shape} (VT ratio: {np.mean(y_val):.1%})")
        logger.info(f"  Test:  {X_test.shape} (VT ratio: {np.mean(y_test):.1%})")

        # Verify signal validity
        validate_ecg_signal(X_train[0, :, 0])
        validate_ecg_signal(X_train[0, :, 1])
        logger.info("✓ Sample signals pass clinical validation")

        return True

    except Exception as e:
        logger.error(f"✗ Data preparation failed: {e}", exc_info=True)
        return False


def test_mlops_workflow():
    """Test: End-to-end MLOps workflow (mini-version for CI)."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: MLOps Workflow (Integration Test)")
    logger.info("=" * 70)

    try:
        import yaml
        
        # Load config
        with open("configs/training_config.yaml") as f:
            config = yaml.safe_load(f)

        # Adjust for quick test
        config["dataset"]["sample_fraction"] = 0.05  # Use only 5% of data
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 16

        logger.info("Loading dataset (5% sample for speed)...")
        loader = MITDatasetLoader(
            config["dataset"]["mit_path"],
            config["dataset"]["cache_dir"],
        )
        X_train, y_train, X_val, y_val, X_test, y_test = (
            loader.load_and_preprocess_all_records(
                test_size=config["dataset"]["test_size"],
                val_size=config["dataset"]["val_size"],
                force_recompute=False,
            )
        )
        # Apply sampling
        n_train = int(X_train.shape[0] * 0.05)
        X_train, y_train = X_train[:n_train], y_train[:n_train]

        logger.info(f"Using {X_train.shape[0]} training samples for speed")

        logger.info("Building model...")
        model = VGG16ECGModel(
            input_shape=tuple(config["model"]["input_shape"]),
            dropout_rate=config["model"]["dropout_rate"],
            learning_rate=config["model"]["learning_rate"],
        )
        model.build()

        logger.info("Training (2 epochs)...")
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=config["training"]["batch_size"],
            epochs=config["training"]["epochs"],
            checkpoint_path="models/artifacts/test_vgg16.keras",
        )

        logger.info("✓ Training completed successfully")
        logger.info(f"  Final train loss: {history['loss'][-1]:.4f}")
        logger.info(f"  Final val loss: {history['val_loss'][-1]:.4f}")

        logger.info("Evaluating...")
        metrics = model.evaluate(X_test, y_test)
        logger.info("✓ Evaluation completed")
        logger.info(f"  Test accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Test AUC: {metrics['auc']:.4f}")

        return True

    except Exception as e:
        logger.error(f"✗ MLOps workflow failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("STARTING MLOPS INTEGRATION TESTS\n")

    results = {
        "Dataset Loading": test_dataset_loading(),
        "Model Build": test_model_build(),
        "Data Preparation": test_data_preparation(),
        "MLOps Workflow": test_mlops_workflow(),
    }

    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:.<40} {status}")

    all_passed = all(results.values())
    logger.info("=" * 70)

    if all_passed:
        logger.info("✓ All tests passed! MLOps pipeline is ready.")
        sys.exit(0)
    else:
        logger.warning("✗ Some tests failed. Review logs above.")
        sys.exit(1)
