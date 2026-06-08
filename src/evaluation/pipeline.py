"""
Complete inference/evaluation pipeline
---------------------------------------
Load trained model and evaluate on test set with comprehensive metrics.

Usage:
    python -m src.evaluation.pipeline --model models/artifacts/vgg16_best.keras
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from src.ingestion.load_mit_dataset import MITDatasetLoader
from src.training.model import VGG16ECGModel
from src.evaluation.evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Load model and generate comprehensive evaluation report."""
    parser = argparse.ArgumentParser(description="Evaluate trained VT model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.keras file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Training config for data paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/evaluation_report.json",
        help="Output report path",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {args.config}")

    # Load dataset
    dataset_config = config.get("dataset", {})
    loader = MITDatasetLoader(
        dataset_config.get("mit_path"),
        dataset_config.get("cache_dir"),
    )

    logger.info("Loading dataset...")
    _, _, _, _, X_test, y_test = loader.load_and_preprocess_all_records()

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = VGG16ECGModel()
    model.load(args.model)

    # Generate predictions
    logger.info("Generating predictions on test set...")
    y_pred_proba = model.predict(X_test)

    # Evaluate
    evaluator = ModelEvaluator(y_test, y_pred_proba)

    logger.info("Computing evaluation metrics...")
    report = evaluator.generate_clinical_report()

    # Generate plots
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    evaluator.plot_roc_curve(str(plot_dir / "roc_curve.png"))
    evaluator.plot_confusion_matrix(str(plot_dir / "confusion_matrix.png"))
    evaluator.plot_pr_curve(str(plot_dir / "pr_curve.png"))

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"✓ Evaluation report saved: {output_path}")
    logger.info(f"✓ Plots saved to {plot_dir}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    binary_metrics = report["binary_metrics"]
    optimal_metrics = report.get("optimal_threshold_metrics", {})
    print(f"Sensitivity (Recall): {binary_metrics['sensitivity']:.4f}")
    print(f"Specificity:          {binary_metrics['specificity']:.4f}")
    print(f"Precision:            {binary_metrics['precision']:.4f}")
    print(f"F1-Score:             {binary_metrics['f1_score']:.4f}")
    print(f"ROC AUC:              {report['roc_metrics']['roc_auc']:.4f}")
    print(f"Avg Precision:        {report['pr_metrics']['average_precision']:.4f}")
    if optimal_metrics:
        print("-" * 70)
        print(
            f"Best threshold (F1):  {optimal_metrics['threshold']:.3f} | "
            f"Sensitivity: {optimal_metrics['sensitivity']:.4f} | "
            f"Precision: {optimal_metrics['precision']:.4f} | "
            f"F1: {optimal_metrics['f1_score']:.4f}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
