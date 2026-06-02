"""
Model Evaluation & Metrics
---------------------------
Comprehensive evaluation metrics for VT detection model.

IEC 62304 Traceability
    Software Unit  : SU-EVAL-01
    Requirement Ref: SRS-EVAL-001 (Model validation & safety metrics)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    matthews_corrcoef,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Computes clinical and statistical metrics for model performance."""

    def __init__(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Initialize evaluator with ground truth and predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground truth labels (0=normal, 1=VT/PVC).
        y_pred_proba : np.ndarray
            Predicted probabilities [0, 1] from model.
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba

        # Default decision threshold = 0.5
        self.y_pred = (y_pred_proba >= 0.5).astype(int)

        logger.info(
            f"Evaluator initialized: {len(y_true)} samples, "
            f"class distribution: {np.mean(y_true):.1%} positive"
        )

    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute 2×2 confusion matrix.

        Returns
        -------
        cm : np.ndarray
            Shape (2, 2): [[TN, FP], [FN, TP]]
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        return cm

    def compute_binary_metrics(self, threshold: float = 0.5) -> dict:
        """Compute binary classification metrics at given threshold.

        Parameters
        ----------
        threshold : float
            Classification threshold. Default: 0.5.

        Returns
        -------
        metrics : dict
            Dictionary with sensitivity, specificity, precision, etc.
        """
        y_pred_at_threshold = (self.y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(self.y_true, y_pred_at_threshold)

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # aka recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = f1_score(self.y_true, y_pred_at_threshold)
        mcc = matthews_corrcoef(self.y_true, y_pred_at_threshold)

        metrics = {
            "threshold": float(threshold),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1_score": float(f1),
            "matthews_corrcoef": float(mcc),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        return metrics

    def compute_roc_metrics(self) -> dict:
        """Compute ROC curve and AUC.

        Returns
        -------
        roc_metrics : dict
            AUC, FPR array, TPR array.
        """
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)

        metrics = {
            "roc_auc": float(roc_auc),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

        logger.info(f"ROC AUC: {roc_auc:.4f}")
        return metrics

    def compute_pr_metrics(self) -> dict:
        """Compute precision-recall curve.

        Returns
        -------
        pr_metrics : dict
            Average precision, precision array, recall array.
        """
        precision, recall, thresholds = precision_recall_curve(
            self.y_true, self.y_pred_proba
        )
        ap = auc(recall, precision)

        metrics = {
            "average_precision": float(ap),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }

        logger.info(f"Average Precision: {ap:.4f}")
        return metrics

    def plot_roc_curve(self, output_path: str = "plots/roc_curve.png") -> None:
        """Generate and save ROC curve.

        Parameters
        ----------
        output_path : str
            Where to save the plot.
        """
        roc_metrics = self.compute_roc_metrics()
        fpr = np.array(roc_metrics["fpr"])
        tpr = np.array(roc_metrics["tpr"])
        auc_score = roc_metrics["roc_auc"]

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkblue", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve – VT Detection Model")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"ROC curve saved: {output_path}")
        plt.close()

    def plot_confusion_matrix(self, output_path: str = "plots/confusion_matrix.png") -> None:
        """Generate and save confusion matrix heatmap.

        Parameters
        ----------
        output_path : str
            Where to save the plot.
        """
        cm = self.compute_confusion_matrix()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "VT/PVC"])
        ax.set_yticklabels(["Normal", "VT/PVC"])
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title("Confusion Matrix – VT Detection Model")

        # Annotate cells
        for i in range(2):
            for j in range(2):
                text = ax.text(
                    j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold",
                )

        fig.colorbar(im, ax=ax)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Confusion matrix saved: {output_path}")
        plt.close()

    def plot_pr_curve(self, output_path: str = "plots/pr_curve.png") -> None:
        """Generate and save precision-recall curve.

        Parameters
        ----------
        output_path : str
            Where to save the plot.
        """
        pr_metrics = self.compute_pr_metrics()
        precision = np.array(pr_metrics["precision"])
        recall = np.array(pr_metrics["recall"])
        ap = pr_metrics["average_precision"]

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="darkgreen", lw=2, label=f"PR curve (AP = {ap:.3f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve – VT Detection Model")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"PR curve saved: {output_path}")
        plt.close()

    def generate_clinical_report(self) -> dict:
        """Generate comprehensive clinical evaluation report.

        Returns
        -------
        report : dict
            Full evaluation report with all key metrics.
        """
        report = {
            "binary_metrics": self.compute_binary_metrics(threshold=0.5),
            "roc_metrics": self.compute_roc_metrics(),
            "pr_metrics": self.compute_pr_metrics(),
        }

        # Find optimal threshold for F1
        best_f1_metrics = self._find_optimal_threshold()
        report["optimal_threshold_metrics"] = best_f1_metrics

        logger.info("Clinical report generated")
        return report

    def _find_optimal_threshold(self) -> dict:
        """Find threshold that maximizes F1-score.

        Returns
        -------
        best_metrics : dict
        """
        best_f1 = -1
        best_metrics = None

        for threshold in np.linspace(0.1, 0.9, 50):
            metrics = self.compute_binary_metrics(threshold=threshold)
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_metrics = metrics

        logger.info(f"Optimal threshold: {best_metrics['threshold']:.3f}, F1: {best_f1:.3f}")
        return best_metrics
