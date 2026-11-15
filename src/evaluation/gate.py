"""
Model Promotion Gate
--------------------
Checks the evaluation report against the acceptance thresholds in the config.
If every check passes, the current ``challenger`` model version is promoted to
``champion`` in the MLflow Model Registry; otherwise the gate fails and the
champion is left unchanged. Either way the result is recorded on the model
version (``gate`` tag) and its run (``eval_*`` metrics).

The gate assumes the report was produced from the model that training just
registered, which is what the DVC pipeline guarantees (train -> evaluate -> gate).

Exit code: 0 = passed, 1 = failed.

Usage:
    python -m src.evaluation.gate --config configs/training_config.yaml \
        --report reports/evaluation_report.json --output reports/gate.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.registry import CHALLENGER_ALIAS, CHAMPION_ALIAS

logger = logging.getLogger(__name__)

# Where each gateable metric lives in reports/evaluation_report.json.
# Binary metrics are at the default 0.5 decision threshold.
METRIC_PATHS = {
    "sensitivity": ("binary_metrics", "sensitivity"),
    "specificity": ("binary_metrics", "specificity"),
    "precision": ("binary_metrics", "precision"),
    "f1_score": ("binary_metrics", "f1_score"),
    "roc_auc": ("roc_metrics", "roc_auc"),
    "average_precision": ("pr_metrics", "average_precision"),
}


def check_thresholds(report: dict, thresholds: dict) -> list[dict]:
    """Compare report metrics against minimum thresholds.

    Parameters
    ----------
    report : dict
        Evaluation report (see ``ModelEvaluator.generate_clinical_report``).
    thresholds : dict
        Metric name -> minimum acceptable value.

    Returns
    -------
    checks : list of dict
        One entry per metric: metric, value, minimum, passed.
    """
    if not thresholds:
        raise ValueError("No gate thresholds configured (gate.thresholds is empty).")

    checks = []
    for metric, minimum in thresholds.items():
        if metric not in METRIC_PATHS:
            raise ValueError(
                f"Unknown gate metric '{metric}'. Known: {', '.join(METRIC_PATHS)}"
            )
        section, key = METRIC_PATHS[metric]
        value = float(report[section][key])
        checks.append(
            {
                "metric": metric,
                "value": value,
                "minimum": float(minimum),
                "passed": value >= float(minimum),
            }
        )
    return checks


def record_and_promote(name: str, checks: list[dict], passed: bool, promote: bool) -> dict:
    """Record the gate result on the challenger version and promote it if it passed.

    Returns
    -------
    model : dict
        name, version (or None if there is no challenger) and promoted.
    """
    client = MlflowClient()
    try:
        challenger = client.get_model_version_by_alias(name, CHALLENGER_ALIAS)
    except MlflowException:
        logger.warning(
            f"No '{CHALLENGER_ALIAS}' version of '{name}' in the registry; nothing to promote."
        )
        return {"name": name, "version": None, "promoted": False}

    version = challenger.version
    for check in checks:
        client.log_metric(challenger.run_id, f"eval_{check['metric']}", check["value"])
    client.set_model_version_tag(name, version, "gate", "passed" if passed else "failed")

    promoted = passed and promote
    if promoted:
        client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
        logger.info(f"Promoted {name} version {version} to '{CHAMPION_ALIAS}'")
    elif passed:
        logger.info(f"{name} version {version} passed, but promotion is disabled (--no-promote)")
    else:
        logger.info(f"{name} version {version} failed the gate; '{CHAMPION_ALIAS}' unchanged")

    return {"name": name, "version": str(version), "promoted": promoted}


def format_summary(result: dict) -> str:
    """Render the gate result as a Markdown table (used for the CI job summary)."""
    status = "PASSED" if result["passed"] else "FAILED"
    lines = [
        f"### Promotion gate: {status}",
        "",
        "| Metric | Value | Minimum | Result |",
        "|---|---|---|---|",
    ]
    for c in result["checks"]:
        mark = "pass" if c["passed"] else "**FAIL**"
        lines.append(f"| {c['metric']} | {c['value']:.4f} | {c['minimum']:.4f} | {mark} |")

    name, version = result["model"]["name"], result["model"]["version"]
    if version is None:
        lines += ["", f"No `{CHALLENGER_ALIAS}` version of `{name}` found in the registry."]
    elif result["model"]["promoted"]:
        lines += ["", f"`{name}` v{version} promoted to `{CHAMPION_ALIAS}`."]
    else:
        lines += ["", f"`{name}` v{version} not promoted."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the gate. Returns the process exit code (0 = passed, 1 = failed)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Gate model promotion on evaluation metrics")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--report", default="reports/evaluation_report.json")
    parser.add_argument("--output", default="reports/gate.json")
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Check thresholds and record the result, but don't move the champion alias",
    )
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    checks = check_thresholds(report, config.get("gate", {}).get("thresholds", {}))
    passed = all(c["passed"] for c in checks)

    mlflow_config = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "sqlite:///mlflow_tracking.db"))
    model = record_and_promote(
        mlflow_config.get("registered_model_name", "vt-detector"),
        checks,
        passed,
        promote=not args.no_promote,
    )

    result = {"passed": passed, "checks": checks, "model": model}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = format_summary(result)
    print(summary)
    # GitHub Actions renders this file on the workflow run page.
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(summary)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
