"""
Drift check: live traffic vs. the reference profile
---------------------------------------------------
Reads the prediction log written by the served model, compares its feature
and prediction distributions with the reference profile (see reference.py)
using Evidently, and records the result:

    reports/drift/drift_report.html   interactive Evidently report
    reports/drift/drift.json          summary: per-column scores and verdicts
    MLflow experiment "vt-detector-monitoring": one run per check

Each column is tested with a two-sample Kolmogorov-Smirnov test (drifted if
p < 0.05), and the traffic counts as drifted when at least ``--drift-share``
of the columns drift. K-S is set explicitly because for references over 1000
rows Evidently otherwise switches to a normalised Wasserstein distance with a
fixed 0.1 threshold, which flags small batches of in-distribution traffic.

The default share is 0.3 (3 of the 10 columns), not Evidently's 0.5: at
p < 0.05 about 0.5 columns drift by chance, so 3 is already unusual, while a
single-cause problem moves only the columns it touches (electrode noise moves
the two noise features, the heart-rate estimate and the prediction: 4 columns,
which a 0.5 share misses).

Live traffic has no labels, so this watches what the model receives and what
it outputs (prediction drift), not its accuracy. Requests the model refused
(clinical validation) never reach the drift test, so their share is checked
separately against ``--max-rejection-rate``.

Exit code: 0, or 1 with --fail-on-drift on drift or a rejection-rate alert,
and with --fail-on-insufficient-data when too few windows were scored.

Usage:
    python -m src.monitoring.drift --log-dir monitoring/predictions
    python -m src.monitoring.drift --log-dir monitoring/predictions --since 30m

IEC 62304 Traceability
    Software Unit  : SU-MON-04
    Requirement Ref: SRS-MON-002 (Input and prediction drift detection)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.monitoring.features import FEATURE_COLUMNS
from src.monitoring.prediction_log import read_prediction_logs

logger = logging.getLogger(__name__)

MONITORED_COLUMNS = FEATURE_COLUMNS + ["prediction"]
DRIFT_METHOD = "ks"
DEFAULT_DRIFT_SHARE = 0.3  # see module docstring
MLFLOW_EXPERIMENT = "vt-detector-monitoring"
REPORT_HTML = "drift_report.html"
SUMMARY_JSON = "drift.json"


def parse_since(value: str) -> datetime:
    """Parse ``--since``: an ISO 8601 time, or a relative age like 30m, 2h, 1d."""
    match = re.fullmatch(r"(\d+)([mhd])", value.strip())
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        delta = {"m": timedelta(minutes=amount), "h": timedelta(hours=amount),
                 "d": timedelta(days=amount)}[unit]
        return datetime.now(timezone.utc) - delta
    parsed = datetime.fromisoformat(value)
    # Treat a time without an offset as UTC, matching the log timestamps.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, drift_share: float):
    """Run Evidently's data drift preset over the monitored columns.

    Returns
    -------
    result : dict
        ``dataset_drift``, ``drifted_share`` and per-column ``columns`` details.
    snapshot : evidently Snapshot
        The full report (``snapshot.save_html(path)``).
    """
    definition = DataDefinition(numerical_columns=MONITORED_COLUMNS)
    report = Report(
        [DataDriftPreset(columns=MONITORED_COLUMNS, num_method=DRIFT_METHOD,
                         drift_share=drift_share)],
        include_tests=True,
    )
    snapshot = report.run(
        current_data=Dataset.from_pandas(current[MONITORED_COLUMNS], data_definition=definition),
        reference_data=Dataset.from_pandas(
            reference[MONITORED_COLUMNS], data_definition=definition
        ),
    )
    raw = snapshot.dict()

    # Evidently's own per-column verdicts: a failed "drift" test means drift.
    drifted = {
        test["metric_config"]["params"]["column"]: test["status"] == "FAIL"
        for test in raw["tests"]
        if test["id"] == "drift"
    }
    columns, drifted_share = {}, None
    for metric in raw["metrics"]:
        config = metric["config"]
        if config["type"].endswith(":ValueDrift"):
            name = config["column"]
            columns[name] = {
                "method": config["method"],
                "threshold": config["threshold"],
                "score": float(metric["value"]),
                "drifted": drifted[name],
                "reference_median": float(reference[name].median()),
                "current_median": float(current[name].median()),
            }
        elif config["type"].endswith(":DriftedColumnsCount"):
            drifted_share = float(metric["value"]["share"])

    result = {
        "dataset_drift": drifted_share >= drift_share,
        "drifted_share": drifted_share,
        "columns": columns,
    }
    return result, snapshot


def traffic_summary(scored: pd.DataFrame, rejected: pd.DataFrame) -> dict:
    """Volume, rejection rate and which model versions served the traffic."""
    n_rejected = int(rejected["n_windows"].sum()) if not rejected.empty else 0
    n_total = len(scored) + n_rejected
    reasons = (
        rejected.groupby("reason")["n_windows"].sum().astype(int).to_dict()
        if not rejected.empty else {}
    )
    both = pd.concat([scored, rejected], ignore_index=True)
    run_ids = {}
    if not both.empty:
        # Windows per model: a scored record is one window, a rejected one is a request.
        windows = both["n_windows"].fillna(1) if "n_windows" in both else pd.Series(1, both.index)
        by_model = windows.groupby(both["model_run_id"].fillna("unknown")).sum()
        run_ids = by_model.astype(int).to_dict()
    return {
        "n_current": len(scored),
        "n_rejected_windows": n_rejected,
        "rejection_rate": n_rejected / n_total if n_total else 0.0,
        "rejection_reasons": reasons,
        "model_run_ids": run_ids,
        "first_timestamp": str(both["timestamp"].min()) if not both.empty else None,
        "last_timestamp": str(both["timestamp"].max()) if not both.empty else None,
    }


def format_summary(summary: dict) -> str:
    """Render the check as Markdown (printed, and used for the CI job summary)."""
    status = {"drift": "DRIFT DETECTED", "no_drift": "no drift",
              "insufficient_data": "not enough traffic"}[summary["status"]]
    if summary["rejection_alert"]:
        status += f"; REJECTION RATE ABOVE {summary['max_rejection_rate']:.0%}"
    lines = [
        f"### Drift check: {status}",
        "",
        f"{summary['n_current']} scored windows vs {summary['n_reference']} reference windows; "
        f"{summary['n_rejected_windows']} windows rejected "
        f"({summary['rejection_rate']:.1%}).",
    ]
    if summary["status"] == "insufficient_data":
        lines += ["", f"Need at least {summary['min_windows']} scored windows to test for drift."]
        return "\n".join(lines) + "\n"

    drifted = [name for name, col in summary["columns"].items() if col["drifted"]]
    lines += [
        f"{len(drifted)} of {len(summary['columns'])} columns drifted"
        f"{': ' + ', '.join(drifted) if drifted else ''} "
        f"(dataset drift at >= {summary['drift_share_threshold']:.0%}).",
        "",
        "| Column | Reference median | Current median | K-S p-value | Drifted |",
        "|---|---|---|---|---|",
    ]
    for name, col in summary["columns"].items():
        mark = "**yes**" if col["drifted"] else "no"
        lines.append(
            f"| {name} | {col['reference_median']:.3f} | {col['current_median']:.3f} "
            f"| {col['score']:.3g} | {mark} |"
        )
    return "\n".join(lines) + "\n"


def log_to_mlflow(summary: dict, output_dir: Path, run_name: str) -> str:
    """Record the check as a run in the monitoring experiment; returns the run id."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "reference": summary["reference"],
            "log_dir": summary["log_dir"],
            "since": summary["since"] or "all",
            "drift_method": DRIFT_METHOD,
            "drift_share_threshold": summary["drift_share_threshold"],
        })
        mlflow.set_tags({
            "status": summary["status"],
            "rejection_alert": str(summary["rejection_alert"]).lower(),
        })
        metrics = {
            "n_current": summary["n_current"],
            "n_rejected_windows": summary["n_rejected_windows"],
            "rejection_rate": summary["rejection_rate"],
        }
        if summary["status"] != "insufficient_data":
            metrics["drifted_share"] = summary["drifted_share"]
            for name, col in summary["columns"].items():
                metrics[f"drift_score_{name}"] = col["score"]
                metrics[f"current_median_{name}"] = col["current_median"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output_dir), artifact_path="drift")
        return run.info.run_id


def main(argv: list[str] | None = None) -> int:
    """Run the drift check. Returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Compare served traffic with the reference")
    parser.add_argument("--config", default="configs/training_config.yaml",
                        help="Read for the MLflow tracking URI")
    parser.add_argument("--reference", default="data/monitoring/reference.csv")
    parser.add_argument("--log-dir", default="monitoring/predictions",
                        help="Directory the server writes to (VT_PREDICTION_LOG_DIR)")
    parser.add_argument("--output-dir", default="reports/drift")
    parser.add_argument("--since", help="Only use traffic after this: ISO time, or 30m / 2h / 1d")
    parser.add_argument("--drift-share", type=float, default=DEFAULT_DRIFT_SHARE,
                        help="Share of drifted columns that counts as dataset drift")
    parser.add_argument("--min-windows", type=int, default=30,
                        help="Minimum scored windows needed to run the test")
    parser.add_argument("--max-rejection-rate", type=float, default=0.05,
                        help="Alert when more of the windows than this were rejected as invalid")
    parser.add_argument("--no-mlflow", action="store_true", help="Don't record an MLflow run")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="Exit with code 1 on drift or a rejection-rate alert")
    parser.add_argument("--fail-on-insufficient-data", action="store_true",
                        help="Also exit with code 1 when there are fewer than --min-windows "
                             "scored windows (when traffic is expected, e.g. it was just "
                             "sent, an empty log means logging or serving is broken)")
    args = parser.parse_args(argv)

    since = parse_since(args.since) if args.since else None
    reference = pd.read_csv(args.reference)
    scored, rejected = read_prediction_logs(args.log_dir, since=since)

    summary = {
        "reference": args.reference,
        "log_dir": args.log_dir,
        "since": since.isoformat() if since else None,
        "n_reference": len(reference),
        "drift_share_threshold": args.drift_share,
        "min_windows": args.min_windows,
        "max_rejection_rate": args.max_rejection_rate,
        **traffic_summary(scored, rejected),
    }
    # Invalid input never reaches the drift test, so a surge of it is its own alarm.
    summary["rejection_alert"] = summary["rejection_rate"] > args.max_rejection_rate

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_HTML).unlink(missing_ok=True)  # don't leave a stale report behind
    if len(scored) < args.min_windows:
        summary["status"] = "insufficient_data"
    else:
        result, snapshot = detect_drift(reference, scored, args.drift_share)
        summary.update(result)
        summary["status"] = "drift" if result["dataset_drift"] else "no_drift"
        snapshot.save_html(str(output_dir / REPORT_HTML))
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    text = format_summary(summary)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)

    if not args.no_mlflow:
        with open(args.config, encoding="utf-8") as f:
            mlflow_config = yaml.safe_load(f).get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "sqlite:///mlflow_tracking.db"))
        run_id = log_to_mlflow(
            summary, output_dir, run_name=f"drift-{datetime.now():%Y%m%d-%H%M%S}"
        )
        logger.info(f"Recorded in MLflow experiment '{MLFLOW_EXPERIMENT}', run {run_id}")

    alarm = args.fail_on_drift and (summary["status"] == "drift" or summary["rejection_alert"])
    no_data = args.fail_on_insufficient_data and summary["status"] == "insufficient_data"
    return 1 if alarm or no_data else 0


if __name__ == "__main__":
    sys.exit(main())
