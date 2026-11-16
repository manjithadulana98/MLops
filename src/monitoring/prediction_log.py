"""
Prediction logging for the served model
---------------------------------------
When ``VT_PREDICTION_LOG_DIR`` is set in the serving environment, the model
appends one JSON line per scored window (timestamp, summary features,
predicted probability) and one line per rejected request. Raw signals are
not stored: the features are enough for drift detection and far smaller.

Each server process writes its own file (``predictions-<pid>.jsonl``), so
several workers never interleave writes. Files are opened per write, so they
can be rotated or deleted while the server runs.

Logging never breaks a prediction: if a write fails, a warning is logged and
the response is returned as normal.

IEC 62304 Traceability
    Software Unit  : SU-MON-02
    Requirement Ref: SRS-MON-001 (Post-deployment input monitoring)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.monitoring.features import compute_features

logger = logging.getLogger(__name__)

LOG_DIR_ENV = "VT_PREDICTION_LOG_DIR"
SCORED = "scored"
REJECTED = "rejected"


class PredictionLogger:
    """Appends prediction records as JSON lines under ``log_dir``."""

    def __init__(self, log_dir: str | Path, model_run_id: str | None = None) -> None:
        self.log_dir = Path(log_dir)
        self.model_run_id = model_run_id

    @classmethod
    def from_env(cls, model_run_id: str | None = None) -> PredictionLogger | None:
        """A logger writing to ``$VT_PREDICTION_LOG_DIR``, or None if it isn't set."""
        log_dir = os.environ.get(LOG_DIR_ENV)
        return cls(log_dir, model_run_id) if log_dir else None

    def log_scored(self, signals: np.ndarray, probabilities: np.ndarray) -> None:
        """Record one line per scored window: its summary features and prediction."""
        try:
            features = compute_features(signals)
            now = _utc_now()
            lines = []
            for i, probability in enumerate(np.asarray(probabilities, dtype=float).ravel()):
                record = {
                    "timestamp": now,
                    "status": SCORED,
                    "model_run_id": self.model_run_id,
                    "prediction": probability,
                }
                record.update({name: float(values[i]) for name, values in features.items()})
                lines.append(json.dumps(record))
        except Exception as err:  # monitoring must not take down serving
            logger.warning(f"Could not compute prediction log record: {err}")
            return
        self._append(lines)

    def log_rejected(self, n_windows: int, reason: str) -> None:
        """Record a request the model refused to score (bad shape or clinical validation)."""
        record = {
            "timestamp": _utc_now(),
            "status": REJECTED,
            "model_run_id": self.model_run_id,
            "n_windows": int(n_windows),
            "reason": reason,
        }
        self._append([json.dumps(record)])

    def _append(self, lines: list[str]) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"predictions-{os.getpid()}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write("".join(line + "\n" for line in lines))
        except OSError as err:
            logger.warning(f"Could not write prediction log to {self.log_dir}: {err}")


def read_prediction_logs(log_dir: str | Path, since: datetime | None = None):
    """Load every ``*.jsonl`` file under ``log_dir``.

    Parameters
    ----------
    log_dir : str | Path
        Directory the server wrote to (``VT_PREDICTION_LOG_DIR``).
    since : datetime, optional
        Keep only records at or after this time (timezone-aware).

    Returns
    -------
    scored, rejected : pandas.DataFrame
        One row per scored window, and one row per rejected request.
    """
    import pandas as pd  # not needed on the serving path

    records = []
    for path in sorted(Path(log_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if since is not None:
        frame = frame[frame["timestamp"] >= pd.Timestamp(since)]

    scored = frame[frame["status"] == SCORED].reset_index(drop=True)
    rejected = frame[frame["status"] == REJECTED].reset_index(drop=True)
    return scored, rejected


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
