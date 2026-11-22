"""
Retraining decision: retrain only on drift that repeats
-------------------------------------------------------
Called by the drift workflow (.github/workflows/drift.yml) after a check finds
drift. Retraining is expensive and a single drifted check can be a fluke (with
the small 1% reference, about 1 clean check in 20), so it retrains only when
the previous check also found drift, on the same model, with the same kind of
traffic.

That also rate-limits it: a retrained model is a different model, so it has to
drift twice before the next retrain, and a rejection-rate alert or broken
logging (bad input or a broken server, which a new model can't fix) never
triggers one.

Each drift run saves ``context.json`` next to ``drift.json``:

    {"model": {...bundle.json: training_run_id, version, ...},
     "traffic": {"drift": "none", "strength": "", ...}}

Usage (in CI):
    python -m src.monitoring.retrain --current reports/drift --previous prev/reports/drift

Prints the decision and, in GitHub Actions, sets the step outputs
``retrain`` (true/false) and ``reason``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_check(directory: str | Path | None) -> dict | None:
    """drift.json merged with context.json from a drift report directory, or None."""
    if directory is None:
        return None
    directory = Path(directory)
    try:
        check = json.loads((directory / "drift.json").read_text(encoding="utf-8"))
        context = json.loads((directory / "context.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {**check, **context}


def should_retrain(current: dict, previous: dict | None) -> tuple[bool, str]:
    """Decide whether to retrain; returns (retrain, reason)."""
    if current.get("status") != "drift":
        return False, f"current check is '{current.get('status')}', not drift"
    if previous is None:
        return False, "no earlier check to compare with (first drift seen)"
    if previous.get("status") != "drift":
        return False, f"previous check was '{previous.get('status')}': drift not repeated yet"
    current_model = current.get("model", {}).get("training_run_id")
    if current_model != previous.get("model", {}).get("training_run_id"):
        return False, "previous drift was on a different model"
    if current.get("traffic", {}).get("drift") != previous.get("traffic", {}).get("drift"):
        return False, "previous drift was on a different kind of traffic"
    return True, f"drift in two consecutive checks of model run {current_model}"


def main(argv: list[str] | None = None) -> bool:
    parser = argparse.ArgumentParser(description="Decide whether repeated drift needs retraining")
    parser.add_argument("--current", required=True, help="This check's report directory")
    parser.add_argument("--previous", help="The previous check's report directory, if any")
    args = parser.parse_args(argv)

    current = load_check(args.current)
    if current is None:
        retrain, reason = False, "current check has no drift.json/context.json"
    else:
        retrain, reason = should_retrain(current, load_check(args.previous))

    print(f"Retrain: {'yes' if retrain else 'no'} ({reason})")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"retrain={str(retrain).lower()}\nreason={reason}\n")
    return retrain


if __name__ == "__main__":
    main()
