"""Send a stream of MIT-BIH windows to a running model server, optionally distorted.

Generates traffic for the drift monitor. Start a server with prediction
logging switched on, for example (PowerShell):

    $env:VT_PREDICTION_LOG_DIR = "monitoring/predictions"
    mlflow models serve -m "models:/vt-detector@challenger" -p 5001 --env-manager local

then send in-distribution traffic, and traffic with a simulated problem:

    python scripts/simulate_traffic.py --windows 150
    python scripts/simulate_traffic.py --windows 150 --drift noise

Windows are centred on randomly chosen annotated beats from the given records,
the same way the training data was cut, so --drift none looks like the
reference profile. Drift modes:

    gain      amplifier gain changed: signal x strength (default 1.5)
    noise     muscle/electrical noise: + Gaussian noise, sd = strength mV (default 0.05)
    baseline  baseline wander: + 0.3 Hz sine, amplitude = strength mV (default 0.5)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import wfdb

SAMPLES = 3600  # 10 s at 360 Hz
FS = 360
DEFAULT_STRENGTH = {"none": 0.0, "gain": 1.5, "noise": 0.05, "baseline": 0.5}


def sample_windows(dataset: Path, records: list[str], n: int, rng) -> np.ndarray:
    """n beat-centred 10 s windows, picked uniformly over all beats in the records."""
    signals, centres = {}, []
    for record in records:
        path = str(dataset / record)
        signals[record] = wfdb.rdrecord(path).p_signal[:, :2]
        beats = wfdb.rdann(path, "atr").sample
        length = signals[record].shape[0]
        # Same rule as the training loader: window starts half a window before
        # the beat, and incomplete windows at either end are skipped.
        starts = np.maximum(beats - SAMPLES // 2, 0)
        centres += [(record, s) for s in starts if s + SAMPLES <= length]
    picks = rng.choice(len(centres), size=n, replace=len(centres) < n)
    return np.stack([
        signals[centres[i][0]][centres[i][1]:centres[i][1] + SAMPLES] for i in picks
    ]).astype(np.float32)


def apply_drift(windows: np.ndarray, mode: str, strength: float, rng) -> np.ndarray:
    if mode == "gain":
        return windows * strength
    if mode == "noise":
        return windows + rng.normal(0.0, strength, windows.shape)
    if mode == "baseline":
        t = np.arange(SAMPLES) / FS
        phase = rng.uniform(0, 2 * np.pi, (len(windows), 1, 1))
        return windows + strength * np.sin(2 * np.pi * 0.3 * t[None, :, None] + phase)
    return windows


def post(url: str, batch: np.ndarray) -> tuple[int, list[float] | str]:
    """Send one batch; returns (HTTP status, predictions or error message)."""
    request = urllib.request.Request(
        url,
        data=json.dumps({"inputs": batch.tolist()}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.load(response)["predictions"]
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:5001/invocations")
    parser.add_argument("--dataset", default="data/MIT_dataset")
    parser.add_argument("--records", nargs="+",
                        help="MIT-BIH record ids to draw from (default: all in --dataset)")
    parser.add_argument("--windows", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--drift", choices=list(DEFAULT_STRENGTH), default="none")
    parser.add_argument("--strength", type=float, help="Size of the drift (see above)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    records = args.records or sorted(p.stem for p in dataset.glob("*.hea"))
    strength = args.strength if args.strength is not None else DEFAULT_STRENGTH[args.drift]
    rng = np.random.default_rng(args.seed)

    windows = sample_windows(dataset, records, args.windows, rng)
    windows = apply_drift(windows, args.drift, strength, rng)
    print(f"Sending {len(windows)} windows from {len(records)} records "
          f"(drift: {args.drift}{f', strength {strength:g}' if args.drift != 'none' else ''}) "
          f"to {args.url}")

    predictions, rejected, errors = [], 0, []
    for start in range(0, len(windows), args.batch_size):
        batch = windows[start:start + args.batch_size]
        status, body = post(args.url, batch)
        if status == 200:
            predictions += body
        elif status == 400:
            rejected += len(batch)
            errors.append(body)
        else:
            sys.exit(f"Server error {status}: {body[:500]}")

    print(f"  scored:   {len(predictions)} windows, mean P(VT/PVC) "
          f"{np.mean(predictions) if predictions else float('nan'):.3f}")
    print(f"  rejected: {rejected} windows (HTTP 400)")
    if errors:
        print(f"  first rejection: {errors[0][:200]}")


if __name__ == "__main__":
    main()
