"""Send one 10-second MIT-BIH window to a running model server and print the result.

Start a server first, for example:

    mlflow models serve -m "models:/vt-detector@champion" -p 5001 --env-manager local

then:

    python scripts/predict_request.py --record 119 --start 60
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter

import wfdb

SAMPLES = 3600  # 10 s at 360 Hz
VENTRICULAR = {"V", "[", "!", "F"}  # same beat codes the model was trained to flag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:5001/invocations")
    parser.add_argument("--dataset", default="data/MIT_dataset")
    parser.add_argument("--record", default="119", help="MIT-BIH record id, e.g. 100, 119")
    parser.add_argument("--start", type=float, default=0.0, help="Window start in seconds")
    args = parser.parse_args()

    path = f"{args.dataset}/{args.record}"
    record = wfdb.rdrecord(path)
    start = int(args.start * record.fs)
    window = record.p_signal[start:start + SAMPLES, :2]
    if window.shape[0] != SAMPLES:
        sys.exit(f"Record {args.record} has no full 10 s window starting at {args.start}s.")

    # What the cardiologists annotated in this window, for comparison.
    beats = Counter(wfdb.rdann(path, "atr", sampfrom=start, sampto=start + SAMPLES).symbol)
    ventricular = sum(n for symbol, n in beats.items() if symbol in VENTRICULAR)

    request = urllib.request.Request(
        args.url,
        data=json.dumps({"inputs": [window.tolist()]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            probability = json.load(response)["predictions"][0]
    except urllib.error.HTTPError as err:
        sys.exit(f"Server rejected the request ({err.code}): {err.read().decode()}")

    print(f"Record {args.record}, {args.start:g}-{args.start + 10:g} s")
    print(f"  Annotated beats:      {dict(beats)}  (ventricular: {ventricular})")
    print(f"  P(VT/PVC) from model: {probability:.3f}")


if __name__ == "__main__":
    main()
